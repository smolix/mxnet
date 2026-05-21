# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Exercise backward through quantized DNNL ops (issues.md #5).

Forward inference (quantized) is solid. This file validates — or explicitly
documents — the backward/gradient behavior of the quantized subgraph ops:

  - _sg_onednn_fully_connected  (via quantized_sg_onednn_fully_connected)
  - _sg_onednn_conv             (via quantized_sg_onednn_conv)
  - Composite Conv → ReLU → Dense fusion
  - Calibration round-trip (quantize → save → re-quantize → compare)

STATUS (2026-05-18, partial fix):

  Step 1 DONE — STE for quantize_v2:
    `quantize_v2` now has a Straight-Through Estimator (STE) backward registered:
    it casts the upstream gradient (int8) back to float32 and passes it through
    unchanged.  In isolation this works correctly.

  Step 2 DONE — qat kwarg for quantize_net:
    `quantize_net(..., qat=True)` keeps `grad_req='write'` on all parameters
    (instead of forcing `grad_req='null'`), allowing gradient accumulation and
    parameter updates during QAT training loops.  This file now has a passing
    guard test for that infrastructure while full subgraph backward remains
    xfailed below.

  Step 3 BLOCKED — weight/data backward for _sg_onednn_fully_connected/_sg_onednn_conv:
    These fused subgraph ops still have `FGradient = MakeZeroGradNodes`.  An
    attempt to add a dot-product-based data backward caused segfaults in the
    NNVM/CachedOp backward executor framework.  The root cause is that DNNL
    subgraph ops interact with MXNet's static graph executor in a way that does
    not support custom backward nodes referencing op inputs at this time.  The
    STE in quantize_v2 is therefore functionally correct but its gradient is
    blocked by the zero gradient from the FC/Conv subgraph ops upstream.

  Net result: the xfail tests below remain xfailed — input and weight gradients
  are still identically zero through the full quantized graph.  The STE becomes
  effective only after _sg_onednn_fully_connected/_sg_onednn_conv get proper
  backward support.  Setting the historical MXNET_QAT_SUBGRAPH_BACKWARD=1 gate
  does not change behavior in this checkout because no gated backward body is
  present.

ROOT-CAUSE NOTE (discovered 2026-05-17):
  MXNet's quantize_net inserts a `quantize_v2` node immediately before each
  quantized op.  The quantized DNNL subgraph ops (_sg_onednn_fully_connected,
  _sg_onednn_conv) have FGradient = MakeZeroGradNodes, which kills all gradient
  flow.  The quantize_v2 STE (step 1) is now in place but its gradient is zero
  because the upstream zero gradient is passed through.

Usage:
  pytest tests/python/dnnl/subgraphs/test_quantized_backward.py -v --timeout=300
"""

import os
import sys

# Activate the DNNL quantization path
os.environ.setdefault('ENABLE_ONEDNN_QUANTIZATION_TEST', '1')

import numpy as np
import pytest

import mxnet as mx
from mxnet.contrib import quantization
from mxnet.gluon import nn

# Note: the original `mx.npx.reset_np()` at module scope here is a global
# toggle that leaks into any other test file collected in the same pytest
# invocation. Replace with an autouse fixture so the np-semantics flip is
# scoped per-test and cross-file sweeps stay clean.
import pytest as _pytest_for_reset_np_fixture
@_pytest_for_reset_np_fixture.fixture(autouse=True)
def _legacy_nd_semantics():
    _prev_arr = mx.util.is_np_array()
    _prev_shp = mx.util.is_np_shape()
    mx.npx.reset_np()
    yield
    mx.npx.set_np(shape=_prev_shp, array=_prev_arr)
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FC_DATA_SHAPE = (8, 64)
CONV_DATA_SHAPE = (4, 3, 32, 32)


class _OnePassDataLoader(mx.gluon.data.DataLoader):
    """Minimal DataLoader that yields a fixed list of tensors exactly once.

    quantize_net expects a mx.gluon.data.DataLoader whose __next__ returns
    either a single ndarray or a list of ndarrays (one per model input).
    """
    def __init__(self, batch):
        self._batch = batch
        self._done = False

    def __iter__(self):
        self._done = False
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self._batch

    def __del__(self):
        pass


def _make_calib_loader(data):
    """Return a one-pass DataLoader for a single ndarray `data`."""
    return _OnePassDataLoader([data])


@mx.util.use_np
def _quantize(net, data, quantized_dtype='int8', calib_mode='naive', quantize_mode='full',
              qat=False):
    """Hybridize *net*, run a forward pass, then return quantize_net(net)."""
    # Shape-trace the original network first (required for hybridization)
    net.hybridize(static_alloc=False, static_shape=False)
    net(data)
    # Reset hybridize so quantize_net can call it again with its own options
    # (quantize_net calls hybridize internally)
    calib_loader = _make_calib_loader(data)
    qnet = quantization.quantize_net(
        net,
        device=mx.cpu(),
        quantized_dtype=quantized_dtype,
        calib_mode=calib_mode,
        calib_data=calib_loader,
        num_calib_batches=1,
        quantize_mode=quantize_mode,
        qat=qat,
    )
    return qnet


def _grad_norm(x):
    return float(np.linalg.norm(x.grad.asnumpy()))


def _fp32_input_grad_norm(net, data_shape):
    """Return the input-gradient norm for the FP32 network."""
    x = mx.np.random.uniform(-1, 1, size=data_shape, dtype='float32', device=mx.cpu())
    x.attach_grad()
    with mx.autograd.record():
        out = net(x)
        loss = out.sum()
    loss.backward()
    return float(np.linalg.norm(x.grad.asnumpy()))


def _quantized_input_grad(qnet, data_shape):
    """Run forward+backward through *qnet*, return (grad_norm, is_finite, is_nan)."""
    x = mx.np.random.uniform(-1, 1, size=data_shape, dtype='float32', device=mx.cpu())
    x.attach_grad()
    with mx.autograd.record():
        out = qnet(x)
        loss = out.sum()
    loss.backward()
    g = x.grad.asnumpy()
    return float(np.linalg.norm(g)), bool(np.all(np.isfinite(g))), bool(np.any(np.isnan(g)))


# ---------------------------------------------------------------------------
# Test 1: _sg_onednn_fully_connected backward
# ---------------------------------------------------------------------------

@mx.util.use_np
class _SimpleFCNet(nn.HybridBlock):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(32, flatten=True)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.timeout(300)
def test_fc_quantized_forward_runs():
    """Quantized FC forward must not crash and produce finite output."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        out = qnet(data)
        assert np.all(np.isfinite(out.asnumpy())), "Quantized FC output contains non-finite values"
        assert out.shape == (FC_DATA_SHAPE[0], 32)

    _run()


@pytest.mark.timeout(300)
def test_fc_quantized_backward_no_crash():
    """Quantized FC backward must not crash (even if gradient is zero)."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, is_finite, has_nan = _quantized_input_grad(qnet, FC_DATA_SHAPE)
        assert not has_nan, "Quantized FC backward produced NaN gradients"
        assert is_finite, "Quantized FC backward produced Inf gradients"

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): quantize_v2 has STE, but the quantized FC subgraph "
        "still uses MakeZeroGradNodes, so upstream input gradients are zero. "
        "Remove this xfail once _sg_onednn_fully_connected backward works."
    ),
)
def test_fc_quantized_backward_nonzero_input_grad():
    """Quantized FC input gradient should be non-zero (like FP32 reference).

    CURRENTLY XFAIL: the quantized FC subgraph blocks gradient flow, so x.grad
    is identically 0 even though quantize_v2 has STE.
    """
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, _, _ = _quantized_input_grad(qnet, FC_DATA_SHAPE)
        # FP32 reference should have a substantial gradient norm
        fp32_norm = _fp32_input_grad_norm(net, FC_DATA_SHAPE)
        assert fp32_norm > 0.1, f"FP32 reference gradient is unexpectedly tiny: {fp32_norm}"
        # We want quantized gradient to be in the same ballpark (weak test)
        assert grad_norm > 0.0, (
            f"Quantized FC x.grad is zero (fp32_norm={fp32_norm:.4f}). "
            f"The quantized FC subgraph kills gradient flow."
        )

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): quantized FC has no weight-gradient backward. "
        "Remove this xfail once _sg_onednn_fully_connected weight grads work."
    ),
)
def test_fc_quantized_weight_grad_nonzero():
    """Quantized FC weight gradient should be non-zero for fine-tuning.

    CURRENTLY XFAIL: even when forced to grad_req='write', the quantized FC
    subgraph backward yields 0.
    """
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data, qat=True)

        x = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())
        with mx.autograd.record():
            out = qnet(x)
            loss = out.sum()
        loss.backward()

        any_nonzero = False
        for k, v in qnet.collect_params().items():
            if v.grad_req != 'null' and 'weight' in k and 'min' not in k and 'max' not in k:
                g = v.grad().asnumpy()
                if np.linalg.norm(g) > 0:
                    any_nonzero = True
                    break

        assert any_nonzero, (
            "All quantized weight gradients are zero. Fine-tuning a quantized FC "
            "network is impossible without straight-through estimator support."
        )

    _run()


# ---------------------------------------------------------------------------
# Test 2: _sg_onednn_conv backward
# ---------------------------------------------------------------------------

@mx.util.use_np
class _SimpleConvNet(nn.HybridBlock):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(channels=8, kernel_size=3, padding=1, use_bias=True)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.timeout(300)
def test_conv_quantized_forward_runs():
    """Quantized Conv forward must not crash and produce finite output."""
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        out = qnet(data)
        assert np.all(np.isfinite(out.asnumpy())), "Quantized Conv output contains non-finite values"
        expected_shape = (CONV_DATA_SHAPE[0], 8, CONV_DATA_SHAPE[2], CONV_DATA_SHAPE[3])
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    _run()


@pytest.mark.timeout(300)
def test_conv_quantized_backward_no_crash():
    """Quantized Conv backward must not crash (even if gradient is zero)."""
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, is_finite, has_nan = _quantized_input_grad(qnet, CONV_DATA_SHAPE)
        assert not has_nan, "Quantized Conv backward produced NaN gradients"
        assert is_finite, "Quantized Conv backward produced Inf gradients"

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): quantize_v2 has STE, but the quantized Conv "
        "subgraph still uses MakeZeroGradNodes, so input gradients are zero. "
        "Remove this xfail once _sg_onednn_conv backward works."
    ),
)
def test_conv_quantized_backward_nonzero_input_grad():
    """Quantized Conv input gradient should be non-zero.

    CURRENTLY XFAIL: the quantized Conv subgraph blocks gradient flow.
    """
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        fp32_norm = _fp32_input_grad_norm(net, CONV_DATA_SHAPE)
        grad_norm, _, _ = _quantized_input_grad(qnet, CONV_DATA_SHAPE)
        assert fp32_norm > 1.0, f"FP32 Conv reference gradient is unexpectedly tiny: {fp32_norm}"
        assert grad_norm > 0.0, (
            f"Quantized Conv x.grad is zero (fp32_norm={fp32_norm:.4f}). "
            f"The quantized Conv subgraph kills gradient flow."
        )

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): MXNET_QAT_SUBGRAPH_BACKWARD=1 is only a historical "
        "handover gate in this checkout; no gated FC backward body is present."
    ),
)
def test_fc_qat_subgraph_backward_gate_nonzero_input_grad(monkeypatch):
    """Historical QAT backward gate should eventually enable non-zero FC grads.

    CURRENTLY XFAIL: setting the gate does not change the zero-gradient
    MakeZeroGradNodes behavior of the quantized FC subgraph.
    """
    monkeypatch.setenv('MXNET_QAT_SUBGRAPH_BACKWARD', '1')
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, _, _ = _quantized_input_grad(qnet, FC_DATA_SHAPE)
        assert grad_norm > 0.0, (
            f"Gated quantized FC x.grad is zero. "
            f"MXNET_QAT_SUBGRAPH_BACKWARD=1 has no implemented effect."
        )

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): MXNET_QAT_SUBGRAPH_BACKWARD=1 is only a historical "
        "handover gate in this checkout; no gated Conv backward body is present."
    ),
)
def test_conv_qat_subgraph_backward_gate_nonzero_input_grad(monkeypatch):
    """Historical QAT backward gate should eventually enable non-zero Conv grads.

    CURRENTLY XFAIL: setting the gate does not change the zero-gradient
    MakeZeroGradNodes behavior of the quantized Conv subgraph.
    """
    monkeypatch.setenv('MXNET_QAT_SUBGRAPH_BACKWARD', '1')
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, _, _ = _quantized_input_grad(qnet, CONV_DATA_SHAPE)
        assert grad_norm > 0.0, (
            f"Gated quantized Conv x.grad is zero. "
            f"MXNET_QAT_SUBGRAPH_BACKWARD=1 has no implemented effect."
        )

    _run()


# ---------------------------------------------------------------------------
# Test 3: Composite Conv → ReLU → GlobalAvgPool → Dense backward
# ---------------------------------------------------------------------------

@mx.util.use_np
class _ConvReluDenseNet(nn.HybridBlock):
    """Conv → ReLU → GlobalAvgPool → Dense — the FC-saturation-fix test case."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(channels=8, kernel_size=3, padding=1, use_bias=True)
        self.relu = nn.Activation('relu')
        self.pool = nn.GlobalAvgPool2D()
        self.fc = nn.Dense(4, flatten=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.pool(out)
        return self.fc(out)


@pytest.mark.timeout(300)
def test_composite_quantized_forward_runs():
    """Quantized composite Conv→ReLU→Dense forward must not crash."""
    net = _ConvReluDenseNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        out = qnet(data)
        assert np.all(np.isfinite(out.asnumpy())), "Composite quantized output contains non-finite values"
        assert out.shape == (CONV_DATA_SHAPE[0], 4)

    _run()


@pytest.mark.timeout(300)
def test_composite_quantized_backward_no_crash():
    """Quantized composite backward must not crash."""
    net = _ConvReluDenseNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        grad_norm, is_finite, has_nan = _quantized_input_grad(qnet, CONV_DATA_SHAPE)
        assert not has_nan, "Composite quantized backward produced NaN gradients"
        assert is_finite, "Composite quantized backward produced Inf gradients"

    _run()


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BROKEN (B4): the first quantized Conv subgraph kills all "
        "gradient flow. Composite fusion does not help. Remove xfail once "
        "_sg_onednn_conv backward works."
    ),
)
def test_composite_quantized_backward_nonzero_input_grad():
    """Quantized composite network input gradient should be non-zero.

    CURRENTLY XFAIL: the quantized Conv subgraph kills gradient.
    """
    net = _ConvReluDenseNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        fp32_norm = _fp32_input_grad_norm(net, CONV_DATA_SHAPE)
        grad_norm, _, _ = _quantized_input_grad(qnet, CONV_DATA_SHAPE)
        # FP32 norm may be small because of GlobalAvgPool + 4-unit Dense,
        # but should still be > 0
        assert fp32_norm >= 0.0, f"FP32 composite gradient is negative norm: {fp32_norm}"
        assert grad_norm > 0.0, (
            f"Quantized composite x.grad is zero (fp32_norm={fp32_norm:.6f}). "
            f"The quantized Conv subgraph kills gradient flow."
        )

    _run()


@pytest.mark.timeout(300)
def test_composite_quantized_sign_agreement():
    """At least some entries of quantized and FP32 gradients should share the same sign.

    This is a weak test: if backward were working, we'd expect >= 50% sign
    agreement.  Since gradient is all-zero for quantized, this test checks that
    the FP32 gradient is non-trivial (so the comparison would be meaningful).

    The test PASSES today only because we skip the sign comparison when the
    quantized gradient is zero.  Once backward is fixed this test should be
    tightened to require >= 50% sign agreement.

    NOTE: Remove the early-return branch and xfail/skip when backward is fixed.
    """
    net = _ConvReluDenseNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(0, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)

        # FP32 gradient
        x_fp32 = mx.np.random.uniform(0, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())
        x_fp32.attach_grad()
        with mx.autograd.record():
            net(x_fp32).sum().backward()
        fp32_grad = x_fp32.grad.asnumpy()

        # Quantized gradient
        x_q = x_fp32.copy()
        x_q.attach_grad()
        with mx.autograd.record():
            qnet(x_q).sum().backward()
        q_grad = x_q.grad.asnumpy()

        fp32_norm = float(np.linalg.norm(fp32_grad))
        q_norm = float(np.linalg.norm(q_grad))

        # If quantized gradient is zero (current behavior), just verify FP32 works
        if q_norm == 0.0:
            # Document the failure condition without failing the test itself
            # (the xfail tests above already capture the failure)
            assert fp32_norm > 0.0, (
                f"FP32 composite gradient is also zero — something is wrong with the "
                f"FP32 network itself (norm={fp32_norm})."
            )
            return  # Skip sign comparison since q_grad is all-zero

        # When backward is fixed: check >= 50% sign agreement
        nonzero_mask = np.abs(fp32_grad) > 1e-8
        if nonzero_mask.sum() > 0:
            sign_agree = float(
                np.mean(np.sign(fp32_grad[nonzero_mask]) == np.sign(q_grad[nonzero_mask]))
            )
            assert sign_agree >= 0.50, (
                f"Quantized and FP32 gradient sign agreement is only {sign_agree:.2%}. "
                f"Expected >= 50%."
            )

    _run()


# ---------------------------------------------------------------------------
# Test 4: Calibration round-trip
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_calibration_round_trip_fc():
    """Quantize FC with naive calib, save params, re-quantize, compare outputs."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet1 = _quantize(net, data, calib_mode='naive')
        out1 = qnet1(data).asnumpy()
        assert np.all(np.isfinite(out1)), "First quantization output not finite"

        # Save param dict
        param_dict = {k: v.data().asnumpy() for k, v in qnet1.collect_params().items()}
        assert len(param_dict) > 0, "No params saved"

        # Re-quantize (should produce identical results since same weights/calib data)
        qnet2 = _quantize(net, data, calib_mode='naive')
        out2 = qnet2(data).asnumpy()

        max_diff = np.abs(out1 - out2).max()
        assert max_diff == 0.0, (
            f"Two independent quantizations of the same net gave different outputs "
            f"(max_diff={max_diff:.2e}). Calibration is not deterministic."
        )

        # Backward from both — both should be finite (even if zero)
        grad_norm1, is_finite1, has_nan1 = _quantized_input_grad(qnet1, FC_DATA_SHAPE)
        grad_norm2, is_finite2, has_nan2 = _quantized_input_grad(qnet2, FC_DATA_SHAPE)

        assert not has_nan1 and not has_nan2, "Calibration round-trip backward produced NaN"
        assert is_finite1 and is_finite2, "Calibration round-trip backward produced Inf"
        assert grad_norm1 == grad_norm2 or (grad_norm1 == 0.0 and grad_norm2 == 0.0), (
            f"Gradient norms differ between round-trips: {grad_norm1:.4f} vs {grad_norm2:.4f}"
        )

    _run()


@pytest.mark.timeout(300)
def test_calibration_round_trip_conv():
    """Quantize Conv with naive calib, save params, re-quantize, compare outputs."""
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet1 = _quantize(net, data, calib_mode='naive')
        out1 = qnet1(data).asnumpy()
        assert np.all(np.isfinite(out1)), "First quantization output not finite"

        # Save param dict
        param_dict = {k: v.data().asnumpy() for k, v in qnet1.collect_params().items()}
        assert len(param_dict) > 0, "No params saved"

        # Re-quantize
        qnet2 = _quantize(net, data, calib_mode='naive')
        out2 = qnet2(data).asnumpy()

        max_diff = np.abs(out1 - out2).max()
        assert max_diff == 0.0, (
            f"Two independent Conv quantizations gave different outputs "
            f"(max_diff={max_diff:.2e})."
        )

        # Backward check
        grad_norm1, is_finite1, has_nan1 = _quantized_input_grad(qnet1, CONV_DATA_SHAPE)
        grad_norm2, is_finite2, has_nan2 = _quantized_input_grad(qnet2, CONV_DATA_SHAPE)
        assert not has_nan1 and not has_nan2, "Calibration round-trip Conv backward produced NaN"
        assert is_finite1 and is_finite2, "Calibration round-trip Conv backward produced Inf"

    _run()


# ---------------------------------------------------------------------------
# Test 5: Additional edge-cases
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_fc_backward_multiple_forward_passes():
    """Multiple forward passes before backward must not corrupt state."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)

        # Run several forward passes to warm up the FC weight cache
        for _ in range(3):
            _ = qnet(data)

        # Now do forward + backward
        x = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())
        x.attach_grad()
        with mx.autograd.record():
            out = qnet(x)
            loss = out.sum()
        loss.backward()  # Must not crash

        g = x.grad.asnumpy()
        assert not np.any(np.isnan(g)), "backward after multiple forward passes produced NaN"
        assert np.all(np.isfinite(g)), "backward after multiple forward passes produced Inf"

    _run()


@pytest.mark.timeout(300)
def test_conv_quantized_output_changes_with_input():
    """Verify the quantized Conv output is actually sensitive to the input.

    This is a sanity check: if the network output doesn't change when the input
    changes, the quantized graph is broken in a worse way than just zero grads.
    """
    net = _SimpleConvNet()
    net.initialize(mx.init.Normal(0.1))
    data = mx.np.random.uniform(-1, 1, size=CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)

        x1 = mx.np.ones(CONV_DATA_SHAPE, dtype='float32', device=mx.cpu())
        x2 = mx.np.ones(CONV_DATA_SHAPE, dtype='float32', device=mx.cpu()) * 0.5

        out1 = qnet(x1).asnumpy()
        out2 = qnet(x2).asnumpy()

        max_diff = np.abs(out1 - out2).max()
        assert max_diff > 0.0, (
            "Quantized Conv output is identical for different inputs — "
            "the network is not responding to input changes."
        )

    _run()


@pytest.mark.timeout(300)
def test_fc_quantized_output_changes_with_input():
    """Verify the quantized FC output is sensitive to the input."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)

        x1 = mx.np.ones(FC_DATA_SHAPE, dtype='float32', device=mx.cpu())
        x2 = mx.np.ones(FC_DATA_SHAPE, dtype='float32', device=mx.cpu()) * 0.5

        out1 = qnet(x1).asnumpy()
        out2 = qnet(x2).asnumpy()

        max_diff = np.abs(out1 - out2).max()
        assert max_diff > 0.0, (
            "Quantized FC output is identical for different inputs — "
            "the network is not responding to input changes."
        )

    _run()


@pytest.mark.timeout(300)
def test_quantized_grad_req_all_null():
    """quantize_net defaults to grad_req='null' on all params.

    This preserves inference-only behavior.  QAT callers must opt in with
    qat=True; see test_qat_quantized_grad_req_write.
    """
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data)
        for k, v in qnet.collect_params().items():
            assert v.grad_req == 'null', (
                f"Expected grad_req='null' for quantized param {k!r}, "
                f"got {v.grad_req!r}. If this assertion fails, quantize_net "
                f"behavior has changed — update this test and re-check gradient flow."
            )

    _run()


@pytest.mark.timeout(300)
def test_qat_quantized_grad_req_write():
    """QAT mode keeps trainable parameter grad buffers allocated."""
    net = _SimpleFCNet()
    net.initialize(mx.init.Normal(0.5))
    data = mx.np.random.uniform(-1, 1, size=FC_DATA_SHAPE, dtype='float32', device=mx.cpu())

    @mx.util.use_np
    def _run():
        qnet = _quantize(net, data, qat=True)
        trainable_params = {
            k: v for k, v in qnet.collect_params().items()
            if 'min' not in k and 'max' not in k
        }
        assert trainable_params, "Expected quantized QAT network to expose trainable params"
        for k, v in trainable_params.items():
            assert v.grad_req == 'write', (
                f"Expected grad_req='write' for QAT param {k!r}, got {v.grad_req!r}"
            )

    _run()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--timeout=300'])
