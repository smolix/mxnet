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

"""Regression test for Apache MXNet issue #18751.

BatchNorm running_mean and running_var were swapped on GPU (cuDNN path)
relative to CPU. The fix also aligns when the running stats are updated:
both CPU (DNNL) and GPU (cuDNN) now update them during the forward pass
so they are visible immediately after bn(x) in training mode.
"""

import pytest
import numpy as np
import mxnet as mx
import mxnet.numpy as mnp
from mxnet import gluon, autograd

requires_gpu = pytest.mark.skipif(
    mx.context.num_gpus() == 0, reason="requires GPU"
)


def _make_bn(ctx, momentum=0.9):
    bn = gluon.nn.BatchNorm(momentum=momentum)
    bn.initialize(ctx=ctx)
    return bn


# ---------------------------------------------------------------------------
# 1. Check that running_mean and running_var are NOT swapped
#    (the original bug: GPU produced mean=0, var=1 when it should be mean≈1, var≈0)
# ---------------------------------------------------------------------------

@requires_gpu
def test_running_stats_not_swapped_after_backward():
    """After fwd+bwd, CPU and GPU agree: running_mean≈1, running_var≈0."""
    data = np.ones((4, 3, 4, 4), dtype=np.float32)
    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0)  # momentum=0: fully replace with batch stats
        x = mnp.array(data, ctx=ctx)
        with autograd.record():
            y = bn(x)
        y.backward(mnp.ones_like(y))
        mx.nd.waitall()
        rm = bn.running_mean.data().asnumpy()
        rv = bn.running_var.data().asnumpy()
        np.testing.assert_allclose(rm, np.ones(3, dtype=np.float32), atol=1e-5,
                                   err_msg=f"{ctx}: running_mean should be 1 after all-ones input")
        np.testing.assert_allclose(rv, np.zeros(3, dtype=np.float32), atol=1e-4,
                                   err_msg=f"{ctx}: running_var should be 0 after all-ones input")


@requires_gpu
def test_running_stats_not_swapped_distinct_channels():
    """With distinct per-channel means, verify no mean/var swap on CPU or GPU."""
    # Channel 0: all 2.0, Channel 1: all 0.5, Channel 2: all -1.0
    data = np.zeros((4, 3, 4, 4), dtype=np.float32)
    data[:, 0, :, :] = 2.0
    data[:, 1, :, :] = 0.5
    data[:, 2, :, :] = -1.0
    expected_mean = np.array([2.0, 0.5, -1.0], dtype=np.float32)
    expected_var  = np.zeros(3, dtype=np.float32)

    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0)
        x = mnp.array(data, ctx=ctx)
        with autograd.record():
            y = bn(x)
        y.backward(mnp.ones_like(y))
        mx.nd.waitall()
        rm = bn.running_mean.data().asnumpy()
        rv = bn.running_var.data().asnumpy()
        np.testing.assert_allclose(rm, expected_mean, atol=1e-5,
                                   err_msg=f"{ctx}: running_mean wrong or swapped with var")
        np.testing.assert_allclose(rv, expected_var, atol=1e-4,
                                   err_msg=f"{ctx}: running_var wrong or swapped with mean")


# ---------------------------------------------------------------------------
# 2. Check that CPU and GPU agree (including after momentum update)
# ---------------------------------------------------------------------------

@requires_gpu
def test_cpu_gpu_running_stats_agree_after_backward():
    """CPU and GPU running_mean / running_var must agree after one fwd+bwd step."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 8, 8, 8)).astype(np.float32)

    results = {}
    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0.9)
        x = mnp.array(data, ctx=ctx)
        with autograd.record():
            y = bn(x)
        y.backward(mnp.ones_like(y))
        mx.nd.waitall()
        results[str(ctx)] = (
            bn.running_mean.data().asnumpy().copy(),
            bn.running_var.data().asnumpy().copy(),
        )

    np.testing.assert_allclose(results['cpu(0)'][0], results['gpu(0)'][0], atol=1e-4,
                               err_msg="running_mean: CPU vs GPU disagreement")
    # running_var tolerances are relaxed to 1e-3: DNNL (CPU) and cuDNN (GPU) use
    # slightly different variance estimators (biased vs Bessel-corrected) which
    # produces ~4e-4 differences on random inputs.
    np.testing.assert_allclose(results['cpu(0)'][1], results['gpu(0)'][1], atol=1e-3,
                               err_msg="running_var: CPU vs GPU disagreement")


# ---------------------------------------------------------------------------
# 3. Check that running stats are updated in the FORWARD pass (visible
#    immediately after bn(x), before any backward call).
# ---------------------------------------------------------------------------

@requires_gpu
def test_running_stats_updated_in_forward():
    """Running stats must be visible immediately after the forward pass."""
    data = np.ones((4, 3, 4, 4), dtype=np.float32)
    # With momentum=0, a single forward pass sets running_mean = batch_mean = 1
    # and running_var = batch_var = 0.
    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0)
        x = mnp.array(data, ctx=ctx)
        with autograd.record():
            y = bn(x)
        mx.nd.waitall()  # NO backward call
        rm = bn.running_mean.data().asnumpy()
        rv = bn.running_var.data().asnumpy()
        np.testing.assert_allclose(rm, np.ones(3, dtype=np.float32), atol=1e-5,
                                   err_msg=f"{ctx}: running_mean not updated in forward")
        np.testing.assert_allclose(rv, np.zeros(3, dtype=np.float32), atol=1e-4,
                                   err_msg=f"{ctx}: running_var not updated in forward")


# ---------------------------------------------------------------------------
# 4. Check that backward does not produce NaN (regression for transfer
#    learning divergence reported in issue #18751).
# ---------------------------------------------------------------------------

@requires_gpu
def test_no_nan_in_backward():
    """Backward pass should not produce NaN gradients."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 8, 8, 8)).astype(np.float32)

    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0.9)
        x = mnp.array(data, ctx=ctx)
        x.attach_grad()
        with autograd.record():
            y = bn(x)
        y.backward(mnp.ones_like(y))
        mx.nd.waitall()
        dx = x.grad.asnumpy()
        assert not np.any(np.isnan(dx)), f"{ctx}: NaN in input gradient"
        assert not np.any(np.isnan(bn.gamma.grad().asnumpy())), f"{ctx}: NaN in gamma gradient"
        assert not np.any(np.isnan(bn.beta.grad().asnumpy())), f"{ctx}: NaN in beta gradient"


# ---------------------------------------------------------------------------
# 5. Multi-step: running stats converge toward correct batch statistics
# ---------------------------------------------------------------------------

@requires_gpu
def test_running_stats_converge():
    """After many steps with same input, running_mean -> batch_mean, running_var -> 0."""
    data = np.ones((4, 3, 4, 4), dtype=np.float32) * 3.0  # all 3s

    for ctx in [mx.cpu(), mx.gpu(0)]:
        bn = _make_bn(ctx, momentum=0.9)
        x = mnp.array(data, ctx=ctx)
        for _ in range(50):
            with autograd.record():
                y = bn(x)
            y.backward(mnp.ones_like(y))
        mx.nd.waitall()
        rm = bn.running_mean.data().asnumpy()
        rv = bn.running_var.data().asnumpy()
        np.testing.assert_allclose(rm, np.full(3, 3.0, dtype=np.float32), atol=0.1,
                                   err_msg=f"{ctx}: running_mean did not converge to 3.0")
        np.testing.assert_allclose(rv, np.zeros(3, dtype=np.float32), atol=0.05,
                                   err_msg=f"{ctx}: running_var did not converge to 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
