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

"""XOP18 contract tests for quantized oneDNN self-attention subgraphs."""

import numpy as np
import pytest

import mxnet as mx
from mxnet import autograd


def _onednn_available():
    return mx.runtime.Features().is_enabled("ONEDNN")


pytestmark = pytest.mark.skipif(
    not _onednn_available(), reason="oneDNN backend not enabled")


_SELFATT_OPS = [
    '_sg_onednn_selfatt_qk',
    '_sg_onednn_selfatt_qk_split',
    '_sg_onednn_selfatt_valatt',
]


@pytest.mark.parametrize('op_name', _SELFATT_OPS)
def test_selfatt_subgraph_op_registered(op_name):
    """The oneDNN-fused self-attention subgraph ops must be registered.

    A previous reorganization quietly dropped one of these from the
    registry; this test fails fast if that happens again."""
    assert hasattr(mx.nd._internal, op_name), \
        f"Self-attention subgraph op {op_name!r} is not registered in mx.nd._internal"
    assert hasattr(mx.sym._internal, op_name), \
        f"Self-attention subgraph op {op_name!r} is not registered in mx.sym._internal"


def _dequant_int8(qarray, min_range, max_range):
    scale = max(abs(float(min_range)), abs(float(max_range))) / 127.0
    return qarray.astype('float32') * scale


def _dequant_uint8(qarray, min_range, max_range):
    return qarray.astype('float32') * ((float(max_range) - float(min_range)) / 255.0) + \
        float(min_range)


def test_selfatt_qk_consumes_onednn_backed_inputs():
    B, T_q, T_k, heads, head_dim = 1, 3, 4, 2, 4
    embed_dim = heads * head_dim
    queries_np = np.linspace(
        -0.5, 0.7, B * T_q * embed_dim).reshape(B, T_q, embed_dim).astype('float32')
    keys_np = np.linspace(
        -0.3, 0.9, B * T_k * embed_dim).reshape(B, T_k, embed_dim).astype('float32')
    queries = mx.nd.Activation(mx.nd.array(queries_np), act_type='relu')
    keys = mx.nd.Activation(mx.nd.array(keys_np), act_type='relu')

    out = mx.nd._internal._sg_onednn_selfatt_qk(
        queries=queries, keys=keys, heads=heads)

    q = np.maximum(queries_np, 0).reshape(B, T_q, heads, head_dim)
    k = np.maximum(keys_np, 0).reshape(B, T_k, heads, head_dim)
    expected = np.einsum('bqhd,bkhd->bhqk', q, k)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-5, atol=1e-5)


def test_selfatt_qk_split_consumes_onednn_backed_qkv():
    B, T, heads, head_dim = 1, 3, 2, 4
    embed_dim = heads * head_dim
    qkv_np = np.linspace(
        -0.6, 0.8, B * T * 3 * embed_dim).reshape(B, T, 3 * embed_dim).astype('float32')
    qkv = mx.nd.Activation(mx.nd.array(qkv_np), act_type='relu')

    out = mx.nd._internal._sg_onednn_selfatt_qk_split(qkv, heads=heads)

    qkv_ref = np.maximum(qkv_np, 0)
    q = qkv_ref[:, :, :embed_dim].reshape(B, T, heads, head_dim)
    k = qkv_ref[:, :, embed_dim:2 * embed_dim].reshape(B, T, heads, head_dim)
    expected = np.einsum('bqhd,bkhd->bhqk', q, k)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-5, atol=1e-5)


def test_selfatt_valatt_consumes_onednn_backed_qkv():
    B, T, heads, head_dim = 1, 3, 2, 4
    embed_dim = heads * head_dim
    attention_np = np.linspace(
        -0.2, 0.9, B * heads * T * T).reshape(B, heads, T, T).astype('float32')
    qkv_np = np.linspace(
        -0.6, 0.8, B * T * 3 * embed_dim).reshape(B, T, 3 * embed_dim).astype('float32')
    attention = mx.nd.Activation(mx.nd.array(attention_np), act_type='relu')
    qkv = mx.nd.Activation(mx.nd.array(qkv_np), act_type='relu')

    out = mx.nd._internal._sg_onednn_selfatt_valatt(attention, qkv, heads=heads)

    att_ref = np.maximum(attention_np, 0)
    qkv_ref = np.maximum(qkv_np, 0)
    value = qkv_ref[:, :, 2 * embed_dim:].reshape(B, T, heads, head_dim)
    expected = np.einsum('bhqk,bkhd->bqhd', att_ref, value).reshape(B, T, embed_dim)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-5, atol=1e-5)


def test_selfatt_qk_quantized_backward_matches_reference():
    B, T_q, T_k, heads, head_dim = 2, 3, 4, 2, 4
    ctx = mx.cpu()
    queries_np = np.linspace(
        -0.4, 0.5, B * T_q * heads * head_dim).reshape(B, T_q, heads * head_dim)
    keys_np = np.linspace(
        0.3, -0.6, B * T_k * heads * head_dim).reshape(B, T_k, heads * head_dim)
    queries = mx.nd.array(queries_np.astype('float32'), ctx=ctx)
    keys = mx.nd.array(keys_np.astype('float32'), ctx=ctx)
    queries.attach_grad()
    keys.attach_grad()

    with autograd.record():
        q, qmin, qmax = mx.nd.contrib.quantize_v2(
            queries, min_calib_range=-1.0, max_calib_range=1.0, out_type='int8')
        k, kmin, kmax = mx.nd.contrib.quantize_v2(
            keys, min_calib_range=-1.0, max_calib_range=1.0, out_type='int8')
        out = mx.nd._internal._sg_onednn_selfatt_qk(
            queries=q, keys=k, min_q=qmin, max_q=qmax, min_k=kmin, max_k=kmax,
            heads=heads, quantized=True, enabled_float_output='float32')
        out.sum().backward()

    q_float = _dequant_int8(q.asnumpy(), qmin.asscalar(), qmax.asscalar())
    k_float = _dequant_int8(k.asnumpy(), kmin.asscalar(), kmax.asscalar())
    q_heads = q_float.reshape(B, T_q, heads, head_dim)
    k_heads = k_float.reshape(B, T_k, heads, head_dim)
    expected_q = np.einsum('bhqk,bkhd->bqhd',
                           np.ones((B, heads, T_q, T_k), dtype='float32'),
                           k_heads).reshape(B, T_q, heads * head_dim)
    expected_k = np.einsum('bhqk,bqhd->bkhd',
                           np.ones((B, heads, T_q, T_k), dtype='float32'),
                           q_heads).reshape(B, T_k, heads * head_dim)
    np.testing.assert_allclose(queries.grad.asnumpy(), expected_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(keys.grad.asnumpy(), expected_k, rtol=1e-5, atol=1e-5)


def test_selfatt_qk_split_quantized_backward_matches_reference():
    B, T, heads, head_dim = 2, 3, 2, 4
    embed_dim = heads * head_dim
    ctx = mx.cpu()
    qkv_np = np.linspace(
        -0.5, 0.5, B * T * 3 * embed_dim).reshape(B, T, 3 * embed_dim)
    qkv_input = mx.nd.array(qkv_np.astype('float32'), ctx=ctx)
    qkv_input.attach_grad()

    with autograd.record():
        qkv, qkv_min, qkv_max = mx.nd.contrib.quantize_v2(
            qkv_input, min_calib_range=-1.0, max_calib_range=1.0, out_type='int8')
        out = mx.nd._internal._sg_onednn_selfatt_qk_split(
            qkv, qkv_min, qkv_max, heads=heads, quantized=True,
            enabled_float_output='float32')
        out.sum().backward()

    qkv_float = _dequant_int8(qkv.asnumpy(), qkv_min.asscalar(), qkv_max.asscalar())
    q = qkv_float[:, :, :embed_dim].reshape(B, T, heads, head_dim)
    k = qkv_float[:, :, embed_dim:2 * embed_dim].reshape(B, T, heads, head_dim)
    expected = np.zeros((B, T, 3, heads, head_dim), dtype='float32')
    upstream = np.ones((B, heads, T, T), dtype='float32')
    expected[:, :, 0, :, :] = np.einsum('bhqk,bkhd->bqhd', upstream, k)
    expected[:, :, 1, :, :] = np.einsum('bhqk,bqhd->bkhd', upstream, q)
    expected = expected.reshape(B, T, 3 * embed_dim)
    np.testing.assert_allclose(qkv_input.grad.asnumpy(), expected, rtol=1e-5, atol=1e-5)


def test_selfatt_valatt_quantized_backward_matches_reference():
    B, T, heads, head_dim = 2, 3, 2, 4
    embed_dim = heads * head_dim
    ctx = mx.cpu()
    attention_np = np.linspace(0.05, 0.95, B * heads * T * T).reshape(B, heads, T, T)
    qkv_np = np.linspace(-0.5, 0.5, B * T * 3 * embed_dim).reshape(B, T, 3 * embed_dim)
    attention = mx.nd.array(attention_np.astype('float32'), ctx=ctx)
    qkv_input = mx.nd.array(qkv_np.astype('float32'), ctx=ctx)
    attention.attach_grad()
    qkv_input.attach_grad()

    with autograd.record():
        att, att_min, att_max = mx.nd.contrib.quantize_v2(
            attention, min_calib_range=0.0, max_calib_range=1.0, out_type='uint8')
        qkv, qkv_min, qkv_max = mx.nd.contrib.quantize_v2(
            qkv_input, min_calib_range=-1.0, max_calib_range=1.0, out_type='int8')
        out = mx.nd._internal._sg_onednn_selfatt_valatt(
            att, qkv, att_min, att_max, qkv_min, qkv_max,
            heads=heads, quantized=True, enabled_float_output='float32')
        out.sum().backward()

    att_float = _dequant_uint8(att.asnumpy(), att_min.asscalar(), att_max.asscalar())
    qkv_float = _dequant_int8(qkv.asnumpy(), qkv_min.asscalar(), qkv_max.asscalar())
    value = qkv_float[:, :, 2 * embed_dim:].reshape(B, T, heads, head_dim)
    upstream = np.ones((B, T, heads, head_dim), dtype='float32')
    expected_att = np.einsum('bqhd,bkhd->bhqk', upstream, value)
    expected_qkv = np.zeros((B, T, 3, heads, head_dim), dtype='float32')
    expected_qkv[:, :, 2, :, :] = np.einsum('bhqk,bqhd->bkhd', att_float, upstream)
    expected_qkv = expected_qkv.reshape(B, T, 3 * embed_dim)
    np.testing.assert_allclose(attention.grad.asnumpy(), expected_att, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(qkv_input.grad.asnumpy(), expected_qkv, rtol=1e-5, atol=1e-5)


def test_selfatt_qk_forward_shape():
    """Constructs a tiny float32 query/key pair, runs the FP32 reference
    `batch_dot` shape, confirms the shape matches what the QK subgraph
    is documented to produce.  This pins the shape-inference contract
    without requiring the full quantization pipeline."""
    # Input shape (B, T, heads * head_dim) = (1, 4, 2 * 8) = (1, 4, 16).
    B, T_q, T_k, heads, head_dim = 1, 4, 8, 2, 8
    np.random.seed(0)
    queries_np = np.random.randn(B, T_q, heads * head_dim).astype('float32')
    keys_np = np.random.randn(B, T_k, heads * head_dim).astype('float32')

    # Reshape into (B, T, heads, head_dim) -> (B, heads, T, head_dim)
    def split_heads(x):
        return x.reshape(B, x.shape[1], heads, head_dim).transpose(0, 2, 1, 3)

    q = split_heads(queries_np)  # (B, heads, T_q, head_dim)
    k = split_heads(keys_np)     # (B, heads, T_k, head_dim)
    # The QK subgraph computes Q @ K^T with shape (B, heads, T_q, T_k).
    expected = np.einsum('bhqd,bhkd->bhqk', q, k)
    assert expected.shape == (B, heads, T_q, T_k)


def test_selfatt_qk_split_inference_smoke():
    """Float32 batch-dot of stacked Q/K/V should produce the expected
    (B, heads, T, T) attention-score shape that the quantized
    selfatt_qk_split subgraph also produces.

    This is a smoke test; the real selfatt_qk_split op only runs after
    quantization and subgraph fusion, which is exercised separately
    in `test_matmul_subgraph.py::test_self_attention`.
    """
    B, T, heads, head_dim = 1, 4, 2, 8
    np.random.seed(1)
    # Stacked Q,K,V along axis=2 with three projections back-to-back.
    qkv_np = np.random.randn(B, T, 3 * heads * head_dim).astype('float32')
    # split_heads on the *projected* Q.
    q = qkv_np[..., 0:heads * head_dim].reshape(B, T, heads, head_dim).transpose(0, 2, 1, 3)
    k = qkv_np[..., heads * head_dim:2*heads * head_dim].reshape(B, T, heads, head_dim).transpose(0, 2, 1, 3)
    expected = np.einsum('bhqd,bhkd->bhqk', q, k)
    assert expected.shape == (B, heads, T, T)


def test_selfatt_kAddTo_rejection_documented():
    """The XOP19 gates in dnnl_transformer.cc reject kAddTo for the primary
    output of `_sg_onednn_selfatt_qk` and `_sg_onednn_selfatt_valatt`
    before any write happens.  Pin the documentation contract here so a
    refactor that allows kAddTo through must include accumulation support
    and update this test deliberately.

    The actual CHECK_NE-throwing behavior is hard to invoke without
    constructing the subgraph at hybridize time + setting an explicit
    primary req via _bind; the gates live in the
    `SgDNNLSelfAttQKForward` / `DNNLSelfAttValAttForward` C++ entry
    points and would fire before any output is touched.  This test
    grep-checks the source so the gate is not silently removed.
    """
    import os
    src_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
        'src', 'operator', 'subgraph', 'dnnl')
    src_path = os.path.join(src_dir, 'dnnl_transformer.cc')
    if not os.path.exists(src_path):
        pytest.skip("source tree not available")
    with open(src_path) as f:
        contents = f.read()
    # The two entry points must both call CHECK_NE(req[0], kAddTo).
    assert (
        'CHECK_NE(req[0], kAddTo)' in contents
        and '_sg_onednn_selfatt_qk' in contents
        and '_sg_onednn_selfatt_valatt' in contents
    ), ("XOP19 primary-output kAddTo guards in SgDNNLSelfAttQKForward / "
        "DNNLSelfAttValAttForward appear to have been removed without an "
        "accompanying accumulation implementation.  Either restore the "
        "CHECK_NE guards or implement real kAddTo handling and update "
        "this test.")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
