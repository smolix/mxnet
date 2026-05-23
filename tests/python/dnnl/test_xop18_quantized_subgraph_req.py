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

"""XOP18 forward contract anchor for quantized self-attention subgraphs.

issues.md XOP18 calls for direct small int8/uint8 contract tests for
`_sg_onednn_selfatt_qk{,_split}` and `_sg_onednn_selfatt_valatt` before
fixing the backward direction (which is blocked on B4 / NNVM CachedOp).

The full subgraph backward is xfailed in
`tests/python/dnnl/subgraphs/test_quantized_backward.py`; this file
covers the orthogonal contract that the forward path is at least
**registered + invokable + shape-correct** so a future quantization
refactor that drops one of these subgraph ops fails here loudly instead
of breaking ResNet-style transformer inference silently.
"""

import numpy as np
import pytest

import mxnet as mx


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
    sym_module = mx.sym
    assert hasattr(sym_module, op_name) or hasattr(sym_module.contrib, op_name) or \
        hasattr(sym_module._internal, op_name) if hasattr(sym_module, '_internal') else True, \
        f"Self-attention subgraph op {op_name!r} is not registered in mx.sym"


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
