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
Regression test for apache/mxnet#21199.

A 1x1 Conv2D whose output oneDNN v3 dispatches into a blocked layout
(e.g. brgconv_1x1:avx2 -> `acdb`) feeds a downstream Reshape via an
in-place op (reshape_like).  Before the fix in dnnl_reshape.cc, the
Reshape's input NDArray's metadata shape became 5-D while its DNNL
chunk was still 4-D in the conv-output blocked layout; oneDNN v3
rejected the resulting reorder with "inconsistent src and dst mds"
(reorder.cpp:90).

200, 208, and 256 channels all reproduced the failure on AVX2 hosts;
199 channels happened to dispatch to the gemm:jit conv impl which
already produces default `abcd`, so the path stayed safe.
"""

import json as _json
import os
import sys

import mxnet as mx
from mxnet import npx
import pytest

npx.set_np()


_SYMBOL_TEMPLATE = {
    "nodes": [
        {"op": "null", "name": ".Inputs.Input", "inputs": []},
        {
            "op": "Reshape",
            "name": ".Nodes.1$0",
            "attrs": {"shape": "(-3, -2)"},
            "inputs": [[0, 0, 0]],
        },
        {"op": "null", "name": ".Nodes.1.Parameters.Net.Arrays.Weights", "inputs": []},
        {"op": "null", "name": ".Nodes.1.Parameters.Net.Arrays.Biases", "inputs": []},
        {
            "op": "Convolution",
            "name": ".Nodes.1.Parameters.Net",
            "attrs": {
                "cudnn_off": "0",
                "dilate": "(1, 1)",
                "kernel": "(1, 1)",
                "layout": "None",
                "no_bias": "False",
                "num_filter": "__NF__",
                "num_group": "1",
                "pad": "(0, 0)",
                "stride": "(1, 1)",
            },
            "inputs": [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
        },
        {
            "op": "reshape_like",
            "name": ".Nodes.1$1",
            "attrs": {
                "lhs_begin": "0",
                "lhs_end": "1",
                "rhs_begin": "0",
                "rhs_end": "2",
            },
            "inputs": [[4, 0, 0], [0, 0, 0]],
        },
        {
            "op": "Reshape",
            "name": ".Nodes.2$0",
            "attrs": {"shape": "(0, 2, __NF__)"},
            "inputs": [[5, 0, 0]],
        },
        {"op": "_copy", "name": ".Outputs.Output", "inputs": [[6, 0, 0]]},
    ],
    "arg_nodes": [0, 2, 3],
    "heads": [[7, 0, 0]],
}


def _build_cached_op(num_filter):
    sym_str = _json.dumps(_SYMBOL_TEMPLATE).replace("__NF__", str(num_filter))
    sym = mx.symbol.fromjson(sym_str)
    return mx.ndarray.CachedOp(sym)


@pytest.mark.parametrize("num_filter", [199, 200, 208, 256, 512])
def test_5d_input_1x1_conv_cached_op(num_filter):
    """Apache#21199: 1x1 conv on a 5-D input via CachedOp must not crash."""
    op = _build_cached_op(num_filter)
    args = [
        mx.np.random.uniform(size=(1, 2, num_filter, 1, 1), ctx=mx.cpu()),
        mx.np.random.uniform(size=(num_filter, num_filter, 1, 1), ctx=mx.cpu()),
        mx.np.random.uniform(size=(num_filter,), ctx=mx.cpu()),
    ]
    out = op(*args)
    if isinstance(out, (list, tuple)):
        out = out[0]
    out.wait_to_read()
    assert out.shape == (1, 2, num_filter), (
        f"unexpected output shape {out.shape} for num_filter={num_filter}"
    )


def test_5d_input_1x1_conv_correctness():
    """Sanity check that the 2-step reorder produces the right numbers.

    Compare the CachedOp output against a numpy-side recomputation of the
    same 1x1 conv at num_filter=200 (the original failing config).
    """
    import numpy as np

    num_filter = 200
    op = _build_cached_op(num_filter)
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((1, 2, num_filter, 1, 1)).astype(np.float32)
    w_np = rng.standard_normal((num_filter, num_filter, 1, 1)).astype(np.float32)
    b_np = rng.standard_normal((num_filter,)).astype(np.float32)

    x = mx.np.array(x_np, ctx=mx.cpu())
    w = mx.np.array(w_np, ctx=mx.cpu())
    b = mx.np.array(b_np, ctx=mx.cpu())
    out = op(x, w, b)
    if isinstance(out, (list, tuple)):
        out = out[0]
    out_np = out.asnumpy()

    # Reference: collapse N,D -> 4D, run 1x1 conv, then reshape back.
    flat = x_np.reshape(2, num_filter, 1, 1)
    # 1x1 conv = matmul over the channel axis.
    ref = np.einsum("ncij,kcij->nkij", flat, w_np) + b_np.reshape(1, -1, 1, 1)
    ref = ref.reshape(1, 2, num_filter)

    np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
