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

"""FU-1 regression test: oneDNN v3 jit_uni_int8_1x1:avx2 quantized 1x1
conv kernel produces channel-zero output when fused with eltwise_relu
post-op AND input channels ic < simd_w (AVX2 simd_w=8).

Reproduces the original ``test_pos_single_concat_pos_neg`` failure with
``data_shape=(4, 3, 24, 24)`` (IC=3 < 8). Without the in-tree gate at
``src/operator/subgraph/dnnl/dnnl_conv_property.h``, the int8 conv +
relu fusion path silently zeroes the output channel on AVX2 hosts and
``assert_almost_equal_with_err`` fails.

This test must pass WITHOUT setting ``MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU=1``
manually -- the gate auto-detects the FU-1 trigger shape.

On AVX-512 hosts the gate does NOT fire (the buggy avx2 kernel is not
dispatched) and the test should still pass.
"""

import os
import sys

import pytest

# Ensure the env var workaround is NOT set: we are validating the
# in-tree gate, not the manual env var escape hatch.
if os.environ.get("MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU", "0") not in ("0", ""):
    pytest.skip(
        "MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU is set externally; this test "
        "needs the env var unset so it actually exercises the in-tree gate.",
        allow_module_level=True,
    )

import mxnet as mx
from mxnet.gluon import nn

# Reuse the existing subgraph helper so the comparison logic stays in lockstep
# with the upstream conv-subgraph tests we are protecting.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "subgraphs"))
from subgraph_common import check_quantize  # noqa: E402


# FU-1 trigger shapes: IC < 8 with a 1x1 quantized conv + relu fusion.
# (4, 3, 24, 24) is the canonical failing shape from
# test_pos_single_concat_pos_neg in tests/python/dnnl/subgraphs/test_conv_subgraph.py.
# (1, 1, 8, 8) and (2, 7, 4, 4) cover IC=1 and IC=7 (just below simd_w).
_FU1_SHAPES = [
    (4, 3, 24, 24),
    (1, 1, 8, 8),
    (2, 7, 4, 4),
]


class _ConvDataConcat(nn.HybridBlock):
    """Exact structure of test_pos_single_concat_pos_neg: 1x1 conv -> relu -> concat."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=1, use_bias=False)
        self.act = nn.Activation(activation="relu")
        self.concat_dim = dim

    def forward(self, x):
        relu_out = self.act(self.conv0(x))
        out = mx.np.concatenate([x, relu_out], axis=self.concat_dim)
        return out


@pytest.fixture(autouse=True)
def _legacy_nd_semantics():
    prev_arr = mx.util.is_np_array()
    prev_shp = mx.util.is_np_shape()
    mx.npx.reset_np()
    yield
    mx.npx.set_np(shape=prev_shp, array=prev_arr)


@mx.util.use_np
@pytest.mark.parametrize("data_shape", _FU1_SHAPES)
@pytest.mark.parametrize("out_type", ["int8", "auto"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_fu1_int8_conv_relu_ic_lt8(data_shape, out_type, seed):
    """Verify FU-1 gate keeps int8 conv+relu output correct on AVX2 hosts.

    Without the gate this fails channel-zeroing assertions on AVX2-only
    boxes (e.g. AMD EPYC Zen 2/3/4); with the gate the conv stays
    unfused (plain int8 conv -> separate relu) and the numerical check
    passes. On AVX-512 hosts the gate is a no-op.
    """
    mx.np.random.seed(seed)

    net = _ConvDataConcat(dim=1)
    check_quantize(net, data_shape, out_type, name="", check_calibration=False)
