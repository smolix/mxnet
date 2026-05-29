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

import numpy as np
import pytest

import mxnet as mx
from mxnet.test_utils import assert_almost_equal


pytestmark = pytest.mark.skipif(
    not mx.runtime.Features().is_enabled("ONEDNN"),
    reason="oneDNN backend not enabled",
)


def test_amp_cast_cpu_fp16_falls_back_from_onednn():
    data_np = np.array([-2.0, -0.5, 0.25, 8.0], dtype=np.float16)
    x = mx.sym.Variable("x", dtype=np.float16)
    sym = mx.sym.amp_cast(x, dtype=np.float32)
    exe = sym._bind(mx.cpu(), {"x": mx.nd.array(data_np, dtype=np.float16, ctx=mx.cpu())})

    exe.forward(is_train=True)
    out = exe.outputs[0]
    assert out.dtype == np.float32
    assert_almost_equal(out.asnumpy(), data_np.astype(np.float32))


def test_amp_multicast_cpu_fp16_falls_back_from_onednn():
    data_np = np.array([[-3.0, 0.5], [2.0, 4.0]], dtype=np.float16)
    x = mx.sym.Variable("x", dtype=np.float16)
    y = mx.sym.Variable("y", dtype=np.float32)
    sym = mx.sym.amp_multicast(x, y, num_outputs=2)
    exe = sym._bind(
        mx.cpu(),
        {
            "x": mx.nd.array(data_np, dtype=np.float16, ctx=mx.cpu()),
            "y": mx.nd.array(data_np.astype(np.float32), dtype=np.float32, ctx=mx.cpu()),
        },
    )

    exe.forward(is_train=True)
    out_x, out_y = exe.outputs
    assert out_x.dtype == np.float32
    assert out_y.dtype == np.float32
    assert_almost_equal(out_x.asnumpy(), data_np.astype(np.float32))
    assert_almost_equal(out_y.asnumpy(), data_np.astype(np.float32))
