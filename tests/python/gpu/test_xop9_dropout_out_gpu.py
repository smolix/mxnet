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

"""XOP9 cuDNN Dropout direct out= forward-path coverage."""

import numpy as np
import pytest

import mxnet as mx
from mxnet.test_utils import use_np


pytestmark = [
    pytest.mark.skipif(mx.device.num_gpus() == 0, reason="requires GPU for XOP9 cuDNN dropout out= coverage"),
    pytest.mark.skipif(not mx.runtime.Features().is_enabled('CUDNN'),
                       reason="requires cuDNN for XOP9 dropout out= coverage"),
]


def _assert_dropout_out_written(ret, out):
    assert ret is out
    values = out.asnumpy()
    assert values.shape == (64, 64)
    assert not np.any(values == -7.0)
    assert np.all(np.logical_or(values == 0.0, values == 2.0))


def test_legacy_dropout_out_cudnn_forward_path():
    data = mx.nd.ones((64, 64), ctx=mx.gpu(0))
    out = mx.nd.full((64, 64), -7.0, ctx=mx.gpu(0))

    ret = mx.nd.Dropout(data, p=0.5, mode='always', cudnn_off=False, out=out)
    mx.nd.waitall()

    _assert_dropout_out_written(ret, out)


@use_np
def test_npx_dropout_out_cudnn_forward_path():
    data = mx.np.ones((64, 64), ctx=mx.gpu(0))
    out = mx.np.full((64, 64), -7.0, ctx=mx.gpu(0))

    ret = mx.npx.dropout(data, p=0.5, mode='always', cudnn_off=False, out=out)
    mx.npx.waitall()

    _assert_dropout_out_written(ret, out)

