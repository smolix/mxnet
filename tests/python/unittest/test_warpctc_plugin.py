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

import os

import numpy as np
import pytest

import mxnet as mx


def _warpctc_symbol():
    plugin_path = os.environ.get('MXNET_WARPCTC_PLUGIN_PATH')
    if plugin_path and not hasattr(mx.sym, 'WarpCTC'):
        mx.library.load(plugin_path, verbose=False)
    if not hasattr(mx.sym, 'WarpCTC'):
        pytest.skip(
            'WarpCTC plugin is not registered.  Build the plugin and set '
            'MXNET_WARPCTC_PLUGIN_PATH=/path/to/libwarpctc_plugin.so to '
            'exercise the sentinel-rejection contract (XOP26).')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    return mx.sym.WarpCTC(data=data, label=label, input_length=3, label_length=3)


def test_warpctc_rejects_add_grad_req_before_overwriting_data_grad():
    sym = _warpctc_symbol()
    data = mx.nd.array(np.asarray([
        [1.2, 3.4, 1.2, -0.1, -2.34],
        [0.1, 0.2, 0.3, 0.22, 0.123],
        [-15, -14, -13, -12, -11],
    ], dtype=np.float32))
    label = mx.nd.array(np.asarray([2, 3, 0], dtype=np.int32))
    data_grad = mx.nd.full(data.shape, 17)

    exe = sym._bind(
        ctx=mx.cpu(),
        args={'data': data, 'label': label},
        args_grad={'data': data_grad},
        grad_req={'data': 'add', 'label': 'null'})
    exe.forward(is_train=True)

    with pytest.raises(mx.MXNetError, match='WarpCTC only supports write requests'):
        exe.backward()
        mx.nd.waitall()

    np.testing.assert_equal(data_grad.asnumpy(), np.full(data.shape, 17))
