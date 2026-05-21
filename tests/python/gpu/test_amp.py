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

import json
import sys
from pathlib import Path
curr_path = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_path.parent))
sys.path.insert(0, str(curr_path.parent/'unittest'))

import mxnet as mx
import pytest
from mxnet import amp
from mxnet.base import MXNetError
from mxnet.test_utils import set_default_device
from mxnet.gluon import nn, rnn

import amp.common as amp_common_tests
from common import assert_raises_cudnn_not_satisfied

AMP_DTYPE = 'float16'

set_default_device(mx.gpu(0))


def test_fp16_coverage():
    amp_common_tests.test_amp_coverage(AMP_DTYPE, 'FP16')


@mx.util.use_np
def test_fp16_basic_use():
    amp_common_tests.test_amp_basic_use(AMP_DTYPE)


@mx.util.use_np
def test_fp16_offline_casting():
    amp_common_tests.test_amp_offline_casting(AMP_DTYPE)


@mx.util.use_np
def test_fp16_offline_casting_shared_params():
    amp_common_tests.test_amp_offline_casting_shared_params(AMP_DTYPE)


@mx.util.use_np
def test_fp16_fp32_ops_order_independence():
    amp_common_tests.test_lp16_fp32_ops_order_independence(AMP_DTYPE)


def _amp_convert_symbol(sym, input_dtypes, param_dtypes):
    return amp.convert_symbol(sym, input_dtypes, param_dtypes, AMP_DTYPE)


def _node_by_name(nodes, name):
    return next(node for node in nodes if node['name'] == name)


def _input_node(nodes, node, input_idx=0):
    return nodes[node['inputs'][input_idx][0]]


def _internal_output_types(sym, **input_types):
    internals = sym.get_internals()
    _, output_types, _ = internals.infer_type(**input_types)
    return {
        name: mx.nd.get_dtype_name(dtype)
        for name, dtype in zip(internals.list_outputs(), output_types)
    }


def test_fp16_quantize_v2_is_fp32_boundary():
    data = mx.sym.var('data')
    weight = mx.sym.var('weight')
    bias = mx.sym.var('bias')
    fc = mx.sym.FullyConnected(data, weight, bias, num_hidden=4, name='fc')
    quantized = mx.sym.contrib.quantize_v2(
        fc, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0, name='quantize')

    converted = _amp_convert_symbol(mx.sym.Group(list(quantized)),
                                    {'data': 'float32'},
                                    {'weight': 'float32', 'bias': 'float32'})
    output_types = _internal_output_types(
        converted, data='float32', weight='float32', bias='float32')

    assert output_types['fc_output'] == AMP_DTYPE
    assert output_types['fc_0_amp_cast_float32_output'] == 'float32'
    assert output_types['quantize_output0'] == 'int8'
    assert output_types['quantize_output1'] == 'float32'
    assert output_types['quantize_output2'] == 'float32'


def test_fp16_quantize_v2_direct_input_is_unsupported():
    data = mx.sym.var('data')
    quantized = mx.sym.contrib.quantize_v2(
        data, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0)

    with pytest.raises(MXNetError, match='quantize_v2'):
        quantized[0].infer_type(data='float16')


def test_fp16_dequantize_is_fp32_boundary():
    data = mx.sym.var('data')
    quantized = mx.sym.contrib.quantize_v2(
        data, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0, name='quantize')
    dequantized = mx.sym.contrib.dequantize(*quantized, out_type='float32', name='dequantize')
    weight = mx.sym.var('weight')
    bias = mx.sym.var('bias')
    fc = mx.sym.FullyConnected(dequantized, weight, bias, num_hidden=4, name='fc')

    converted = _amp_convert_symbol(fc,
                                    {'data': 'float32'},
                                    {'weight': 'float32', 'bias': 'float32'})
    nodes = json.loads(converted.get_internals().tojson())['nodes']
    fc_node = _node_by_name(nodes, 'fc')
    output_types = _internal_output_types(
        converted, data='float32', weight='float32', bias='float32')

    assert _input_node(nodes, fc_node)['name'] == 'dequantize'
    assert output_types['dequantize_output'] == 'float32'
    assert output_types['dequantize_0_amp_cast_float16_output'] == AMP_DTYPE
    assert output_types['fc_output'] == AMP_DTYPE


@mx.util.use_np
def test_fp16_test_node_excluding():
    amp_common_tests.test_amp_node_excluding(AMP_DTYPE)


@mx.util.use_np
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_amp_conversion_rnn():
    # Upstream #18099 reported a waitall() failure here under MXNet 1.6.
    # The cuDNN-9 v8 RNN port on this fork does not exhibit that failure.
    with mx.Device(mx.gpu(0)):
        model = nn.HybridSequential()
        model.add(rnn.LSTM(hidden_size=10, num_layers=2, bidirectional=True))
        model.add(nn.Dense(2))
        model.initialize(device=mx.gpu(0))
        model.hybridize()
        data = mx.np.ones((2, 3, 4), ctx=mx.gpu(0))
        out = model(data)
        new_model = amp.convert_hybrid_block(model, data, target_dtype=AMP_DTYPE)
        out2 = new_model(data)
        mx.test_utils.assert_almost_equal(out.asnumpy(), out2.asnumpy(), atol=1e-2, rtol=1e-2)
