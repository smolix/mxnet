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

import mxnet as mx
import numpy as onp
import pytest

from mxnet import amp
from mxnet.base import MXNetError
from mxnet.test_utils import assert_almost_equal


AMP_DTYPE = 'float16'


def _internal_output_types(sym, **input_types):
    internals = sym.get_internals()
    _, output_types, _ = internals.infer_type(**input_types)
    return {
        name: mx.nd.get_dtype_name(dtype)
        for name, dtype in zip(internals.list_outputs(), output_types)
    }


def _convert_to_fp16_amp(sym, input_dtypes, param_dtypes=None):
    return amp.convert_symbol(sym, input_dtypes, param_dtypes or {}, AMP_DTYPE)


def test_fp16_fp32_amp_boundary_to_int8_quantize_v2():
    data = mx.sym.var('data')
    weight = mx.sym.var('weight')
    bias = mx.sym.var('bias')
    fc = mx.sym.FullyConnected(data, weight, bias, num_hidden=4, name='fc')
    quantized = mx.sym.contrib.quantize_v2(
        fc, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0, name='quantize')

    converted = _convert_to_fp16_amp(mx.sym.Group(list(quantized)),
                                     {'data': 'float32'},
                                     {'weight': 'float32', 'bias': 'float32'})
    output_types = _internal_output_types(
        converted, data='float32', weight='float32', bias='float32')

    assert output_types['fc_output'] == AMP_DTYPE
    assert output_types['fc_0_amp_cast_float32_output'] == 'float32'
    assert output_types['quantize_output0'] == 'int8'
    assert output_types['quantize_output1'] == 'float32'
    assert output_types['quantize_output2'] == 'float32'


def test_int8_fp32_dequantize_amp_boundary_back_to_fp16():
    data = mx.sym.var('data')
    quantized = mx.sym.contrib.quantize_v2(
        data, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0, name='quantize')
    dequantized = mx.sym.contrib.dequantize(*quantized, out_type='float32', name='dequantize')
    weight = mx.sym.var('weight')
    bias = mx.sym.var('bias')
    fc = mx.sym.FullyConnected(dequantized, weight, bias, num_hidden=4, name='fc')

    converted = _convert_to_fp16_amp(fc,
                                     {'data': 'float32'},
                                     {'weight': 'float32', 'bias': 'float32'})
    output_types = _internal_output_types(
        converted, data='float32', weight='float32', bias='float32')

    assert output_types['quantize_output0'] == 'int8'
    assert output_types['quantize_output1'] == 'float32'
    assert output_types['quantize_output2'] == 'float32'
    assert output_types['dequantize_output'] == 'float32'
    assert output_types['dequantize_0_amp_cast_float16_output'] == AMP_DTYPE
    assert output_types['fc_output'] == AMP_DTYPE
    assert output_types['fc_0_amp_cast_float32_output'] == 'float32'


def test_int8_fp32_quantize_dequantize_runtime_matrix():
    data_np = onp.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype='float32')
    data = mx.nd.array(data_np, dtype='float32')

    qdata, min_val, max_val = mx.nd.contrib.quantize_v2(
        data, 'int8', min_calib_range=-1.0, max_calib_range=1.0)
    dequantized = mx.nd.contrib.dequantize(qdata, min_val, max_val, out_type='float32')

    assert qdata.dtype == onp.int8
    assert min_val.dtype == onp.float32
    assert max_val.dtype == onp.float32
    assert dequantized.dtype == onp.float32
    assert_almost_equal(min_val.asnumpy(), onp.array([-1.0], dtype='float32'))
    assert_almost_equal(max_val.asnumpy(), onp.array([1.0], dtype='float32'))
    assert_almost_equal(dequantized.asnumpy(), data_np, atol=1.0 / 127.0, rtol=1.0 / 127.0)


@pytest.mark.parametrize('case', ['quantize_v2_fp16_input', 'dequantize_fp16_output'])
def test_int8_fp16_rejection_matrix(case):
    if case == 'quantize_v2_fp16_input':
        data = mx.sym.var('data')
        quantized = mx.sym.contrib.quantize_v2(
            data, out_type='int8', min_calib_range=-1.0, max_calib_range=1.0)
        with pytest.raises(MXNetError, match='quantize_v2'):
            quantized[0].infer_type(data='float16')
    else:
        qdata = mx.sym.var('qdata')
        min_val = mx.sym.var('min_val')
        max_val = mx.sym.var('max_val')
        with pytest.raises(MXNetError, match='float16'):
            mx.sym.contrib.dequantize(qdata, min_val, max_val, out_type='float16')
