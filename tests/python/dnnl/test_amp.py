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

import sys
from pathlib import Path
curr_path = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_path.parent))

import pytest
import mxnet as mx
import amp.common as amp_common_tests
from mxnet.test_utils import assert_almost_equal
from mxnet.amp.lists.symbol_bf16 import (BF16_FUNCS, BF16_FP32_FUNCS, WIDEST_TYPE_CASTS,
                                         CONDITIONAL_FP32_FUNCS)

from op_cfg import get_op_cfg_generator, get_symblock_from_args_scenario, CFG_RTOL_ATOL
from dnnl_test_utils import has_native_onednn_bf16, require_native_onednn_bf16


ALL_BF16_OPS = BF16_FUNCS + BF16_FP32_FUNCS + WIDEST_TYPE_CASTS
ALL_BF16_OPS += [op_name for op_name, attr_name, attr_vals in CONDITIONAL_FP32_FUNCS]

AMP_DTYPE = 'bfloat16'

NATIVE_ONEDNN_BF16_OPS = {
    'Convolution',
    'Deconvolution',
    'Pooling',
    'Activation',
    'LeakyReLU',
    'BatchNorm',
    'LRN',
    'softmax',
    'log_softmax',
    '_npi_exp',
    '_npi_tanh',
    '_npi_sqrt',
    '_npi_square',
    'dot',
    'batch_dot',
    '_npi_dot',
    '_sg_onednn_batch_dot',
    '_sg_onednn_batch_norm',
}

NATIVE_ONEDNN_BF16_SKIP_REASON = \
    'oneDNN native BF16 primitives are unavailable on this CPU'


def test_bf16_coverage():
    amp_common_tests.test_amp_coverage(AMP_DTYPE, 'BF16')


@mx.util.use_np
def test_bf16_basic_use():
    amp_common_tests.test_amp_basic_use(AMP_DTYPE)


@mx.util.use_np
def test_bf16_offline_casting():
    require_native_onednn_bf16()
    amp_common_tests.test_amp_offline_casting(AMP_DTYPE)


@mx.util.use_np
def test_bf16_offline_casting_shared_params():
    amp_common_tests.test_amp_offline_casting_shared_params(AMP_DTYPE)


@mx.util.use_np
def test_bf16_fp32_ops_order_independence():
    amp_common_tests.test_lp16_fp32_ops_order_independence(AMP_DTYPE)


@mx.util.use_np
def test_bf16_test_node_excluding():
    amp_common_tests.test_amp_node_excluding(AMP_DTYPE)


def get_param_name(param):
    if isinstance(param, (mx.nd.NDArray, mx.np.ndarray)):
        return 'Tensor' + str(param.shape)
    if isinstance(param, (tuple, list)):
        return str(type(param)(get_param_name(elem) for elem in param))
    # Render callables (weight-tensor factories, lambdas) by name, NOT repr:
    # repr embeds the object's memory address (e.g. "<function f at 0x7f..>"),
    # which differs per process and makes pytest-xdist abort collection with
    # "Different tests were collected between gw0 and gw1" under `-n`.
    if callable(param):
        return getattr(param, '__name__', None) or type(param).__name__
    return str(param)


def get_test_name(param):
    if isinstance(param, str):
        return f'"{param}" '  # op_name
    if isinstance(param, dict):
        elements = []
        for args_names, args_cfgs in param.items():
            if isinstance(args_cfgs, tuple):
                binded_args = args_names.split(',')
                for arg_name, arg_val in zip(binded_args, args_cfgs):
                    elements.append(f'"{arg_name}": {get_param_name(arg_val)}')
            else:
                arg_name, arg_val = args_names, args_cfgs
                elements.append(f'"{arg_name}": {get_param_name(arg_val)}')
        return ' ' + ', '.join(elements)
    raise TypeError('Op configuration should only consist of its name (str) and arg config (dict)')


def get_bf16_op_cfg_generator():
    native_bf16_available = has_native_onednn_bf16()
    native_bf16_skip = pytest.mark.skip(reason=NATIVE_ONEDNN_BF16_SKIP_REASON)
    for op_name, args_scenario in get_op_cfg_generator(ALL_BF16_OPS, AMP_DTYPE):
        marks = []
        if op_name in NATIVE_ONEDNN_BF16_OPS and not native_bf16_available:
            marks.append(native_bf16_skip)
        yield pytest.param(op_name, args_scenario, marks=marks)


@pytest.mark.parametrize(argnames=('op_name', 'args_scenario'),
                         argvalues=get_bf16_op_cfg_generator(),
                         ids=get_test_name)
def test_bf16_op(op_name, args_scenario):
    symblock, bf16_symblock_input_data = get_symblock_from_args_scenario(op_name, args_scenario)
    rtol, atol = args_scenario.get(CFG_RTOL_ATOL, (0.01, None))
    
    fp32_symblock_input_data = []
    for tensor in bf16_symblock_input_data:
        if mx.nd.get_dtype_name(tensor.dtype) == 'bfloat16':
            tensor = tensor.astype('float32')
        fp32_symblock_input_data.append(tensor)

    try:
        bf16_outs = symblock(*bf16_symblock_input_data)
        fp32_outs = symblock(*fp32_symblock_input_data)
        mx.nd.waitall()
    except mx.MXNetError as e:
        pytest.fail(str(e))

    if not isinstance(bf16_outs, (list, tuple)):
        bf16_outs = [bf16_outs]
    if not isinstance(fp32_outs, (list, tuple)):
        fp32_outs = [fp32_outs]

    assert any(mx.nd.get_dtype_name(tensor.dtype) == 'bfloat16'
               for tensor in bf16_symblock_input_data + bf16_outs)
    assert len(bf16_outs) == len(fp32_outs)
    for bf16_out, fp32_out in zip(bf16_outs, fp32_outs):
        assert_almost_equal(bf16_out.astype('float32'), fp32_out.astype('float32'), rtol, atol)
