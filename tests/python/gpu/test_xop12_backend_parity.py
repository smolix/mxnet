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

"""XOP12 backend-parity layer.

For a representative set of operators (and the contract grad_req values)
that the CPU harness in `test_xop12_operator_req_contract.py` covers,
verify that running the same forward+backward on GPU produces gradients
that match the CPU result within a numeric tolerance.

This catches three classes of regression at once:
- cuDNN/cuBLAS req-mapping bug that diverges from CPU semantics
- backend-specific scratch reuse that contaminates the gradient
- silent dtype demotion on the GPU path

Together with the CPU contract harness this gives XOP12 a backend-parity
dimension without forking the OPS_UNDER_CONTRACT table.

Add a new operator: append a `pytest.param` row with the symbol builder,
input shape, and filler; the harness handles the rest.
"""

import numpy as np
import pytest

import mxnet as mx


def _gpu_available():
    return mx.runtime.Features().is_enabled("CUDA") and mx.context.num_gpus() > 0


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="GPU required for backend-parity")


def _rand_default_input(shape):
    def filler(rng):
        return rng.randn(*shape).astype('float32')
    return filler


PARITY_OPS = [
    pytest.param(
        lambda x: mx.sym.softmax(data=x, axis=-1),
        (4, 8), _rand_default_input((4, 8)),
        id='softmax_axis_last',
    ),
    pytest.param(
        lambda x: mx.sym.log_softmax(data=x, axis=-1),
        (4, 8), _rand_default_input((4, 8)),
        id='log_softmax_axis_last',
    ),
    pytest.param(
        lambda x: mx.sym.sum(data=x, axis=1),
        (3, 5), _rand_default_input((3, 5)),
        id='sum_axis_1',
    ),
    pytest.param(
        lambda x: mx.sym.Activation(data=x, act_type='relu'),
        (4, 5), _rand_default_input((4, 5)),
        id='activation_relu',
    ),
    pytest.param(
        lambda x: mx.sym.Activation(data=x, act_type='sigmoid'),
        (4, 5), _rand_default_input((4, 5)),
        id='activation_sigmoid',
    ),
    pytest.param(
        lambda x: mx.sym.Activation(data=x, act_type='tanh'),
        (4, 5), _rand_default_input((4, 5)),
        id='activation_tanh',
    ),
    pytest.param(
        lambda x: mx.sym.LeakyReLU(data=x, act_type='leaky', slope=0.01),
        (4, 5), _rand_default_input((4, 5)),
        id='leaky_relu',
    ),
]


def _run_backward(build_symbol, input_shape, input_filler,
                  grad_req, grad_init_value, ctx):
    rng = np.random.RandomState(0)
    input_np = input_filler(rng)
    grad_init = np.full(input_shape, grad_init_value, dtype='float32')
    grad_buf = mx.nd.array(grad_init, ctx=ctx)
    x = mx.sym.Variable('x')
    out_sym = build_symbol(x)
    args = {'x': mx.nd.array(input_np, ctx=ctx)}
    args_grad = {'x': grad_buf}
    req_map = {'x': grad_req}
    exe = out_sym._bind(ctx=ctx, args=args, args_grad=args_grad,
                        grad_req=req_map)
    exe.forward(is_train=True)
    head_grad = mx.nd.ones_like(exe.outputs[0])
    exe.backward(head_grad)
    mx.nd.waitall()
    return grad_buf.asnumpy(), exe.outputs[0].asnumpy()


@pytest.mark.parametrize('build_symbol, input_shape, input_filler', PARITY_OPS)
@pytest.mark.parametrize('grad_req', ['write', 'add'])
def test_cpu_gpu_parity_for_grad_req(build_symbol, input_shape, input_filler,
                                      grad_req):
    """Run forward+backward on CPU and GPU with the same input and grad_req.
    Output and data-gradient must match within fp32 tolerance."""
    cpu_grad, cpu_out = _run_backward(build_symbol, input_shape, input_filler,
                                       grad_req=grad_req,
                                       grad_init_value=3.0, ctx=mx.cpu())
    gpu_grad, gpu_out = _run_backward(build_symbol, input_shape, input_filler,
                                       grad_req=grad_req,
                                       grad_init_value=3.0, ctx=mx.gpu(0))
    np.testing.assert_allclose(gpu_out, cpu_out, rtol=1e-4, atol=1e-5,
        err_msg=f"Forward diverges between CPU and GPU (grad_req={grad_req})")
    np.testing.assert_allclose(gpu_grad, cpu_grad, rtol=1e-4, atol=1e-5,
        err_msg=f"Gradient diverges between CPU and GPU (grad_req={grad_req})")


@pytest.mark.parametrize('build_symbol, input_shape, input_filler', PARITY_OPS)
def test_cpu_gpu_parity_null_preserves_sentinel(build_symbol, input_shape,
                                                 input_filler):
    """grad_req='null' must leave the gradient buffer untouched on both CPU and GPU
    (and the buffers must agree — same sentinel)."""
    cpu_grad, _ = _run_backward(build_symbol, input_shape, input_filler,
                                 grad_req='null', grad_init_value=17.0,
                                 ctx=mx.cpu())
    gpu_grad, _ = _run_backward(build_symbol, input_shape, input_filler,
                                 grad_req='null', grad_init_value=17.0,
                                 ctx=mx.gpu(0))
    np.testing.assert_array_equal(cpu_grad, np.full_like(cpu_grad, 17.0))
    np.testing.assert_array_equal(gpu_grad, np.full_like(gpu_grad, 17.0))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
