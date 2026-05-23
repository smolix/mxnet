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

"""XOP12 reusable operator output-request contract harness.

For every operator wired into this harness we exercise four request
behaviors via Symbol._bind with an explicit `args_grad` buffer:

- ``null``    — the gradient buffer must be unchanged after backward.
- ``write``   — the gradient buffer must hold the new gradient
                (sentinel must be overwritten).
- ``add``     — the gradient buffer must equal `init + new_gradient`.
- (no req=inplace test here because the inplace check requires aliasing
   input/output storage and most operators reject that contract at bind
   time; the existing per-operator tests cover inplace where it's
   meaningful.)

Adding a new operator means appending a row to `OPS_UNDER_CONTRACT` with
its symbolic constructor + an input shape.  The harness handles the
sentinel/grad-buffer plumbing so all operators are checked identically.

This file is the **anchor** the issues.md XOP12 row tracks: it is small
(3 operators today: softmax, sum, layer_norm) and intentional so that
new XOP fixes can be required to plug in.  Expand by adding rows, not
by forking the harness.

When a backend-specific contract (e.g. cuDNN beta=1 for kAddTo) is
violated by the kernel, the parameterized failure tells you which
operator + which request broke.  That makes regressions actionable in
one bisect step instead of guessing across operator/backend/req
combinations.
"""

import numpy as np
import pytest

import mxnet as mx


# ----------------------------------------------------------------------
# Operator wiring.  Each row registers:
#   - id: a stable test-id slug
#   - build_symbol(input_var) → Symbol returning a single output
#   - input_shape: tuple
#   - input_filler: callable rng → np.ndarray of input_shape
# Backward goes through the symbolic head_grad of ones.
# ----------------------------------------------------------------------

def _rand_softmax_input(rng):
    return rng.randn(4, 8).astype('float32')


def _rand_sum_input(rng):
    return rng.randn(3, 5).astype('float32')


def _rand_layernorm_input(rng):
    return rng.randn(2, 6).astype('float32')


OPS_UNDER_CONTRACT = [
    pytest.param(
        lambda x: mx.sym.softmax(data=x, axis=-1),
        (4, 8),
        _rand_softmax_input,
        id='softmax_axis_last',
    ),
    pytest.param(
        lambda x: mx.sym.sum(data=x, axis=1),
        (3, 5),
        _rand_sum_input,
        id='sum_axis_1',
    ),
    pytest.param(
        # LayerNorm needs gamma/beta; freeze them inside the closure.
        lambda x: mx.sym.LayerNorm(
            data=x,
            gamma=mx.sym.Variable('_lngamma'),
            beta=mx.sym.Variable('_lnbeta'),
            axis=-1, eps=1e-5, output_mean_var=False),
        (2, 6),
        _rand_layernorm_input,
        id='layernorm_axis_last',
    ),
]


def _layernorm_extras(input_shape):
    # LayerNorm gamma/beta along the normalized axis (assumed -1).
    c = input_shape[-1]
    return {
        '_lngamma': mx.nd.ones((c,), dtype='float32'),
        '_lnbeta':  mx.nd.zeros((c,), dtype='float32'),
    }


def _run_backward(build_symbol, input_shape, input_filler,
                  grad_req, grad_init_value):
    rng = np.random.RandomState(0)
    input_np = input_filler(rng)
    grad_init = np.full(input_shape, grad_init_value, dtype='float32')
    # Allocate the gradient buffer ourselves so we can observe it even when
    # the executor decides not to expose it via grad_dict (which happens
    # for grad_req='null').
    grad_buf = mx.nd.array(grad_init)
    x = mx.sym.Variable('x')
    out_sym = build_symbol(x)
    args = {'x': mx.nd.array(input_np)}
    args_grad = {'x': grad_buf}
    req_map = {'x': grad_req}
    for name in out_sym.list_arguments():
        if name in args:
            continue
        if name == '_lngamma' or name == '_lnbeta':
            extras = _layernorm_extras(input_shape)
            args[name] = extras[name]
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        else:
            raise NotImplementedError(
                f"Harness doesn't know how to provide variable {name!r}; "
                "extend _run_backward when adding an operator that needs it.")
    exe = out_sym._bind(ctx=mx.cpu(), args=args, args_grad=args_grad,
                        grad_req=req_map)
    exe.forward(is_train=True)
    # Backward with head_grad of ones makes the per-element backward
    # contribution depend only on the forward computation, which keeps
    # the test cheap and deterministic.
    head_grad = mx.nd.ones_like(exe.outputs[0])
    exe.backward(head_grad)
    mx.nd.waitall()
    return grad_buf.asnumpy(), grad_init


# ----------------------------------------------------------------------
# Contract tests.  Each parameter is one operator row.
# ----------------------------------------------------------------------


@pytest.mark.parametrize('build_symbol, input_shape, input_filler',
                         OPS_UNDER_CONTRACT)
def test_grad_req_null_preserves_sentinel(build_symbol, input_shape, input_filler):
    """``grad_req='null'`` must leave the gradient buffer untouched."""
    sentinel = 17.0
    grad, init = _run_backward(build_symbol, input_shape, input_filler,
                               grad_req='null', grad_init_value=sentinel)
    np.testing.assert_array_equal(grad, init,
        err_msg="grad_req='null' wrote to the gradient buffer")


@pytest.mark.parametrize('build_symbol, input_shape, input_filler',
                         OPS_UNDER_CONTRACT)
def test_grad_req_write_overwrites(build_symbol, input_shape, input_filler):
    """``grad_req='write'`` must overwrite a pre-filled sentinel."""
    sentinel = 99.0
    grad, _ = _run_backward(build_symbol, input_shape, input_filler,
                            grad_req='write', grad_init_value=sentinel)
    # The result must not be the sentinel everywhere (some elements may
    # coincide by chance, so reject the all-equal case).
    assert not np.allclose(grad, sentinel), \
        "grad_req='write' did not overwrite the sentinel buffer"


@pytest.mark.parametrize('build_symbol, input_shape, input_filler',
                         OPS_UNDER_CONTRACT)
def test_grad_req_add_accumulates(build_symbol, input_shape, input_filler):
    """``grad_req='add'`` must equal init + write result."""
    sentinel = 5.0
    grad_add, init = _run_backward(build_symbol, input_shape, input_filler,
                                   grad_req='add', grad_init_value=sentinel)
    grad_write, _ = _run_backward(build_symbol, input_shape, input_filler,
                                  grad_req='write', grad_init_value=0.0)
    np.testing.assert_allclose(grad_add, grad_write + sentinel,
        rtol=1e-5, atol=1e-5,
        err_msg="grad_req='add' did not equal init + write gradient")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
