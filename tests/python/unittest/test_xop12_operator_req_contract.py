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

def _rand_default_input(shape):
    def filler(rng):
        return rng.randn(*shape).astype('float32')
    return filler


def _rand_positive_input(shape):
    def filler(rng):
        # log/sqrt operators want positive inputs to avoid NaNs that would
        # mask a contract failure with a different failure.
        return (rng.rand(*shape) + 0.1).astype('float32')
    return filler


OPS_UNDER_CONTRACT = [
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
        # LayerNorm needs gamma/beta; freeze them inside the closure.
        lambda x: mx.sym.LayerNorm(
            data=x,
            gamma=mx.sym.Variable('_lngamma'),
            beta=mx.sym.Variable('_lnbeta'),
            axis=-1, eps=1e-5, output_mean_var=False),
        (2, 6), _rand_default_input((2, 6)),
        id='layernorm_axis_last',
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
    pytest.param(
        lambda x: mx.sym.sqrt(data=x),
        (4, 5), _rand_positive_input((4, 5)),
        id='sqrt_positive',
    ),
    pytest.param(
        lambda x: mx.sym.log(data=x),
        (4, 5), _rand_positive_input((4, 5)),
        id='log_positive',
    ),
    pytest.param(
        lambda x: mx.sym.broadcast_add(lhs=x, rhs=mx.sym.Variable('_bcast_rhs')),
        (4, 5), _rand_default_input((4, 5)),
        id='broadcast_add',
    ),
    pytest.param(
        lambda x: mx.sym.broadcast_mul(lhs=x, rhs=mx.sym.Variable('_bcast_rhs')),
        (4, 5), _rand_default_input((4, 5)),
        id='broadcast_mul',
    ),
    # XOP14 coverage: library-mapped operators with backward.
    pytest.param(
        lambda x: mx.sym.Convolution(
            data=x,
            weight=mx.sym.Variable('_convw'),
            bias=mx.sym.Variable('_convb'),
            kernel=(3, 3), stride=(1, 1), pad=(1, 1),
            num_filter=2),
        (1, 2, 5, 5), _rand_default_input((1, 2, 5, 5)),
        id='convolution_2d',
    ),
    pytest.param(
        lambda x: mx.sym.FullyConnected(
            data=x,
            weight=mx.sym.Variable('_fcw'),
            bias=mx.sym.Variable('_fcb'),
            num_hidden=4, no_bias=False, flatten=True),
        (3, 6), _rand_default_input((3, 6)),
        id='fully_connected',
    ),
    pytest.param(
        lambda x: mx.sym.Pooling(
            data=x, kernel=(2, 2), stride=(2, 2), pool_type='avg'),
        (1, 2, 4, 4), _rand_default_input((1, 2, 4, 4)),
        id='pooling_avg_2x2',
    ),
    pytest.param(
        lambda x: mx.sym.Pooling(
            data=x, kernel=(2, 2), stride=(2, 2), pool_type='max'),
        (1, 2, 4, 4), _rand_default_input((1, 2, 4, 4)),
        id='pooling_max_2x2',
    ),
    pytest.param(
        # Concat across axis 1: backward must produce a gradient for each
        # input independently — exercises the concat-split copyback path.
        lambda x: mx.sym.concat(
            x, mx.sym.Variable('_concat_rhs'), dim=1),
        (2, 3), _rand_default_input((2, 3)),
        id='concat_axis_1',
    ),
    # Note: BatchNorm needs an aux-state-aware binding pass, Embedding
    # rejects data gradient by design, and Dropout's stochastic backward
    # would fail the "add == init + write" sanity check.  Each gets its
    # own dedicated contract test alongside this harness.
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
        elif name == '_bcast_rhs':
            # Broadcast helpers in OPS_UNDER_CONTRACT use a per-row scalar
            # rhs so the broadcast actually does something but the gradient
            # math stays simple.
            rhs = np.full((1, input_shape[-1]), 0.5, dtype='float32')
            args[name] = mx.nd.array(rhs)
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        elif name in ('_bngamma', '_bnbeta', '_bnmean', '_bnvar'):
            # BatchNorm gamma/beta/moving-mean/moving-var; size = C (axis 1
            # of the NCHW input).
            c = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
            if name in ('_bngamma', '_bnvar'):
                buf = mx.nd.ones((c,), dtype='float32')
            else:
                buf = mx.nd.zeros((c,), dtype='float32')
            args[name] = buf
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        elif name in ('_convw', '_convb'):
            # 2x2 stride-1 pad-1 conv with num_filter=2 and 2-channel input.
            if name == '_convw':
                args[name] = mx.nd.ones((2, input_shape[1], 3, 3), dtype='float32') * 0.1
            else:
                args[name] = mx.nd.zeros((2,), dtype='float32')
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        elif name in ('_fcw', '_fcb'):
            # FullyConnected: weight is (num_hidden, in_units).
            in_units = int(np.prod(input_shape[1:])) if len(input_shape) > 1 \
                else int(input_shape[0])
            if name == '_fcw':
                args[name] = mx.nd.ones((4, in_units), dtype='float32') * 0.1
            else:
                args[name] = mx.nd.zeros((4,), dtype='float32')
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        elif name == '_embw':
            args[name] = mx.nd.ones((10, 4), dtype='float32') * 0.1
            args_grad[name] = mx.nd.zeros_like(args[name])
            req_map[name] = 'null'
        elif name == '_concat_rhs':
            args[name] = mx.nd.ones(input_shape, dtype='float32') * 0.25
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


# ----------------------------------------------------------------------
# Aux-state preservation dimension.
#
# Some operators (BatchNorm, SyncBatchNorm, LSTM/GRU when using statefulness
# helpers) have *aux states* — buffers like running_mean/running_var that
# the executor updates in training mode and must NOT touch in inference
# mode. A regression where inference-mode forward writes to running_mean
# would silently shift the model's outputs over iterations.
#
# This is orthogonal to grad_req for the main parameters: aux states are
# their own update channel. We exercise the contract for BatchNorm here.
# ----------------------------------------------------------------------


def test_batchnorm_inference_mode_does_not_mutate_running_stats():
    """BatchNorm in inference mode (use_global_stats=True, autograd off)
    must leave running_mean / running_var untouched."""
    rng = np.random.RandomState(0)
    x_np = rng.randn(8, 4, 16, 16).astype('float32')
    gamma_np = np.ones(4, dtype='float32')
    beta_np = np.zeros(4, dtype='float32')
    running_mean_np = rng.randn(4).astype('float32')
    running_var_np = (rng.rand(4) + 0.5).astype('float32')

    rm_before = running_mean_np.copy()
    rv_before = running_var_np.copy()

    # Imperative inference-mode call.
    data = mx.nd.array(x_np)
    gamma = mx.nd.array(gamma_np)
    beta = mx.nd.array(beta_np)
    running_mean = mx.nd.array(running_mean_np)
    running_var = mx.nd.array(running_var_np)
    out = mx.nd.BatchNorm(
        data=data, gamma=gamma, beta=beta,
        moving_mean=running_mean, moving_var=running_var,
        use_global_stats=True, fix_gamma=False)
    out.wait_to_read()
    np.testing.assert_array_equal(
        running_mean.asnumpy(), rm_before,
        err_msg="BatchNorm inference mode mutated running_mean")
    np.testing.assert_array_equal(
        running_var.asnumpy(), rv_before,
        err_msg="BatchNorm inference mode mutated running_var")


def test_batchnorm_training_mode_updates_running_stats():
    """BatchNorm in training mode (use_global_stats=False) must update
    running_mean / running_var (the inverse of the inference contract)."""
    rng = np.random.RandomState(0)
    x_np = rng.randn(8, 4, 16, 16).astype('float32')
    gamma_np = np.ones(4, dtype='float32')
    beta_np = np.zeros(4, dtype='float32')
    running_mean_np = np.zeros(4, dtype='float32')
    running_var_np = np.ones(4, dtype='float32')

    data = mx.nd.array(x_np)
    gamma = mx.nd.array(gamma_np)
    beta = mx.nd.array(beta_np)
    running_mean = mx.nd.array(running_mean_np)
    running_var = mx.nd.array(running_var_np)
    with mx.autograd.record():
        out = mx.nd.BatchNorm(
            data=data, gamma=gamma, beta=beta,
            moving_mean=running_mean, moving_var=running_var,
            use_global_stats=False, fix_gamma=False)
    out.wait_to_read()
    # Running stats must change away from their initial (0, 1).
    assert not np.allclose(running_mean.asnumpy(), 0.0), \
        "BatchNorm training mode did not update running_mean"
    assert not np.allclose(running_var.asnumpy(), 1.0), \
        "BatchNorm training mode did not update running_var"


# ----------------------------------------------------------------------
# kWriteInplace contract sanity check.
#
# kWriteInplace is set by the executor's storage allocator when input
# and output of an op are scheduled to share a buffer. Pure Python can't
# force kWriteInplace via Symbol._bind args_grad, but we can verify that
# operators which declare inplace_pairs (Activation is the canonical one)
# don't crash when imperative API computes them with the result reused
# as the next input — the executor's storage planner reaches the same
# kWriteInplace path.
# ----------------------------------------------------------------------


def test_activation_inplace_chain_does_not_corrupt_output():
    """Chaining relu(relu(relu(x))) should produce the same result as a
    single relu(x) for positive x — a regression where kWriteInplace
    accidentally aliased an intermediate would show up as wrong values."""
    rng = np.random.RandomState(0)
    x_np = rng.randn(4, 8).astype('float32')
    x = mx.nd.array(x_np)
    chained = mx.nd.Activation(
        mx.nd.Activation(mx.nd.Activation(x, act_type='relu'),
                         act_type='relu'),
        act_type='relu')
    direct = mx.nd.Activation(x, act_type='relu')
    chained.wait_to_read()
    direct.wait_to_read()
    np.testing.assert_array_equal(
        chained.asnumpy(), direct.asnumpy(),
        err_msg="kWriteInplace optimization corrupted Activation chain")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
