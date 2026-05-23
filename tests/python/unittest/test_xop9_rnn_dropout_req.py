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

"""XOP9 RNN dropout req coverage.

`mx.sym.RNN(..., dropout=...)` carries reserve-space state for the dropout
mask between the cuDNN forward and backward passes.  The XOP9 audit closed
plain Dropout's kNullOp/kAddTo handling, but the RNN dropout path uses
its own reserve-space state and needed dedicated coverage.

Backward sentinel/accumulate contract per grad_req:

- `null`: data gradient buffer is untouched
- `write`: buffer is overwritten with the new gradient (no leftover sentinel)
- `add`:  buffer equals `init + new_gradient`

These tests exercise the RNN forward+backward with `dropout > 0` on CPU
(native path).  GPU cuDNN RNN dropout would need its own pass once a
matching reserve-space contract test infrastructure exists.
"""

import numpy as np
import pytest

import mxnet as mx


@pytest.fixture(autouse=True)
def _deterministic_rng():
    mx.random.seed(0)
    np.random.seed(0)
    yield


def _run_rnn_backward(grad_req, init_value, *, mode='lstm', dropout=0.3):
    """Run RNN forward+backward on CPU with the given data grad_req.

    Returns the data-gradient buffer after backward.

    Reseeds RNG inside the helper so repeated calls produce identical
    parameters/data — required for the kAddTo accumulation invariant
    (`add == init + write`) which compares two runs.
    """
    np.random.seed(0)
    mx.random.seed(0)
    seq_len, batch, hidden, layers = 4, 2, 3, 1
    # input feature size for vanilla RNN-tanh equals hidden_size.
    in_size = hidden
    data_np = np.random.randn(seq_len, batch, in_size).astype('float32')
    if mode == 'lstm':
        gates = 4
    elif mode == 'gru':
        gates = 3
    else:
        gates = 1
    weight_size = (in_size * hidden +
                   hidden * hidden +
                   2 * hidden) * gates * layers
    parameters_np = np.random.randn(weight_size).astype('float32') * 0.1
    state_np = np.zeros((layers, batch, hidden), dtype='float32')
    state_cell_np = np.zeros((layers, batch, hidden), dtype='float32')

    data_sym = mx.sym.Variable('data')
    params_sym = mx.sym.Variable('parameters')
    state_sym = mx.sym.Variable('state')
    args = {
        'data': mx.nd.array(data_np),
        'parameters': mx.nd.array(parameters_np),
        'state': mx.nd.array(state_np),
    }
    args_grad = {
        'data': mx.nd.array(np.full(data_np.shape, init_value, dtype='float32')),
        'parameters': mx.nd.zeros_like(args['parameters']),
        'state': mx.nd.zeros_like(args['state']),
    }
    req_map = {'data': grad_req, 'parameters': 'null', 'state': 'null'}

    extra_kwargs = {}
    if mode == 'lstm':
        state_cell_sym = mx.sym.Variable('state_cell')
        args['state_cell'] = mx.nd.array(state_cell_np)
        args_grad['state_cell'] = mx.nd.zeros_like(args['state_cell'])
        req_map['state_cell'] = 'null'
        extra_kwargs['state_cell'] = state_cell_sym

    out_sym = mx.sym.RNN(
        data=data_sym,
        parameters=params_sym,
        state=state_sym,
        state_size=hidden,
        num_layers=layers,
        mode=mode,
        p=dropout,
        **extra_kwargs,
    )

    exe = out_sym._bind(ctx=mx.cpu(), args=args, args_grad=args_grad,
                        grad_req=req_map)
    exe.forward(is_train=True)
    head_grad = mx.nd.ones_like(exe.outputs[0])
    exe.backward(head_grad)
    mx.nd.waitall()
    return args_grad['data'].asnumpy()


@pytest.mark.parametrize('mode', ['rnn_tanh', 'rnn_relu', 'gru', 'lstm'])
def test_rnn_dropout_grad_req_null_preserves_sentinel(mode):
    """grad_req='null' must leave the data-gradient buffer untouched even
    when the RNN's dropout reserve-space state is being threaded through."""
    sentinel = 13.0
    grad = _run_rnn_backward('null', sentinel, mode=mode)
    np.testing.assert_array_equal(
        grad, np.full_like(grad, sentinel),
        err_msg=f"RNN[{mode}] dropout: grad_req='null' wrote to data grad buffer")


@pytest.mark.parametrize('mode', ['rnn_tanh', 'rnn_relu', 'gru', 'lstm'])
def test_rnn_dropout_grad_req_write_overwrites(mode):
    """grad_req='write' must overwrite the sentinel-filled data-gradient
    buffer with the new gradient."""
    sentinel = 99.0
    grad = _run_rnn_backward('write', sentinel, mode=mode)
    assert not np.allclose(grad, sentinel), \
        f"RNN[{mode}] dropout: grad_req='write' did not overwrite sentinel"


@pytest.mark.parametrize('mode', ['rnn_tanh', 'rnn_relu', 'gru', 'lstm'])
def test_rnn_grad_req_add_accumulates_no_dropout(mode):
    """grad_req='add' must equal init + write gradient.

    Dropout is disabled (`p=0`) for this check because RNN dropout's mask
    is sampled non-deterministically between calls, so two runs with
    different grad_req would observe different masks and break the
    `init + write` invariant.  The kNullOp / kWriteTo parameterized tests
    above still exercise the dropout-enabled path; this test isolates the
    accumulation semantics."""
    sentinel = 5.0
    add_grad = _run_rnn_backward('add', sentinel, mode=mode, dropout=0.0)
    write_grad = _run_rnn_backward('write', 0.0, mode=mode, dropout=0.0)
    np.testing.assert_allclose(
        add_grad, write_grad + sentinel, rtol=1e-5, atol=1e-5,
        err_msg=f"RNN[{mode}]: grad_req='add' did not equal "
                "init + write gradient")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
