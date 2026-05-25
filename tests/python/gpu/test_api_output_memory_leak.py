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

"""GPU regression tests for API wrappers that return ADTs of NDArrays.

The leak fixed by `CreateADTFromOutputVector` was not a cuDNN, autograd, or
scheduler leak.  It was an ownership leak in Python API wrappers that returned
multiple NDArrays through a runtime ADT: the wrapper copied each freshly
allocated `NDArray*` output into an `NDArrayHandle` object and then dropped the
original heap handle.

These tests use fixed-shape workloads and empty the GPU storage cache between
iterations.  A real leaked output handle remains live and cannot be reclaimed
by `empty_cache`, so the pre-fix code grows roughly by output size every
iteration.  The fixed code grows during warmup only and then plateaus.
"""

import gc

import pytest

import mxnet as mx
from mxnet import np, npx
from mxnet.gluon import rnn


pytestmark = pytest.mark.skipif(mx.context.num_gpus() == 0, reason="requires GPU")


def _sync_gc_empty_cache(ctx):
    mx.npx.waitall()
    gc.collect()
    ctx.empty_cache()
    mx.npx.waitall()


def _used_bytes(device_id=0):
    free, total = mx.context.gpu_memory_info(device_id)
    return total - free


def _wait_first_output(outputs):
    if hasattr(outputs, "wait_to_read"):
        outputs.wait_to_read()
    else:
        outputs[0].wait_to_read()


def _assert_no_gpu_output_leak(label, ctx, make_outputs, iterations=35, allowed_growth_mib=80):
    # Warm up operator initialization, cuDNN descriptors, and one-time runtime
    # allocations before taking the baseline.
    for _ in range(3):
        outputs = make_outputs()
        _wait_first_output(outputs)
        del outputs
    _sync_gc_empty_cache(ctx)
    before = _used_bytes(ctx.device_id)

    for _ in range(iterations):
        outputs = make_outputs()
        _wait_first_output(outputs)
        del outputs
        _sync_gc_empty_cache(ctx)

    after = _used_bytes(ctx.device_id)
    growth_mib = (after - before) / (1 << 20)
    print(f"[{label}] before={before / (1 << 20):.1f} MiB "
          f"after={after / (1 << 20):.1f} MiB growth={growth_mib:.1f} MiB")
    assert growth_mib <= allowed_growth_mib, (
        f"{label} leaked GPU memory: grew by {growth_mib:.1f} MiB over "
        f"{iterations} iterations, allowed {allowed_growth_mib} MiB"
    )


@pytest.mark.timeout(300)
def test_npx_rnn_state_outputs_adt_does_not_leak_gpu_outputs():
    """Direct regression for `_npx.rnn(..., state_outputs=True)`.

    With the old wrapper this leaks the visible output NDArray each call.  The
    tensor size below is ~6.25 MiB, so 35 iterations would grow by more than
    200 MiB pre-fix.
    """
    npx.set_np()
    ctx = mx.gpu(0)
    state_size = 100
    batch_size = 32
    seq_len = 256
    input_size = 100
    num_layers = 2
    directions = 2
    param_size = (
        directions * 4 * state_size * (input_size + state_size + 2) +
        directions * 4 * state_size * (directions * state_size + state_size + 2)
    )
    params = np.random.randn(param_size, device=ctx).astype("float32")
    state = np.zeros((directions * num_layers, batch_size, state_size),
                     device=ctx, dtype="float32")
    state_cell = np.zeros_like(state)

    def make_outputs():
        data = np.random.randn(seq_len, batch_size, input_size, device=ctx).astype("float32")
        return npx.rnn(
            data, params, state, state_cell,
            state_size=state_size, num_layers=num_layers, bidirectional=True,
            p=0, mode="lstm", state_outputs=True)

    _assert_no_gpu_output_leak("npx.rnn-state-outputs", ctx, make_outputs)


@pytest.mark.timeout(300)
def test_gluon_lstm_adt_outputs_do_not_leak_gpu_outputs():
    """End-user Gluon path used by D2L's sentiment-analysis RNN notebook."""
    npx.set_np()
    ctx = mx.gpu(0)
    encoder = rnn.LSTM(100, num_layers=2, bidirectional=True, input_size=100)
    encoder.initialize(ctx=ctx)

    def make_outputs():
        data = np.random.randn(256, 32, 100, device=ctx).astype("float32")
        return encoder(data)

    _assert_no_gpu_output_leak("gluon.lstm", ctx, make_outputs)


@pytest.mark.timeout(300)
def test_representative_multi_output_adt_wrappers_do_not_leak_gpu_outputs():
    """Exercise non-RNN ADT-returning wrappers fixed by the same helper."""
    npx.set_np()
    ctx = mx.gpu(0)
    topk_data = np.random.randn(2048, 1024, device=ctx).astype("float32")
    ln_data = np.random.randn(256, 64, 100, device=ctx).astype("float32")
    gamma = np.ones((100,), device=ctx, dtype="float32")
    beta = np.zeros((100,), device=ctx, dtype="float32")

    def make_outputs():
        top_values, top_indices = npx.topk(
            topk_data, axis=1, k=512, ret_typ="both", dtype="float32")
        normed, mean, std = npx.layer_norm(
            ln_data, gamma, beta, axis=-1, eps=1e-5, output_mean_var=True)
        return top_values, top_indices, normed, mean, std

    _assert_no_gpu_output_leak("multi-output-adt-wrappers", ctx, make_outputs)
