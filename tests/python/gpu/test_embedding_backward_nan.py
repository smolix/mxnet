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

"""Stress-test reproducer for Apache MXNet issue #11314.

AddTakeGradLargeBatchCaller (and the v2-era EmbeddingGradKernel on GPU)
have been reported to intermittently emit NaN at random positions in the
weight gradient on wide-Embedding backward passes.  This test exercises
the kernel hard on Blackwell (sm_120) under CUDA 13 to confirm or refute
the failure mode.

Two scenarios per iteration:
  (a) grad_weight pre-filled with zeros (the normal kAddTo path)
  (b) grad_weight pre-filled with a "trap" pattern (0xFF = NaN-ish bit
      pattern for fp32) to expose any uninitialised-read accumulation.
"""

import os
import sys

import numpy as np
import pytest

import mxnet as mx
from mxnet import autograd, gluon, nd


# Pick a GPU.  The sweep usually has one GPU busy; let the runner pick
# via CUDA_VISIBLE_DEVICES.  We treat device 0 inside the visible set.
CTX = mx.gpu(0)


def _alloc_trap_grad(shape, dtype, fill_byte):
    """Allocate an ndarray on CTX, fill every byte with `fill_byte`.

    fill_byte=0x00  -> zero grad (clean kAddTo start)
    fill_byte=0xFF  -> for fp32 this is -NaN bit pattern, so any uninit
                       read into the accumulator will propagate NaN.
    """
    arr = nd.zeros(shape, ctx=CTX, dtype=dtype)
    if fill_byte != 0x00:
        # poke raw bytes via numpy
        host = np.full(shape, 0, dtype=dtype)
        host_bytes = host.view(np.uint8)
        host_bytes[...] = fill_byte
        arr[:] = nd.array(host, ctx=CTX, dtype=dtype)
    arr.wait_to_read()
    return arr


def _run_one_iteration(vocab, dim, batch, seq, dtype, fill_byte, seed):
    rs = np.random.RandomState(seed)
    idx_np = rs.randint(0, vocab, size=(batch, seq)).astype(np.int32)

    # Random weight (small magnitude so any NaN is from the kernel,
    # not from numerical overflow).
    w_np = rs.randn(vocab, dim).astype(dtype) * 0.01
    weight = nd.array(w_np, ctx=CTX, dtype=dtype)
    weight.attach_grad()

    # Pre-fill the grad buffer with the trap pattern.  attach_grad has
    # already allocated `weight.grad`; overwrite its raw bytes.
    if fill_byte == 0x00:
        weight.grad[:] = 0
    else:
        host = np.full((vocab, dim), 0, dtype=dtype)
        host.view(np.uint8)[...] = fill_byte
        weight.grad[:] = nd.array(host, ctx=CTX, dtype=dtype)
    weight.grad.wait_to_read()

    idx = nd.array(idx_np, ctx=CTX, dtype=np.int32)

    with autograd.record():
        out = nd.Embedding(data=idx, weight=weight,
                           input_dim=vocab, output_dim=dim)
        # Pretend a downstream loss = sum(out); upstream grad is all 1s,
        # but we go through full autograd so the path matches real use.
        loss = out.sum()
    loss.backward()

    g = weight.grad
    g.wait_to_read()
    nan_count = int(nd.contrib.isnan(g).sum().asscalar()) \
        if hasattr(nd.contrib, "isnan") else \
        int(np.isnan(g.asnumpy()).sum())
    inf_count = int(np.isinf(g.asnumpy()).sum()) if nan_count == 0 else -1
    return nan_count, inf_count


@pytest.mark.parametrize("fill_byte", [0x00, 0xFF])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_embedding_backward_no_nan(fill_byte, dtype):
    """Run a wide-embedding backward 100 times; assert no NaN anywhere."""
    vocab = 50000
    dim = 256
    batch = 8192
    seq = 64
    iters = int(os.environ.get("EMB_NAN_ITERS", "100"))

    failures = []
    for i in range(iters):
        nan_count, inf_count = _run_one_iteration(
            vocab, dim, batch, seq, dtype, fill_byte, seed=1000 + i)
        if nan_count > 0 or inf_count > 0:
            failures.append((i, nan_count, inf_count))
            # Print early so we see it in the log even on subsequent passes
            print(f"  iter {i}: nan={nan_count} inf={inf_count}",
                  file=sys.stderr, flush=True)

    assert not failures, (
        f"Embedding backward produced NaN/Inf in "
        f"{len(failures)}/{iters} iterations "
        f"(dtype={np.dtype(dtype).name}, fill=0x{fill_byte:02x}). "
        f"First failure: iter={failures[0][0]} nan={failures[0][1]} "
        f"inf={failures[0][2]}")


if __name__ == "__main__":
    # Quick CLI mode (used by the bash repro loop, not by pytest).
    vocab, dim, batch, seq = 50000, 256, 8192, 64
    iters = int(os.environ.get("EMB_NAN_ITERS", "100"))
    dtype = np.float32
    overall = 0
    for fb_name, fb in [("zero", 0x00), ("trap", 0xFF)]:
        fail = 0
        for i in range(iters):
            nan, inf = _run_one_iteration(vocab, dim, batch, seq, dtype,
                                          fb, seed=1000 + i)
            if nan or inf:
                fail += 1
                print(f"  [{fb_name}] iter {i}: nan={nan} inf={inf}",
                      flush=True)
        print(f"[{fb_name}] fill=0x{fb:02x}: {fail}/{iters} iters had "
              f"NaN/Inf", flush=True)
        overall += fail
    sys.exit(1 if overall else 0)
