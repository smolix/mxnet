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

"""Single-process / multi-GPU NCCL kvstore smoke tests.

Tests the 'nccl' KVStore type (KVStoreNCCL / all-reduce within one process).
All tests skip cleanly when fewer than 2 GPUs are available.

Supported dtypes for NCCL kvstore push/pull:
  float32, float16, float64, uint8, int32, int64
NOT supported: int8 (signed) — kvstore_nccl.h GetNCCLType() has no case for it.

Important constraint: NCCL kvstore does NOT support adding new keys after
push() has been called on any key.  Once NCCL communicators are initialized
(on first push), the key set is frozen.  All keys must be init()'d before
the first push().
"""

import time
import numpy as np
import pytest
import mxnet as mx

NUM_GPUS = mx.device.num_gpus()

pytestmark = [
    pytest.mark.skipif(
        NUM_GPUS < 2,
        reason=f"NCCL tests require >= 2 GPUs, found {NUM_GPUS}"
    ),
    pytest.mark.skipif(
        not mx.runtime.Features().is_enabled("NCCL"),
        reason="MXNet was built without NCCL"
    ),
]


def _make_kv():
    """Create a fresh NCCL kvstore instance."""
    return mx.kv.create('nccl')


# ---------------------------------------------------------------------------
# Helper: single-key push-pull with a fresh kvstore each time
# ---------------------------------------------------------------------------
def _push_pull_assert_fresh(shape, dtype):
    """Push dtype arrays from each GPU, pull back, verify all-reduce sum."""
    kv = _make_kv()
    key = 1  # integer key; fresh kv per call
    expected = sum(g + 1 for g in range(NUM_GPUS))  # sum 1..NUM_GPUS

    arr_list = [
        mx.nd.ones(shape, mx.gpu(g), dtype=dtype) * (g + 1)
        for g in range(NUM_GPUS)
    ]
    kv.init(key, mx.nd.ones(shape, mx.gpu(0), dtype=dtype))
    res = [mx.nd.zeros(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]
    kv.push(key, arr_list)
    kv.pull(key, res)
    mx.nd.waitall()

    for g in range(NUM_GPUS):
        got = res[g].asnumpy()
        if np.issubdtype(got.dtype, np.integer):
            assert (got == expected).all(), \
                f"GPU {g}: expected {expected}, got mean={got.mean()}"
        else:
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4), \
                f"GPU {g}: expected {expected:.4f}, got mean={got.mean():.6f}"


# ---------------------------------------------------------------------------
# Test 1: float32, small tensor (1024 elements = 4 KiB)
# ---------------------------------------------------------------------------
def test_nccl_float32_small():
    _push_pull_assert_fresh((1024,), 'float32')


# ---------------------------------------------------------------------------
# Test 2: float32, large tensor (1M elements = 4 MiB)
# ---------------------------------------------------------------------------
def test_nccl_float32_large():
    _push_pull_assert_fresh((1024 * 1024,), 'float32')


# ---------------------------------------------------------------------------
# Test 3: float16 push/pull
# ---------------------------------------------------------------------------
def test_nccl_float16():
    _push_pull_assert_fresh((1024,), 'float16')


# ---------------------------------------------------------------------------
# Test 4: uint8 push/pull (int8/signed is NOT supported by NCCL kvstore)
# ---------------------------------------------------------------------------
def test_nccl_uint8():
    kv = _make_kv()
    shape = (1024,)
    dtype = 'uint8'
    key = 1
    # use value=1 on each GPU so sum = NUM_GPUS, stays within uint8 range
    arr_list = [mx.nd.ones(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]
    kv.init(key, mx.nd.ones(shape, mx.gpu(0), dtype=dtype))
    res = [mx.nd.zeros(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]
    kv.push(key, arr_list)
    kv.pull(key, res)
    mx.nd.waitall()
    expected = NUM_GPUS
    for g in range(NUM_GPUS):
        got = res[g].asnumpy()
        assert (got == expected).all(), \
            f"GPU {g}: expected {expected}, got mean={got.mean()}"


# ---------------------------------------------------------------------------
# Test 5: multi-dimensional shape
# ---------------------------------------------------------------------------
def test_nccl_multidim_shape():
    _push_pull_assert_fresh((4, 32, 32), 'float32')


# ---------------------------------------------------------------------------
# Test 6: multiple keys in a single kvstore
#
# Constraint: ALL keys must be init()'d before any push().  Adding a new key
# after push() has been called is not supported by KVStoreNCCL.
# ---------------------------------------------------------------------------
def test_nccl_multiple_keys():
    kv = _make_kv()
    shape = (512,)
    dtype = 'float32'
    keys = [1, 2, 3, 4]
    expected = sum(g + 1 for g in range(NUM_GPUS))

    # Init ALL keys before any push
    for key in keys:
        kv.init(key, mx.nd.ones(shape, mx.gpu(0), dtype=dtype))

    # Now push+pull each key
    for key in keys:
        arr_list = [mx.nd.ones(shape, mx.gpu(g), dtype=dtype) * (g + 1)
                    for g in range(NUM_GPUS)]
        res = [mx.nd.zeros(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]
        kv.push(key, arr_list)
        kv.pull(key, res)
        mx.nd.waitall()
        for g in range(NUM_GPUS):
            got = res[g].asnumpy()
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4), \
                f"key={key} GPU {g}: expected {expected:.4f}, got mean={got.mean():.6f}"


# ---------------------------------------------------------------------------
# Test 7: bandwidth measurement at 1 MiB, 16 MiB, 256 MiB
#         (informational, not a correctness assertion)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size_mib", [1, 16, 256])
def test_nccl_bandwidth(size_mib, request):
    """Measure push+pull round-trip bandwidth.  No hard threshold — prints GB/s."""
    n_warmup = 5
    n_bench = 20 if size_mib <= 16 else 5
    dtype = 'float32'
    elem_size = 4
    n_elems = (size_mib * 1024 * 1024) // elem_size
    shape = (n_elems,)
    key = 1  # integer key; fresh kv per parametrize call

    kv = _make_kv()
    kv.init(key, mx.nd.ones(shape, mx.gpu(0), dtype=dtype))
    arr_list = [mx.nd.ones(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]
    res = [mx.nd.zeros(shape, mx.gpu(g), dtype=dtype) for g in range(NUM_GPUS)]

    for _ in range(n_warmup):
        kv.push(key, arr_list)
        kv.pull(key, res)
    mx.nd.waitall()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        kv.push(key, arr_list)
        kv.pull(key, res)
    mx.nd.waitall()
    elapsed = time.perf_counter() - t0

    # push + pull = 2 * tensor_bytes bidirectional per iteration
    total_bytes = n_bench * 2 * (n_elems * elem_size)
    bw_gbps = total_bytes / elapsed / 1e9

    print(f"\n  [{size_mib:4d} MiB] NCCL push+pull bandwidth: {bw_gbps:.2f} GB/s "
          f"({n_bench} iters, {elapsed:.3f}s)")

    # Store the metric for optional reporting; correctness is covered above.
    request.node._nccl_bw = (size_mib, bw_gbps)


# ---------------------------------------------------------------------------
# Test 8: int8 NOT supported (MUST BE LAST — see warning below)
#
# Pushing a dtype not in kvstore_nccl.h's GetNCCLType() switch causes an
# async error that is raised at waitall() time.  This error corrupts the
# MXNet NCCL async stream state for the remainder of the process: subsequent
# NCCL push/pull calls on any kvstore in the same process will return zeroed
# results instead of raising.  There is no public API to reset this state
# short of process exit.
#
# This test is placed LAST in this file.  Do not move it earlier.
# ---------------------------------------------------------------------------
def test_nccl_int8_not_supported():
    kv = _make_kv()
    shape = (64,)
    key = 1
    # init as float32 first; then push int8 — type mismatch triggers async error
    kv.init(key, mx.nd.ones(shape, mx.gpu(0)))
    arr_list = [mx.nd.ones(shape, mx.gpu(g), dtype='int8') for g in range(NUM_GPUS)]
    with pytest.raises(mx.base.MXNetError, match="Unknown type passed to NCCL KVStore"):
        kv.push(key, arr_list)
        mx.nd.waitall()
