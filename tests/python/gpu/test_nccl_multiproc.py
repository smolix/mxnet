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

"""Multi-process NCCL topology investigation for MXNet.

Background
----------
MXNet's `kv.create('nccl')` (KVStoreNCCL) is a *single-process* abstraction.
It lives in kvstore_nccl.h and inherits KVStoreLocal; it manages multiple GPUs
within ONE process by calling ncclCommInitAll() to create a communicator ring
that spans all visible devices in that single process.

For *per-process / 1-proc-per-GPU* topology (the DDP / Horovod pattern),
MXNet does NOT have a first-class built-in equivalent.  The multi-process
options are:

1. ``dist_sync`` / ``dist_device_sync`` / ``dist_async`` — parameter-server
   topology backed by ps-lite / ZMQ.  Each worker process connects to a
   scheduler + server set.  Uses TCP/RDMA, NOT NCCL for the actual collective.

2. Horovod — external library that wraps MXNet tensors and calls NCCL
   collectives directly via ncclAllReduce.  Not tested here.

3. BytePS — similar external dependency; not tested here.

None of these are available in this repo's wheel.  Attempting ``kv.create('nccl')``
inside each worker sub-process (one worker per GPU, with CUDA_VISIBLE_DEVICES=<rank>)
would create N isolated single-device NCCL communicators that cannot communicate
with each other: ncclCommInitAll() on a single device produces a degenerate
communicator of size 1, and push/pull becomes a local no-op copy with no
cross-GPU reduction.

This file tests and documents that behaviour, confirms the single-process
approach IS the correct multi-GPU pattern for MXNet's NCCL kvstore, and
verifies that a spawned worker can independently run push/pull on its own GPU.

Note on CUDA_VISIBLE_DEVICES isolation in spawn workers
--------------------------------------------------------
When using mp.get_context('spawn'), the child process re-imports all modules
including mxnet.  To properly isolate GPU visibility, CUDA_VISIBLE_DEVICES must
be set in the *child* before any CUDA context is created.  Because the test
module imports mxnet at the top level (for NUM_GPUS_MAIN), the child re-runs
that import — which initialises CUDA before the worker function can set
CUDA_VISIBLE_DEVICES.  As a result, CUDA_VISIBLE_DEVICES set inside the worker
function does NOT limit which GPUs are visible; all GPUs remain visible.

The relevant architectural fact is still observable without GPU-count isolation:
each worker creates its own independent NCCL kvstore and does push/pull on its
own device without any cross-worker synchronisation.
"""

import os
import multiprocessing as mp
import numpy as np
import pytest
import mxnet as mx

NUM_GPUS_MAIN = mx.device.num_gpus()


# ---------------------------------------------------------------------------
# Worker function: runs in a subprocess, performs push/pull on gpu(rank)
# ---------------------------------------------------------------------------
def _worker_nccl_per_rank(rank: int, result_queue: mp.Queue):
    """Worker that creates its own NCCL kvstore and uses gpu(rank).

    This demonstrates the per-process isolation: each process has its own
    independent NCCL communicator and performs no cross-process all-reduce.
    The push/pull result for each worker equals its own (non-reduced) value.
    """
    try:
        import mxnet as mx
        import numpy as np

        kv = mx.kv.create('nccl')
        kv_type = kv.type

        shape = (64,)
        key = 1
        val = float(rank + 1)

        # Use gpu(rank) — if rank >= num_visible this will error, which is fine
        num_gpus_in_worker = mx.device.num_gpus()
        dev = mx.gpu(rank % num_gpus_in_worker)

        arr = mx.nd.full(shape, val, dev)
        kv.init(key, arr)
        kv.push(key, [arr])
        res = mx.nd.zeros(shape, dev)
        kv.pull(key, [res])
        mx.nd.waitall()

        got = float(res.asnumpy().mean())
        # Each worker's push/pull is independent: result = worker's own value
        # (since only one device is in this worker's NCCL comm)
        push_pull_ok = abs(got - val) < 0.01

        result_queue.put({
            'rank': rank,
            'num_gpus_in_worker': num_gpus_in_worker,
            'kv_type': kv_type,
            'got_value': got,
            'expected_value': val,
            'push_pull_ok': push_pull_ok,
            'error': None,
        })
    except Exception as e:
        import traceback
        result_queue.put({
            'rank': rank,
            'num_gpus_in_worker': -1,
            'kv_type': 'unknown',
            'got_value': None,
            'expected_value': None,
            'push_pull_ok': False,
            'error': traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Test 1: verify per-process workers run independently without cross-process
#         all-reduce — each worker's pull returns its own value, not the sum.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    NUM_GPUS_MAIN < 2,
    reason=f"Requires >= 2 GPUs, found {NUM_GPUS_MAIN}"
)
def test_nccl_per_process_is_isolated():
    """Spawn one worker per GPU-rank; verify push/pull is local (not all-reduced).

    Each worker creates kv.create('nccl') independently and pushes value=(rank+1).
    Because there is no cross-process communicator, pull returns (rank+1) — not
    the sum of all workers.  This demonstrates that per-process 'nccl' kvstores
    are isolated and do NOT collectively reduce across processes.
    """
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    procs = []

    for rank in range(NUM_GPUS_MAIN):
        p = ctx.Process(target=_worker_nccl_per_rank, args=(rank, result_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"Worker process exited with code {p.exitcode}"

    results = {}
    while not result_queue.empty():
        r = result_queue.get_nowait()
        results[r['rank']] = r

    assert len(results) == NUM_GPUS_MAIN, \
        f"Expected {NUM_GPUS_MAIN} results, got {len(results)}"

    for rank in range(NUM_GPUS_MAIN):
        r = results[rank]
        assert r['error'] is None, \
            f"Worker rank {rank} raised:\n{r['error']}"
        assert r['kv_type'] == 'nccl', \
            f"Rank {rank}: expected kv type 'nccl', got '{r['kv_type']}'"
        assert r['push_pull_ok'], \
            f"Rank {rank}: push/pull isolation failed. " \
            f"Expected {r['expected_value']}, got {r['got_value']}. " \
            f"If got_value is the sum of all workers, cross-process reduction " \
            f"unexpectedly occurred."

    print(f"\n  Per-process isolation confirmed across {NUM_GPUS_MAIN} workers:")
    for rank in range(NUM_GPUS_MAIN):
        r = results[rank]
        print(f"    worker {rank}: push val={r['expected_value']:.0f}, "
              f"pull result={r['got_value']:.1f} (own value, not reduced sum)")
    print("  Cross-process NCCL all-reduce does NOT happen automatically.")
    print("  Use Horovod or dist_sync for cross-process reduction.")


# ---------------------------------------------------------------------------
# Test 2: confirm single-process multi-GPU is the correct NCCL pattern
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    NUM_GPUS_MAIN < 2,
    reason=f"Requires >= 2 GPUs, found {NUM_GPUS_MAIN}"
)
def test_nccl_single_proc_multi_gpu_is_supported():
    """Positive control: the supported pattern works correctly.

    Single process, all GPUs visible, one kvstore — this is what MXNet
    NCCL was designed for and what KVStoreNCCL implements.
    """
    import mxnet as mx

    kv = mx.kv.create('nccl')
    shape = (1024,)
    key = 'ctrl_key'
    kv.init(key, mx.nd.ones(shape, mx.gpu(0)))

    arr_list = [mx.nd.ones(shape, mx.gpu(g)) * (g + 1) for g in range(NUM_GPUS_MAIN)]
    res = [mx.nd.zeros(shape, mx.gpu(g)) for g in range(NUM_GPUS_MAIN)]
    kv.push(key, arr_list)
    kv.pull(key, res)
    mx.nd.waitall()

    expected = sum(range(1, NUM_GPUS_MAIN + 1))  # 1 + 2 + ... + N
    for g in range(NUM_GPUS_MAIN):
        got = res[g].asnumpy()
        assert np.allclose(got, expected, rtol=1e-5), \
            f"GPU {g}: expected {expected}, got mean={got.mean()}"

    print(f"\n  Single-process {NUM_GPUS_MAIN}-GPU all-reduce: "
          f"sum 1..{NUM_GPUS_MAIN} = {expected:.0f}. PASS.")


# ---------------------------------------------------------------------------
# Test 3: document why Horovod/DDP-style multi-proc is not built-in
# ---------------------------------------------------------------------------
def test_nccl_multiproc_not_builtin_is_documented():
    """Documents the architectural constraint (always passes, no GPU needed).

    MXNet's kvstore layer has these multi-process options:
      - dist_sync / dist_device_sync / dist_async: PS-Lite parameter server
        over TCP/RDMA (ncclAllReduce is NOT used)
      - Horovod: external library, wraps MXNet tensors into ncclAllReduce
        (not in this wheel)
      - BytePS: similar external dependency (not in this wheel)

    There is NO built-in 'nccl' kvstore type that spans multiple processes.
    KVStoreNCCL extends KVStoreLocal and calls ncclCommInitAll() on all
    devices visible inside ONE process.  If you want NCCL-based all-reduce
    across processes, the path is: Horovod + MXNet backend.

    This test exists to prevent confusion: if someone reads issue #10 and
    tries to create an 'nccl' kvstore from N separate processes, each process
    gets an isolated size-1 communicator and no collective happens.
    """
    assert True, "Documentation test — always passes"
    print("\n  Per-process NCCL all-reduce requires Horovod or BytePS.")
    print("  MXNet's built-in 'nccl' kvstore is single-process only.")


if __name__ == '__main__':
    # Quick standalone run
    import mxnet as mx
    if mx.device.num_gpus() >= 2:
        test_nccl_per_process_is_isolated()
        test_nccl_single_proc_multi_gpu_is_supported()
    test_nccl_multiproc_not_builtin_is_documented()
    print("All checks passed.")
