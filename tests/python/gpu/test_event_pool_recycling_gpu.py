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

"""Regression test for the async-engine CUDA event-pool slot recycling fix.

The event-bus dependency engine (MXNET_ENGINE_TYPE=...Async) tracks cross-op,
cross-stream ordering with a round-robin pool of reusable CUDA events. The
weak_ptr handed out for a pooled event never expires (the pool owns the event for
its lifetime), so when a slot is "lapped" (reused for a newer, unrelated record
after pool_size more events) a consumer that still references the old pool_index
would wait on the wrong record -> potential cross-stream under-synchronization ->
wrong results. The fix detects the lapped slot (CUDAEventPool::IsLapped) and falls
back to a host sync of the recorded stream.

This test forces constant lapping by shrinking the pool to 2 events and running a
deep dependency chain under the async engine in a subprocess (engine type and pool
size are read once at init, so they must be set before mxnet is imported), then
checks the result against a float64 reference.
"""

import os
import subprocess
import sys
import textwrap

import pytest


_WORKLOAD = textwrap.dedent(
    """
    import numpy as onp, mxnet as mx
    from mxnet import np, npx
    npx.set_np()
    ctx = mx.gpu(0)
    onp.random.seed(0)
    bad = 0
    for _ in range(30):
        n = 4096
        a_np = onp.random.uniform(-1, 1, (n,)).astype('float64')
        b_np = onp.random.uniform(-1, 1, (n,)).astype('float64')
        a = np.array(a_np, ctx=ctx, dtype='float64')
        b = np.array(b_np, ctx=ctx, dtype='float64')
        acc = a
        ref = a_np.copy()
        for _ in range(60):              # >> pool size 2 -> constant lapping
            acc = acc * 1.0009765625 + b
            acc = acc - (b * 0.5)
            ref = ref * 1.0009765625 + b_np
            ref = ref - (b_np * 0.5)
        got = acc.asnumpy()
        rel = onp.abs(got - ref).max() / (onp.abs(ref).max() + 1e-12)
        if rel > 1e-9 or onp.isnan(got).any():
            bad += 1
    npx.waitall()
    print("MISMATCHES", bad)
    assert bad == 0, f"{bad} mismatches -> event-pool recycling under-synchronized"
    print("OK")
    """
)


def _have_gpu():
    try:
        import mxnet as mx
        a = mx.nd.ones((1,), ctx=mx.gpu(0))
        a.wait_to_read()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_gpu(), reason="no usable GPU")
@pytest.mark.parametrize('pool_size', ['2', '4'])
def test_async_engine_event_pool_recycling(pool_size):
    env = dict(os.environ)
    env['MXNET_ENGINE_TYPE'] = 'ThreadedEnginePerDeviceAsync'
    env['MXNET_CUDA_EVENT_POOL_SIZE'] = pool_size
    env.setdefault('OMP_NUM_THREADS', '8')
    p = subprocess.run([sys.executable, '-c', _WORKLOAD],
                       env=env, capture_output=True, text=True, timeout=300)
    assert 'OK' in p.stdout, "stdout:\n{}\nstderr:\n{}".format(p.stdout, p.stderr[-2000:])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
