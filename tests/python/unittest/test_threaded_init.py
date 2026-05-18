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

"""
Smoke test for apache/mxnet#17495 — Singleton thread-safety.

Exercises simultaneous import and basic usage of MXNet singletons from N
threads.  Before the C++17 audit / Profiler::Get() fix, naive DCL on a
std::shared_ptr was a data race.  With C++17 magic-statics for
Engine::Get(), Storage::Get(), OpenMP::Get(), CpuEngine::Get(), and the
fixed Profiler::Get(), all singletons should initialise safely.
"""
import threading
import traceback

import pytest


N_THREADS = 8
TIMEOUT   = 30  # seconds


def _worker(results, idx):
    try:
        import mxnet as mx  # noqa: PLC0415
        # Touch each major singleton used from Python land.
        ctx = mx.cpu()
        a   = mx.nd.array([1.0, 2.0, 3.0], ctx=ctx)
        b   = mx.nd.array([4.0, 5.0, 6.0], ctx=ctx)
        c   = (a + b).asnumpy()
        assert list(c) == [5.0, 7.0, 9.0], f"Thread {idx}: unexpected result {c}"
        results[idx] = "ok"
    except Exception:
        results[idx] = traceback.format_exc()


def test_threaded_import_and_alloc():
    """All N threads must complete without error."""
    results = [None] * N_THREADS
    threads = [
        threading.Thread(target=_worker, args=(results, i), daemon=True)
        for i in range(N_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=TIMEOUT)

    failures = []
    for i, r in enumerate(results):
        if r is None:
            failures.append(f"Thread {i}: TIMEOUT (did not complete in {TIMEOUT}s)")
        elif r != "ok":
            failures.append(f"Thread {i}: EXCEPTION\n{r}")

    assert not failures, "\n".join(failures)


def test_profiler_get_is_stable_across_threads():
    """Profiler::Get() must return the same singleton from every thread."""
    import mxnet as mx  # noqa: PLC0415
    from mxnet import profiler  # noqa: PLC0415

    ids   = [None] * N_THREADS
    ready = threading.Barrier(N_THREADS)

    def _probe(idx):
        ready.wait()          # start all threads at the same instant
        ids[idx] = id(profiler)

    threads = [threading.Thread(target=_probe, args=(i,), daemon=True) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=TIMEOUT)

    assert all(x == ids[0] for x in ids), f"Profiler module ids differ: {ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
