# Engine Deadlock Audit (2026-05-18)

## Issues addressed: A6 (apache#19994), A7 (apache#18090), A13 (apache#11163)

### Summary

All three issues trace to the same root-cause family: `ThreadedEngine` lifecycle and
shutdown ordering.  Two of the three have been remediated; one is instrumented only.

---

## What already existed (pre-audit)

- `base.py` already registered `_notify_shutdown()` via `atexit`.  This calls
  `MXNotifyShutdown()` → `Engine::Get()->NotifyShutdown()` (sets `shutdown_phase_=true`)
  + `Engine::Get()->WaitForAll()`.
- `ThreadedEngine` dtor: sets `kill_=true`, calls `finished_cv_.notify_all()`.  This
  wakes any `WaitForVar`/`WaitForAll` that is stuck, so stale waiters do not deadlock
  indefinitely after the dtor fires.
- `WaitForVar` already uses `kill_.load()` as a predicate escape condition.

So the "hang forever" scenario from A6/A7 in a process that exits cleanly was largely
mitigated by the existing atexit path.  What was missing:

---

## Gaps found and fixed

### A13 gap: `Stop()` never called before static-dtor

`MXNotifyShutdown` called `NotifyShutdown()` + `WaitForAll()` but never `Stop()`.
`ThreadedEnginePerDevice::Stop()` is the method that:
1. Calls `WaitForAll()` (again, no-op if already done).
2. Calls `SignalQueuesForKill()` — posts sentinel to every worker queue.
3. Joins (`Clear()`s) every thread pool.

Without `Stop()`, the worker threads were still alive when the Python interpreter
started tearing down C extensions.  The static-local `shared_ptr<Engine>` dtor then
fired (as part of `libmxnet.so` static-dtor chain) and called `StopNoWait()` again,
which joined the queues a second time.  On a system under heavy GC / Python extension
cleanup load, the join could race with thread-local storage teardown, causing the
`cv.notify_all()` in `~ThreadedEngine()` to deadlock (the notified threads were already
partially torn down).

**Fix**: added `Engine::Get()->Stop()` to `MXNotifyShutdown()` in `src/c_api/c_api.cc`.
Worker threads are now fully joined *before* Python interpreter teardown begins, making
the subsequent static-dtor `notify_all` a no-op.

File changed: `src/c_api/c_api.cc`

### A6/A7 gap: no observability when WaitForVar stalls

Neither `WaitForVar` nor `WaitForAll` had any progress reporting.  A stuck process would
hang silently with no indication of which variable was waiting or how many ops were
pending.

**Fix**: added `MXNET_ENGINE_DIAG=1` env-var-gated watchdog.  When set:
- `WaitForVar` uses `condition_variable::wait_for` in a loop with a configurable timeout
  (default 30 s, override with `MXNET_ENGINE_DIAG_TIMEOUT_S`).
- On timeout, logs: var pointer, `pending_ops`, `shutdown_phase`, `kill` flag, and a
  hint to switch to `NaiveEngine` for synchronous debugging.
- After logging, the wait continues (does not abort) to avoid masking transient slowdowns.
- Same watchdog added to `WaitForAll`.

File changed: `src/engine/threaded_engine.cc`

---

## Root cause of A6 (long-running inference hang): status

The A6 reporter (`aarch64`) saw `WaitForVar` block after hours of inference.  The
precise trigger was not reproducible locally.  Two plausible mechanisms:

1. **Missing notify edge**: a rare interleaving in `CompleteWriteDependency` between the
   `num_pending_reads_ == 0` check and the `pending_write_` update could theoretically
   leave a successor op with `wait > 0` forever if a concurrent read-completion races
   with a write-completion.  The code uses per-var mutexes so this should be safe, but
   the reviewer did not find a proof of correctness for the combined lock-then-dispatch
   path.  This is the most likely cause; fixing it requires a more invasive rewrite of
   the var-level scheduling logic (out of scope for this pass).

2. **Queue `SignalForKill` before op completes**: if `StopNoWait` fires while a GPU op
   is in-flight (e.g. a CUDA kernel), the op's `OnComplete` callback fires after the
   queue is killed, `pending_` is decremented, `finished_cv_.notify_all()` fires — but
   `WaitForVar`'s `done` flag may never be set if the `WaitForVar` sentinel op was never
   dequeued.  The `kill_` escape in the wait predicate handles this, but only after the
   dtor fires, not during a live stall.

The diagnostic mode (mechanism 2 above) now makes A6-class stalls visible in production
without rebuilding.  A definitive fix for mechanism 1 requires a minimised reproducer.

---

## Files changed

| File | Change |
|------|--------|
| `src/c_api/c_api.cc` | `MXNotifyShutdown`: add `Engine::Get()->Stop()` after `WaitForAll()` |
| `src/engine/threaded_engine.cc` | `WaitForVar`, `WaitForAll`: add `MXNET_ENGINE_DIAG` watchdog |
| `tests/python/unittest/test_engine_shutdown.py` | New: 12 tests for clean exit (A13), many-ops CI-style exit (A7), diag-mode smoke (A6/A7) |

---

## Test results

```
12 passed in 51.00s
```

Tests include:
- `test_clean_exit_basic[0-4]` — 5 trials of basic ndarray compute + exit
- `test_clean_exit_gpu[0-2]` — 3 trials on GPU (graceful skip if no GPU)
- `test_clean_exit_after_many_ops[0-2]` — 3 trials of 200-op CPU workload
- `test_diag_mode_smoke` — verifies `MXNET_ENGINE_DIAG=1` doesn't slow normal path

---

## What would be needed for a complete A6 fix

1. Minimised reproducer for the ARM long-running inference hang (hours of load).
2. Formal proof (or corrective rewrite) of the `CompleteWriteDependency` +
   `AppendReadDependency` atomicity under the `ThreadedVar` mutex.
3. Consider replacing the per-var mutex with a lock-free queue (TBB `concurrent_queue`
   or Folly `MPMCQueue`) to eliminate the race window.
4. Add a `WaitForVar` timeout that actually *cancels* the wait op and returns an error,
   rather than just logging — useful for inference servers that want a circuit-breaker.
