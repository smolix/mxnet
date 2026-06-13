# Handoff: intermittent cold-start deadlock in `mx.np.where` / `nonzero` (GPU)

**Status:** MITIGATION APPLIED — pending validation under load (2026-06-13).
Engine changes landed on branch `fix/nonzero-coldstart-deadlock-engine`. Built
with CUDA on an RTX 3060 and passed **100/100 cold starts under self-induced
GPU+CPU load, 0 hangs** (+40/40 idle), no `MXNET_ENGINE_DIAG` timeouts. NOT yet
definitive: that box never reproduced the original hang when idle (0/540 in the
original campaign), so a clean run there proves no-regression + correct-path, not
that the rare race is gone. **Re-verify on the heavily-loaded box** (see
"Verification on the server" below).

The original OPEN analysis is preserved unchanged below the resolution.

---

## RESOLUTION (2026-06-13)

A native-debugger reproduction was *not* required: a close code review located the
real defects. The handoff's prime suspect — `dmlc::ManualEvent` as a lost-wakeup
source — was **exonerated** (the waiter holds `mutex_` from its `!signaled_` check
through entering `condition_variable_.wait`, and `signal()` must take `mutex_` to
notify, so no notify can slip into the gap). `WaitForVar`/`OnComplete` are also
correct. Two genuine problems were fixed instead:

**Fix 1 — don't hold `create_mutex_` across CUDA stream init** (the §3 fragility).
`LazyAllocArray::Get()` holds `create_mutex_` across the whole `creator()`, and for
GPU pools `creator()` ran `ThreadPool(..., wait=true)` → `WaitForReady()`, which
blocks until each worker has done `cudaStreamCreate` / `new GPUAuxStream`. So the
pusher held a lock across CUDA driver calls of unbounded duration — exactly matching
the "only under heavy multi-process contention, only on the first data-dependent op"
signature. Change (`src/engine/thread_pool.h`, `src/engine/threaded_engine_perdevice.cc`):
- `ThreadPool::WaitForReady()` made public.
- GPU worker pools constructed with `wait=false`, so `Get()` returns (releasing
  `create_mutex_`) before any stream is created.
- The readiness wait moved *outside* the lock via a new `EnsureWorkersReady()`,
  gated by a per-block `std::atomic<bool> workers_ready` (one relaxed load per push
  after warmup → no hot-path cost). Same ordering invariant as before (stream is
  registered before the op is pushed), but a slow/contended stream init can no
  longer serialize unrelated pool creations or completion callbacks behind the lock.

**Fix 2 — never drop a counted op** (the §4 bug). Every GPU/CPU `PushToExecute`
site did `auto ptr = ...Get(...); if (ptr) { ...Push... }` and **silently dropped
the op when `Get()` returned `nullptr`** (i.e. `is_clearing_` during shutdown). The
op was already counted in `pending_` (`ThreadedEngine::Push`), so a drop wedges
`pending_` and hangs `WaitForVar`/`WaitForAll` forever — the exact reported symptom
(`pending_ops>=1`, unkillable, holds the GIL), though gated on engine teardown.
Added `FinishUnschedulableOpr()`: completes the op inline (runs the completion
callback, skips the body) so `pending_` is decremented and waiters wake.

### Verification on the server (the definitive step)
1. Build the branch; un-skip `test_np_more_array_like_wrappers` in
   `tests/python/unittest/test_numpy_op.py`.
2. Recreate the original contention (real training load is closest) and loop fresh
   cold-start processes with `MXNET_ENGINE_DIAG=1 MXNET_ENGINE_DIAG_TIMEOUT_S=10`.
   A ready harness mirror is at `/tmp/repro/` on the dev box (`coldstart_probe.py`,
   `stress.sh N LOAD HARD_TIMEOUT`).
3. If a hang still occurs: capture the `[MXNET_ENGINE_DIAG] ... pending_ops=` line,
   then `gdb -p <pid> -batch -ex "thread apply all bt"`. If `pending_ops>=1` with a
   GPU worker stuck inside a `cuda*` call, it's the pure driver stall (hyp 1, not
   engine-fixable) and the remaining mitigation is to warm a GPU stream before the
   first data-dependent op, or retry. If `pending_ops==0`, it's a completion/wakeup
   bug (revisit `WaitForVar`).

### Build notes (CUDA-toolkit gotcha found while building)
- A real blocker: `nvcc` 13.2 with `/usr/local/cuda`→13.3 headers on the include
  path → CCCL "compiler and toolkit headers are incompatible". Align them:
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.3`.
- Dev box lacked system OpenBLAS/gfortran/cuDNN; used a throwaway conda prefix for
  OpenBLAS+gfortran and built `USE_CUDNN=OFF USE_ONEDNN=OFF USE_OPENCV=OFF
  MXNET_CUDA_ARCH=8.6`. Runtime needed `LD_PRELOAD` of a libstdc++ ≥ GLIBCXX_3.4.33
  because the conda python ships an older one. (cuDNN-off is fine for this repro —
  the change is in worker-pool creation and `nonzero` doesn't use cuDNN.)

---

## ORIGINAL ANALYSIS (status was: OPEN)

Could not be reproduced on the dev box once the triggering external load
disappeared, and `ptrace` was blocked there so no native debugger was available.

**Owner action requested:** reproduce under load + attach a debugger (gdb /
cuda-gdb) to a hung process, get the native (C++) backtrace of *all* threads,
and confirm which of the hypotheses below is correct, then fix at the source.

---

## 1. Symptom

Running the GPU test suite against the wheel, `test_numpy_op.py` intermittently
**hangs forever** (no progress, GPUs go idle) inside:

```
tests/python/unittest/test_numpy_op.py::test_np_more_array_like_wrappers
```

`pytest-timeout` fires its faulthandler dump but **cannot kill the process** —
the hang is a C-level deadlock that holds the GIL, so the SIGALRM/thread timeout
handlers never run. Only an OS-level `kill -9` (or `timeout -s KILL`) stops it.

Faulthandler Python stack at hang (only Python-stack threads are shown;
the C++ engine worker threads have no PyThreadState so faulthandler does NOT
list them — do not conclude "only 2 threads exist"):

```
Thread <main>:
  File ".../mxnet/ndarray/numpy/_op.py", line 8558 in nonzero      # out = _api_internal.nonzero(a).transpose()
  File ".../mxnet/ndarray/numpy/_op.py", line 9778 in where
  File ".../mxnet/numpy/multiarray.py", line 12422 in where
  File ".../test_numpy_op.py", line 2964 in test_np_more_array_like_wrappers   # actual = np.where([[0, 1], [2, 0]])
```

The hanging call is the **single-argument** `np.where(cond)`, which is
`nonzero(cond)`. `test_np_more_array_like_wrappers` (test_numpy_op.py:2942)
calls, in order: `np.where([[0,1],[2,0]])` (line 2964), `np.nonzero([0,2,0,3])`,
`np.unique(...)` — all **data-dependent ops** (output shape known only at
runtime), so the main thread must **block** until the op completes.

## 2. Minimal reproducer (op-level, no pytest)

```python
import mxnet as mx
from mxnet import np, npx
npx.set_np()
a = np.where([[0, 1], [2, 0]])      # single-arg where -> nonzero, on default device = GPU
for e in a: e.asnumpy()             # forces the blocking wait
```

Run as a **fresh process** (cold start) on a GPU. Observed hang rate:
- **~1 in 8 fresh processes** while an unrelated heavy job saturated GPU0 + CPU.
- **0 in ~540 fresh processes** once the box went idle (see §5).

So it is a **cold-start + contention** race: it needs scheduling/GPU pressure to
open the timing window, and it is on the **first** GPU data-dependent op of the
process (a warm process ran the same op **45,000+** times with no hang).

A ready-made probe + harnesses are checked in under `.tmp/` (see §7).

## 3. What the op does (why a data-dependent op blocks the caller)

`nonzero` has **no `FInferShape`** — it sets its output shape at runtime inside
the op via `out.Init(s)`. So the FFI returns an output whose shape is only known
after the op runs; the main thread blocks (`WaitToRead` -> `WaitForVar`) until
the engine completes it.

Code:
- GPU compute: `src/operator/numpy/np_nonzero_op.cu:42` `NonzeroForwardGPU`.
  Does `cub::DeviceScan::InclusiveSum` -> `cudaMemcpyAsync(D2H count)` ->
  **`cudaStreamSynchronize`** (line 86) -> `out.Init(s)` (allocates output).
  Registered as `FComputeEx<gpu>`, `FResourceRequest = kTempSpace`.
- CPU compute: `src/operator/numpy/np_nonzero_op.cc:56`.
- FFI: `src/api/operator/numpy/np_nonzero_op.cc:30` -> `Invoke(...)`.
- Python: `python/mxnet/ndarray/numpy/_op.py:8482` (`nonzero`),
  `:8558` `out = _api_internal.nonzero(a).transpose()`; `where` at `:9778`.

`np.where([[0,1],[2,0]])` first converts the Python list to a GPU NDArray
(an H2D **copy** op -> `gpu_copy_workers_`), then runs `nonzero` ->
`gpu_normal_workers_`. So cold start lazily creates **multiple** worker pools.

## 4. Engine facts established (default config)

- Default engine is **`ThreadedEnginePerDevice`**, and **`IsEngineAsync()` is
  false by default** (`src/engine/threaded_engine.cc:647`): it only returns true
  when `MXNET_ENGINE_TYPE` *ends with* "Async". So the **new GPU dependency /
  CUDA-event-pool path is OFF** (`OnStartGPU`/`OnStartCPU` early-return; the
  default `OnCompleteGPU` just does `worker_stream->Wait()` then
  `OnCompleteStatic`). **Do not chase the event-pool / IsLapped / TOCTOU code —
  it does not run by default.** (Confirm on the repro: `echo $MXNET_ENGINE_TYPE`
  should be empty.)
- GPU worker pools are created **lazily** on first use:
  `src/engine/threaded_engine_perdevice.cc:213` (`gpu_normal_workers_.Get`),
  `:173` (copy workers). Creation runs `ThreadPool(size, fn, wait=true)` which
  **blocks the pusher** in `WaitForReady()` until each worker signals its
  `ManualEvent` *after* allocating its CUDA stream
  (`GPUWorker`, perdevice.cc:285; stream alloc at :298-:304; ready signalled by
  `SetReadyOnDestroy` at end of the `do{...}while(false)` block).
- `LazyAllocArray::Get` holds `create_mutex_` across the **entire** worker-pool
  creation (`src/common/lazy_alloc_array.h:96`).
- Main thread blocks in `ThreadedEngine::WaitForVar`
  (`src/engine/threaded_engine.cc:434`): it pushes a "WaitForVar" CPU async op
  depending on the output var, then waits on `finished_cv_` until `done`.

### Primitives I reviewed and believe are individually CORRECT
(so the bug is most likely a cold-start *interaction*, not one broken object):
- `dmlc::ConcurrentBlockingQueue::Push/Pop`
  (`3rdparty/dmlc-core/include/dmlc/concurrency.h:160-234`): notify computed
  under lock, predicate re-checked -> no lost wakeup.
- `ThreadedEngine::WaitForVar` CV handshake (threaded_engine.cc:434-498):
  `done` set under `finished_m_`, predicate re-checked -> no lost wakeup.
- `ThreadedEngine::OnComplete` (threaded_engine.cc:542-600).
- `dmlc::ManualEvent::wait/signal`
  (`3rdparty/dmlc-core/include/dmlc/thread_group.h:34-73`): ugly unlocked atomic
  store in `signal()`, but lock discipline still closes the window. NOTE: `wait()`
  uses a **plain `cv.wait(lock)` with no predicate** — only vulnerable to
  *spurious wakeup* (early return), which would NOT cause a hang. Still, this is
  the weakest-looking spot; worth re-checking under a debugger.

## 5. Reproduction attempts (all on the dev box, after the external job ended)

| attempt | conditions | result |
|---|---|---|
| 8 isolated fresh procs | external 25h job hammering GPU0+CPU | **1 hang** (1/8) |
| 54 single fresh procs | idle box | 0 |
| matrix 6 cfgs x 40 (default/naive/1-thread/warmup/nonzero/sum-control) | idle/mixed | **0/240** |
| 113 fresh procs | synthetic load: matmul loops on GPU0/2/3 + CPU spinners on all cores | 0 |
| 60 high-concurrency burst (12 cold starts at once) | idle | 0 |
| 167 fresh procs | under the live full test sweep (3-shard real mxnet load) | 0 |
| (running at handoff) | 3 resnet18 training loops + probe | TBD |

**Conclusion:** the race is real but **timing/contention-sensitive** and did not
reproduce once the specific external workload disappeared. The single confirmed
reproduction (1/8) coincided with that external job. The matrix is inconclusive
*because nothing hung* — it does NOT clear naive vs threaded.

## 6. The ONE diagnostic that will crack it: `MXNET_ENGINE_DIAG`

`ThreadedEngine::WaitForVar`/`WaitForAll` have a built-in watchdog:

```
MXNET_ENGINE_DIAG=1 MXNET_ENGINE_DIAG_TIMEOUT_S=10 <run>
```

On a hang it logs (threaded_engine.cc:484):
```
[MXNET_ENGINE_DIAG] WaitForVar timeout after Ns: var=0x... pending_ops=N shutdown_phase=.. kill=0
```

**`pending_ops` is the fork:**
- `pending_ops > 0` -> an op is **stuck unexecuted / mid-flight**: the worker
  never popped it, or it's stuck in `cudaStreamSynchronize` (np_nonzero_op.cu:86)
  / a CUDA driver stall under contention, or the pusher is stuck creating the
  worker pool (`WaitForReady`). Then attach the debugger and look at the GPU
  worker thread(s) and the pusher.
- `pending_ops == 0` -> all ops completed but `done`/`finished_cv_` wasn't
  observed -> a wakeup/ordering bug in `WaitForVar` (less likely per §4).

Capture this first thing once reproduced.

## 7. Ready-made tools (checked in under `.tmp/`, gitignored — copy them over)

- `.tmp/coldstart_probe.py` — single cold-start op; `PROBE_MODE=where|nonzero|sum|warmup`.
- `.tmp/diag_repro.sh` — loop fresh probes with `MXNET_ENGINE_DIAG=1` until one hangs (idle).
- `.tmp/burst_repro.sh` — 12 concurrent cold starts/round (max contention).
- `.tmp/load_and_probe.sh` — CPU spinners + GPU matmul load + probe.
- `.tmp/realwl_repro.sh` + `.tmp/real_workload.py` — resnet18 training loops as load + probe (closest mimic of the original job; **try this first** on the debugger box).
- `.tmp/catch_nonzero_hang.sh` / `.tmp/proc_catch.sh` — catch a hang and dump
  gdb / `/proc/<pid>/task/*/stack` + `wchan`. **gdb attach is blocked here**
  (`ptrace: Operation not permitted`); on a normal box, `gdb -p <pid> -batch -ex
  "thread apply all bt"` will work and is the key step.

### Debugger recipe (the actual next step)
1. On the slower single-GPU box, run a heavy background load (real training is
   best) to recreate contention.
2. Loop the minimal reproducer (§2) as fresh processes with `MXNET_ENGINE_DIAG=1`.
3. When one hangs: record the `[MXNET_ENGINE_DIAG] ... pending_ops=` line, then
   `gdb -p <pid> -batch -ex "set pagination off" -ex "thread apply all bt"` and,
   for CUDA, `cuda-gdb` likewise. Capture **all** threads (the C++ engine workers
   are the interesting ones — invisible to faulthandler).
4. Identify: is the GPU worker stuck in `cudaStreamSynchronize`
   (np_nonzero_op.cu:86) / a CUDA driver call (=> possibly a driver-level stall
   under contention, not an MXNet bug), or is it blocked on an MXNet mutex/CV
   (=> our bug, fix in `threaded_engine*`), or is the **main thread** stuck in
   `ThreadPool::WaitForReady` (perdevice worker creation) waiting on a
   `ManualEvent` that was never signalled?

## 8. Hypotheses, ranked
1. **CUDA driver-level stall** of `cudaStreamSynchronize` on a just-created
   stream under heavy multi-process contention (cold start). Would show
   `pending_ops>=1` with the GPU worker in a CUDA call. If so it's not
   MXNet-fixable in the engine; mitigations: warm the stream/engine before first
   data-dependent op, or retry.
2. **Lazy worker-pool creation race**: pusher holds `create_mutex_` in
   `WaitForReady` while the worker does CUDA init under contention; some
   `ManualEvent`/stream-init interaction stalls. Look at perdevice.cc:285-320 +
   thread_pool.h WaitForReady + ManualEvent.
3. **Lost/ë mis-ordered completion** in `WaitForVar` (least likely; would show
   `pending_ops==0`).

A quick empirical disambiguator (no debugger): re-run the §5 matrix **while the
box is under real load** and compare `MXNET_ENGINE_TYPE=NaiveEngine` (synchronous,
no worker threads) vs default. If NaiveEngine never hangs but default does, it's
the threaded engine (hyp 2/3). If NaiveEngine ALSO hangs, it's the op / CUDA
(hyp 1). I could not run this cleanly because nothing hung once idle.

## 9. Quarantine in this PR
`test_np_more_array_like_wrappers` is marked skipped (see the skip reason in
`tests/python/unittest/test_numpy_op.py`) so the suite does not hang in CI.
**Remove that skip to reproduce / verify the fix.** Everything else in the
campaign is fixed and green (see PR description).

## 10. Not done on purpose
No release wheel was built/tagged for this (per instruction). The cython wheel
*does* build & pass everything except this flaky test; rebuild + tag only after
the fix.
