# Bug audit — 2026-06 (data races, leaks, UAF, error handling)

Findings from a parallel deep-dive. Status: `confirmed` = substantiated by reading
code; `needs-verify` = plausible mechanism, not yet proven triggerable; `fixed` =
addressed. Only act on items I can prove; verify before fixing.

## Tier 1 — confirmed, contained, clearly-correct fix

### B1. CUB global-reduce fast path ignores caller-supplied `workspace`  [confirmed]
`src/operator/tensor/broadcast_reduce_op.cc` (CUB branch in `ReduceAxesRTCComputeImpl`)
+ `src/operator/tensor/reduce_cub.cu` (`CubGlobalSumReduce` uses `ctx.requested[0]`).
The CUB branch sits *before* the `if (workspace == nullptr)` check and allocates its
scratch from `ctx.requested[0]` instead of the supplied `workspace`. `GetSpace`
returns the same base pointer when the existing buffer is big enough, so for callers
that carve both the reduction *input* and the workspace from `ctx.requested[0]`
(e.g. GPU `_npi_kron` backward with a scalar operand; conv/deconv bias-grad with
num_filter==1), CUB clobbers the input → wrong result/gradient, silently.
**RESOLUTION: NOT A REAL BUG — fix reverted.** The aliasing is real (CUB ignores
the `workspace` param and re-requests `ctx.requested[0]`), but it is **benign**:
np.var/std/mean axis=None are numerically correct on fp16/fp32/fp64 (CUB's pass-1
writes each temp slot only after its block has read the overlapping input tile, so
no corruption manifests). I tried the obvious "fall back to RTC when a workspace is
supplied" guard, but that **regressed fp16 global var/std to `inf`**: the CUB path
accumulates in double, while the RTC path the guard diverts to overflows fp16 when
summing ~1M squared deviations. Caught by the new test
`test_linalg_reduce_safety_gpu.py::test_global_var_std_mean_gpu[*-float16]`. The
CUB path is both correct and more accurate, so the guard was reverted. Left as-is.

### B2. Async use-after-free in numpy linalg cusolver wrappers  [confirmed]
Device buffer Alloc'd → async cusolver call (returns immediately) → CPU `Free` with
NO stream sync. Same class `LinalgEphemeralGPUStorage` was created to fix, but not
applied here:
- `src/operator/numpy/linalg/np_qr-inl.h` GEQRF (~167-177), ORGQR (~197-208)
- `src/operator/numpy/linalg/np_solve-inl.h` GETRF (~132-144, frees `workspace`+`info`), GETRS (~157-168)
**Fix:** use `EPHEMERAL_GPU_STORAGE_ALLOC(...)` + drop the manual `Free`, mirroring
`linalg_impl.h`'s `linalg_gelqf`. (workspace-query Alloc/Free are synchronous size
queries — leave those.)

### B3. cuBLAS handle math mode not restored if the GEMM fallback throws  [confirmed]
`src/operator/linalg_impl.h`: `SetCublasMathMode(handle, X)` ... `CUBLAS_CALL(gemm)`
... `cublasSetMathMode(handle, saved)`. `CUBLAS_CALL` throws under LOG_FATAL_THROW;
the threaded engine catches and continues, so the long-lived per-stream handle is
left in TF32/tensor-op mode → every later GEMM on that stream runs in the wrong math
mode (silent numerical corruption). Sites: fp32 368/402, fp16 525/623, fp64-batch
753/796, fp32-batch 821/901, macros 480/499, 714/733, 922/943.
**Fix:** RAII guard that restores math mode in a non-throwing dtor.

### B4. Failed cuBLASLt workspace cudaMalloc leaves a sticky CUDA error  [confirmed]
`src/common/cuda/cublaslt_gemm.cc` `LtPool::AllocWorkspace` (~124-132): on `cudaMalloc`
failure returns nullptr but never calls `cudaGetLastError()`, so the sticky
`cudaErrorMemoryAllocation` is later observed by the next kernel's post-launch check
and misattributed (crash with wrong message). Codebase clears such errors elsewhere
(kvstore/comm.h:57, pooled_storage_manager). Only with MXNET_USE_CUBLASLT=1.
**Fix:** `cudaGetLastError()` after the failed malloc.

### B5. `DestroySolverHandle` never resets ownership flag  [confirmed, latent]
`3rdparty/mshadow/mshadow/stream_gpu-inl.h:157-164`: unlike Destroy{Blas,Dnn}Handle,
doesn't set `solver_handle_ownership_ = NoHandle` → latent double-free if a
destroy-without-recreate path is ever added. **Fix:** add the one line.

## Tier 2 — note, higher risk or pre-existing; defer / careful

### T1. CUDA event-pool slot recycling under-synchronizes  [FIXED]
Confirmed real (async/event-bus engine, MXNET_ENGINE_TYPE=...Async). The pool of
reusable CUDA events hands out a `weak_ptr` + monotonic `pool_index`; the pool owns
the `shared_ptr` for its lifetime so the `weak_ptr` NEVER expires, making the
`expired()`-based recycling guard in OnStart{CPU,GPU} dead code. Once a slot is
"lapped" (>= pool_size later events issued; pool was 64, shared per device), its
cached `cudaEvent_t` reflects a newer, unrelated record -- possibly on a DIFFERENT
stream. A consumer still holding the old `pool_index` then `cudaStreamWaitEvent`s on
that reused event and waits for the wrong stream's work, not the op it depends on ->
cross-stream UNDER-synchronization -> stale reads / wrong results (the producer op
can still be running, since OnCompleteGPU records the event without waiting).

Fix (engine.h + threaded_engine.cc):
- `CUDAEventPool::IsLapped(pool_index)` reports when a slot has been re-issued;
  `EventInfo` now carries the issuing pool so consumers can check it (handles the
  cross-device case too). The two dependency-wait loops (OnStartCPU /
  OnStartGPU) fall back to a host `cudaStreamSynchronize` of the recorded stream
  when the slot is lapped (correct -- it drains the depended-on op -- just
  coarser), instead of a device wait on the reused event.
- Pool size is now `MXNET_CUDA_EVENT_POOL_SIZE` (default 1024, was a hardcoded 64)
  so lapping -- and the serializing fallback -- is rare under real workloads.
The common (not-lapped) path is byte-for-byte the old behavior; default
(non-async) engine is unaffected. Validated: op suites correct under the async
engine with default AND forced-tiny (=2/4) pools that lap every op; a 30x60-op
dependency-chain stress matches a float64 reference; new regression test
tests/python/gpu/test_event_pool_recycling_gpu.py. (The async engine + CUDA
graphs combo still aborts on capture -- a separate, pre-existing issue documented
in CUDA_GRAPHS_PLAN.md, untouched by this fix.)

### T2. `Stream<gpu>` ctor leaves `prop`/`dev_id` uninitialized  [FIXED]
`stream_gpu-inl.h` ctor; `prop` read in `batched_gemm`. Always set via engine
(`dev_id>=0`) today, so latent. Fixed: ctor now zero-inits `prop()` and
`dev_id(-1)`.

### T3. Unchecked `cudaMemsetAsync`/`cudaMemcpyAsync` returns  [pre-existing, broad]
Many sites discard the `cudaError_t` (transformer.cu, softmax.cu, several
numpy/tensor -inl.h). A failure gets misattributed to a later op. Low-med; broad
churn — selective wrapping only.

### T5. np.mean(fp16) over a large axis overflows to inf  [FIXED]
Confirmed real and NumPy-divergent: `np.mean(fp16, axis)` over a large reduced
extent returned `inf` where NumPy returns a finite ~1.0. Root cause: the RTC
reduce accumulates in fp32 (AccType<half>=float) but writes the *un-normalized*
sum to the fp16 output (`OType::to(val)`) BEFORE the mean division, so the sum
(e.g. 200000) overflows fp16's ~65504. (np.sum(fp16) also returns inf, but that
MATCHES NumPy, so it is left alone. bf16 shares fp32's exponent range and does not
overflow.) Fix (broadcast_reduce_op.cc): for normalize + fp16 output + the direct
op path (workspace==nullptr), reduce the sum into an fp32 scratch then
divide-and-cast to fp16, keeping the sum wide until after division. Verified:
fp16 mean large-axis now finite & correct; fp16 sum unchanged (inf, =NumPy);
small fp16 / fp32 / fp64 mean unaffected. Test added.
NOTE remaining: np.var/std(fp16) over a large axis (the moments path, which passes
its own workspace) can still overflow the same way — not covered by this scoped
fix; would need the moments path to compute in fp32. Logged for follow-up.

### T4. Throwing destructors under LOG_FATAL_THROW  [FIXED]
Confirmed real: 16 destructors called throwing macros (CUDA_CALL / CUDNN_CALL /
CUDA_DRIVER_CALL / NVRTC_CALL). Since C++11 destructors are implicitly
noexcept(true), a throw from these calls std::terminate *immediately* (not only
during unwinding). Realistic triggers: `cudaEventSynchronize` in `~CUDAEvent`
surfacing a latched async error from a prior kernel (terminate masks the real
error), and destroy calls failing during CUDA/driver shutdown.

Fix: added non-throwing macros CUDA_CALL_NONFATAL / CUDA_DRIVER_CALL_NONFATAL
(log WARNING instead of CHECK; tolerate cudaErrorCudartUnloading /
CUDA_ERROR_DEINITIALIZED) -- mirroring the pre-existing CUDNN_CALL_NONFATAL.
Converted all 16 destructors:
- core: CUDAEvent (cudaEventSynchronize+Destroy), DeviceStore (cudaSetDevice),
  CudaModule::Chunk (cuModuleUnload + nvrtcDestroyProgram).
- 13 cuDNN op dtors (conv/deconv/pool/activation/softmax-act/lrn/bn/bilinear/
  spatial-transformer/dropout/rnn/quantized conv+pool): CUDNN_CALL ->
  CUDNN_CALL_NONFATAL (56 calls), scoped to the brace-matched dtor bodies.
A failed cleanup now logs a warning and leaks at most a small handle, instead of
crashing the process. Correctness of normal execution is unchanged.

## Verified CORRECT (no action) — checked and cleared
- `FlatOuterAxisSum`/`FlatGlobalSum`/`TryFastCpuFloatSum` (this branch) — no race, correct.
- bf16 `FloatToBF16` RNE + RTC `__float2bfloat16` — correct.
- large-N transpose guard + mean dtype FFI + take dedup + result_type rewrite — correct.
- cuBLASLt descriptor/workspace freeing per call (no leak); pooled_storage_manager
  OOM retry + accounting; reduce_rtc workspace; RTC kernel cache locking; LtPool
  cache locking; LinalgEphemeralGPUStorage RAII.
