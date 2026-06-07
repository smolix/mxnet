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

### T1. CUDA event-pool `weak_ptr` never expires  [needs-verify, risky]
`include/mxnet/engine.h` `CUDAEvent` keeps the owning `shared_ptr` for pool lifetime;
only `weak_ptr` is handed out, so the expiry-based recycling in `OnStart*` is dead
code and events recycle by 64-slot counter wraparound (shared per device). Possible
cross-stream ordering hole under >64 outstanding deps with the async engine
(non-default). Needs a runtime stress check to prove wraparound overtakes an in-flight
dependency before changing — fixing blindly risks regressions. Defer.

### T2. `Stream<gpu>` ctor leaves `prop`/`dev_id` uninitialized  [needs-verify, latent]
`stream_gpu-inl.h` ctor; `prop` read in `batched_gemm`. Always set via engine
(`dev_id>=0`) today, so latent. Cheap safe fix: init `dev_id(-1)` + zero `prop`.

### T3. Unchecked `cudaMemsetAsync`/`cudaMemcpyAsync` returns  [pre-existing, broad]
Many sites discard the `cudaError_t` (transformer.cu, softmax.cu, several
numpy/tensor -inl.h). A failure gets misattributed to a later op. Low-med; broad
churn — selective wrapping only.

### T5. RTC fp16 reduce overflows for large sums  [needs-verify, pre-existing]
Surfaced while reverting B1: routing fp16 global var/std through the RTC reduce gave
`inf` (sum of ~1M squared deviations overflows fp16's ~65504 max). The CUB path
accumulates in double and is fine, and masks this for scalar-output sum/mean. But
fp16 large-*axis* reductions (non-scalar) always use RTC — if RTC accumulates in
fp16 rather than a wider AccType there, `np.sum`/`np.mean`/`np.var` over a large
fp16 axis would overflow. Verify whether the RTC reduce honors safe-acc
(MXNET_SAFE_ACCUMULATION) for fp16 by default; if not, large fp16 axis reductions
are silently wrong. Separate from B1; not chased here.

### T4. Throwing destructors under LOG_FATAL_THROW  [pre-existing, systemic]
`CUDAEvent::~CUDAEvent`, `DeviceStore::~DeviceStore`, several cuDNN op dtors use
`CUDA_CALL`/`CUDNN_CALL` (throw). If run during unwinding → std::terminate. Systemic;
risky to change broadly. Defer.

## Verified CORRECT (no action) — checked and cleared
- `FlatOuterAxisSum`/`FlatGlobalSum`/`TryFastCpuFloatSum` (this branch) — no race, correct.
- bf16 `FloatToBF16` RNE + RTC `__float2bfloat16` — correct.
- large-N transpose guard + mean dtype FFI + take dedup + result_type rewrite — correct.
- cuBLASLt descriptor/workspace freeing per call (no leak); pooled_storage_manager
  OOM retry + accounting; reduce_rtc workspace; RTC kernel cache locking; LtPool
  cache locking; LinalgEphemeralGPUStorage RAII.
