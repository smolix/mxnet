# Performance notes — measured facts

This file was rewritten on 2026-06-06 after the earlier "Blackwell / CUDA 13
port" notes were found to be **stale** — they described an intermediate port
state and quoted numbers measured on a different GPU. The claims below are
measured on the current branch and hardware. The retired claims and why they
were wrong are recorded at the bottom for traceability.

## Test hardware

- **GPU:** NVIDIA GeForce RTX 4090 (Ada, sm_89), 24 GB GDDR6X (~1008 GB/s).
- **Toolchain:** CUDA 13.3, cuDNN 9.23, gcc 13.3, oneDNN v3.11.3 (vendored).
- Build: `USE_CUDA=ON`, `USE_CUDNN=ON`, `USE_ONEDNN=ON`, `USE_LAPACK=ON`,
  `MXNET_CUDA_ARCH=8.9`. (Reproduce the scan with `perf_scan.py`.)

Theoretical peaks used below (RTX 4090): fp16 tensor (fp32 accum) ~165 TFLOP/s,
TF32 ~82.6, fp32 ~82.6, memory bandwidth ~1008 GB/s.

## Measured throughput (this branch, RTX 4090)

| Workload | Measurement | % of peak | Verdict |
|---|---|---|---|
| matmul fp16 4096³ (`nd.dot`) | **162 TFLOP/s** | ~98% fp16 | already optimal |
| matmul fp16 2048³ | 123 TFLOP/s | ~75% | good |
| matmul fp32 4096³ (TF32) | 55 TFLOP/s | ~67% TF32 | acceptable |
| matmul bf16 (FullyConnected) | works (fixed) | — | see note |
| Conv2D fwd (cuDNN) 256ch | ~60 TFLOP/s | — | cuDNN-bound |
| elementwise add (large) | ~800 GB/s effective | ~80% | good |
| **reduction sum / max (any shape)** | **~300 GB/s** | **~30%** | **slow — see below** |
| softmax 4096×4096 | 0.17 ms | — | fine |
| layernorm 4096×4096 | 0.69 ms | — | fine |
| tiny-op dispatch (async) | ~34 µs/op | — | architectural |

## What is NOT a problem on this hardware (retired claims)

The earlier notes listed these as open perf problems. Measurement shows they are
already resolved or never reproduced on Ada:

- **cuBLASLt / tensor-core matmul.** Claimed "fp16 ~50× off peak". Reality: fp16
  matmul is **~98% of peak** via the legacy `cublasGemmEx` path; enabling the
  (already-implemented) `cublaslt_gemm` path (`MXNET_USE_CUBLASLT=1`) gives no
  measurable change on Ada. The ~50× figure was a Blackwell-era measurement.
- **cuDNN 9 RNN.** Claimed GPU RNN `LOG(FATAL)`s on CUDA 13. Reality:
  `rnn-inl.h` already uses `cudnnSetRNNDescriptor_v8`; GPU LSTM/GRU/RNN
  forward+backward all work on CUDA 13 / cuDNN 9.
- **oneDNN old / disabled.** Claimed vendored oneDNN is ~v0.21 and
  `USE_ONEDNN=OFF`. Reality: vendored oneDNN is **v3.11.3** (has AMX),
  `USE_ONEDNN=ON`, actively used (DNNL_VERBOSE shows brgemm/brconv primitives).

## Real, open performance issue

### Reductions are slow on both GPU and CPU (root-caused; needs a kernel project)

Measured `sum`/`mean`/`max` reduction throughput:

| case | GPU (RTX 4090) | CPU (EPYC 7502, 16t) |
|---|---|---|
| trailing-axis reduce (e.g. (4096,4096) axis=1) | ~30% BW | fast (oneDNN `jit:avx`, 0.5 ms) |
| global reduce (→ scalar) | ~30% BW (300 GB/s) | **17 GB/s, single-threaded** |
| outer/strided-axis reduce (axis=0) | ~30% BW | **~3 GB/s** (21 ms) native |

Root causes:
- **GPU** (`reduce_rtc.cc` + `ReduceImplConfig` in `broadcast_reduce-inl.h`,
  `nthread_reduce=512`, `kBaseGridNum=1024`, `maxLoopPerTB=64`): for the
  all-reduce N=1 case `gridDim.x` collapses to 1; parallelism is only in
  `gridDim.y` (~512 blocks) with a long per-block serial loop and a
  shared-memory tree reduction; loads are not vectorized (no float4).
- **CPU global reduce**: the native `seq_reduce_compute` (`broadcast_reduce-inl.h`)
  parallelizes its `#pragma omp parallel for` over the OUTPUT elements N. A
  global reduce has N=1, so it runs on a single thread regardless of
  `OMP_NUM_THREADS` (oneDNN's global reduction is likewise single-threaded).
  Needs a two-stage reduction (split the reduced dim M across threads → partial
  sums → combine) when N is small.
- **CPU outer-axis reduce**: parallel over N but the per-output reduction walks
  memory with a large stride and is not vectorized → ~3 GB/s.

Investigated and rejected (measured): routing non-trailing-axis reductions to
oneDNN. oneDNN v3.11.3's reduction is only optimized for consecutive trailing
dims; outer-axis via oneDNN is *worse* than native ((4096,4096) axis=0:
~99 ms oneDNN vs ~21 ms native), so the existing `SupportDNNLReduceImpl` gate
is correct and was kept.

Status / fixes:
- **CPU global sum/mean: FIXED** (commit "Add fast flat OpenMP path for global
  sum/mean reduction"). A flat `#pragma omp parallel for reduction(+)` with a
  double accumulator (`FlatGlobalSum` + a `SupportDNNLReduceImpl` global
  exclusion) replaced the single-threaded path: float32 17→~48 GB/s (~2.8x,
  and more accurate: rel_err 5e-9 vs 4.7e-6), float64 sum ~91 GB/s (~5x).
  2429 reduction tests pass. (float64 *mean* still takes the native path — a
  perf loose end, not a regression.)
- **CPU outer/strided-axis: FIXED** (commit "reductions: cache-friendly CPU
  outer-axis sum/mean"). `FlatOuterAxisSum` streams contiguous rows into
  per-thread double accumulators (reused thread_local scratch), then combines:
  ~22ms → ~0.67ms (~33x) on the measured case, double-accumulated. Guarded to
  leading-axes {0..k} float sum/mean via `TryFastCpuFloatSum`.
- **GPU global sum/mean: FIXED** (`reduce_cub.cu`, merged in #44). Routed to
  `cub::DeviceReduce::Sum` with a `thrust::make_transform_iterator` double
  accumulator + normalize/req kernel: float32 ~190→385 GB/s, float64 ~508.
- **GPU axis (non-global) reductions: IMPROVED** (commit "reductions(gpu): don't
  transpose large-N axis reductions"). The `calc_num_load` heuristic wrongly
  chose the transpose layout for trailing-axis reductions; disabling transpose
  when N>=8192 gives (256,256,256) axis=2: 187→317 GB/s (1.69x), (8192,256)
  axis=1: 89→131 GB/s, no regression elsewhere, accuracy unchanged. Remaining
  headroom (common cases ~35-40% BW) would need vectorized (float4) loads in the
  RTC reduce kernel — a larger kernel project, deferred.
- **mean dtype: FIXED** (same commit). `np.mean` of float64/float16 was being
  truncated to float32 by the FFI forcing the default dtype; now preserves the
  floating input dtype (NumPy/PyTorch semantics).

## Notes / smaller items

- **bf16 GPU** was broken (separate fixes landed this branch): elementwise/
  reduction aborted ("Unknown type flag 12" — missing `mshadow_type_info` case
  and RTC `bfloat16` type); `nd.dot(bf16)` segfaulted (now a clean error);
  `FullyConnected(bf16)` aborted (NVCC `MSHADOW_REAL_TYPE_SWITCH` omits bf16 —
  now dispatched explicitly to the bf16 `linalg_gemm`). Covered by
  `tests/python/gpu/test_bf16_gpu_ops.py`.
- **Dispatch overhead** (~34 µs/op async) is architectural (Python→FFI→engine→
  launch). It dominates tiny-op workloads and shows up as low GPU utilization on
  correctness suites; reducing it is an engine/FFI effort, not a kernel change.
- `nd.dot` supports fp16/fp32/fp64 only by design; bf16 matmul is reached via
  `FullyConnected` (and other linalg_gemm callers), not `nd.dot`.

---

## Retired claims (original 2026 Blackwell notes)

The original file claimed the build was "not yet performance-tuned" with fp16
matmul "~50× off peak", legacy `cublasGemmEx` "no longer competitive", the
cuDNN-9 RNN API gated out to `LOG(FATAL)`, and oneDNN at ~v0.21 with
`USE_ONEDNN=OFF`. Each was either Blackwell-specific or describes an earlier
port state; none hold for the current branch on Ada (see "What is NOT a
problem" above). Kept here only so the history is traceable.
