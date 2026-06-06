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

### GPU reductions run at ~30% of memory bandwidth (top opportunity)

`sum`, `max`, `min` (and reduction-backed ops) sustain only **~300 GB/s**
(~30% of the 1008 GB/s peak) for large inputs, across all output shapes
(scalar output and axis reductions alike). A bandwidth-bound reduction should
reach ~80–90% (cf. CUB/thrust); elementwise `add` on the same data already
reaches ~80%, so this is a ~2.5–3× gap specific to the reduction kernel.

- Location: `src/operator/tensor/reduce_rtc.cc` (RTC reduce kernels) +
  `ReduceImplConfig` in `src/operator/tensor/broadcast_reduce-inl.h`
  (constants `nthread_reduce=512`, `kBaseGridNum=1024`, `maxLoopPerTB=64`).
- Hypothesis (for the all-reduce N=1 case): `gridDim.x` collapses to 1, so all
  parallelism is in `gridDim.y=Mnext` (~512 blocks) with a long per-block serial
  loop (`maxLoopPerTB=64`) and shared-memory tree reduction; loads are not
  vectorized (no float4). Likely under-occupied and latency-bound.
- Why it's not patched here: the config is shared by every reduction shape and
  dtype; a correct fix needs vectorized loads and/or a retuned grid validated
  across the full (N, M, axis, dtype) space and ideally multiple archs, with
  before/after benchmarks per shape. That is a scoped optimization project, not
  a safe one-liner. Tracked for follow-up.

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
