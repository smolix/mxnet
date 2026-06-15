# What this fork fixes / changes

This is the consolidated change log for the **`smolix/mxnet`** fork — a maintained
port of the (archived) Apache MXNet 2.0 that runs existing/legacy MXNet code on
**CUDA 13 / Blackwell GPUs** and on **native Apple Silicon CPU**.

It supersedes the many per-topic engineering notes that used to live at the repo
root (bug audits, progress logs, scoping docs). Those notes have been retired; the
fine-grained history is preserved in `git log`. Anything still *open* lives in
[`OPEN_ISSUES.md`](OPEN_ISSUES.md) (summary) and
[`OPEN_ISSUES_DETAILS.md`](OPEN_ISSUES_DETAILS.md) (deep context).

Validation hosts referenced below: 4× RTX 4090 (Ada, `sm_89`), RTX PRO 4000 /
RTX 50-series (Blackwell, `sm_120`), AMD EPYC 7B12 (Zen 2 CPU), and Apple Silicon
(arm64). Current release line: `2.0.0+cu13.bw.<YYYYMMDD>` (latest published wheel
`2.0.0+cu13.bw.20260614`) plus the macOS CPU wheel `2.0.0+cpu.macos.<YYYYMMDD>`.

---

## 1. Platform & hardware enablement (the reason the fork exists)

- **CUDA 13.0 toolchain.** Builds and runs on the current CUDA 13 / cuDNN 9 /
  oneDNN v3 / NCCL 2.28 stack. Upstream was frozen at CUDA 11 / cuDNN 8 / oneDNN v2
  and does not build on modern toolchains or Blackwell GPUs.
- **Blackwell `sm_120` (consumer) and `sm_100` (datacenter).** The release wheel
  ships an explicit multi-arch fatbin — `sm_80` (A100), `sm_86` (RTX 30xx),
  `sm_89` (Ada/RTX 40xx, L40), `sm_90` (Hopper/H100), `sm_100` (B100/B200/GB200),
  `sm_120` (RTX 50xx, RTX PRO 6000) SASS, plus `compute_120` PTX forward-compat.
  Blackwell is two non-compatible families and `compute_120` PTX does **not** JIT
  down to `sm_100`, so both are listed explicitly.
- **cuDNN 9.x.** Including a rewritten v8-style RNN path (LSTM / GRU / vanilla RNN,
  fwd + bwd). Bumped 9.14 → 9.22 for better `sm_120` heuristic coverage; the wheel
  builds against 9.22/9.23. The first-GPU-use "cuDNN lib mismatch" warning now fires
  only on a *major*-version difference (cuDNN 9.x is minor-version ABI-compatible both
  directions), so a wheel built against 9.23 whose pin resolves 9.22 is silent
  (OI-20, `src/base.cc`).
- **cuBLAS ≥13.5 / driver R590+ (OI-19, accepted constraint).** The
  `nvidia-cublas>=13.5` pin is the permanent resolution of a two-sided squeeze:
  cuBLAS 13.1.1.3 (R580-safe) crashes `syrk`, while 13.2+ needs R590+ for large GEMMs
  — both cannot be satisfied, so the fork targets R590+ and drops the CUDA 13.0 / R580
  line (pin an older wheel there). Not a defect; closed as an inherent constraint.
- **NCCL 2.28** single-process / multi-GPU.
- **TF32 enabled by default on FP32 conv** (mirrors PyTorch/TensorFlow defaults;
  ~2.87× on `sm_120` vs the legacy non-TF32 mode). `batch_dot`/`matmul` default to
  full fp32 with TF32 opt-in.

## 2. Compute backends

- **oneDNN v3.11** (vendored under `3rdparty/onednn`) — float backend everywhere;
  full INT8 path on x86 (per-OC weight scales, fused conv/FC, fused sum,
  dequant-to-fp32 output).
- **cuBLASLt GEMM** (`MXNET_USE_CUBLASLT=1`) — heuristic-cached `cublasLtMatmul`
  for fp32 (PR-A) plus fp16/fp64 (PR-B), with a per-device handle + workspace +
  LRU algorithm cache. Bitwise-parity verified against the legacy path. Also used
  for capture-safe `batch_dot`/`matmul` (see §3).
- **cuDNN 9.14 → 9.22 bump** — closes the `sm_120` depthwise heuristic gap:
  depthwise 3×3 256→256 went **0.16 → 1.14 TFLOPS (~7×)**; other shapes within
  noise.
- **AMP subgraph bf16→fp32 fallback** — oneDNN v3 dropped AVX2 bf16 emulation, so
  on CPUs without AVX-512-BF16 the AMP subgraph ops (FC, conv, transformer QK/value
  matmuls, concat) now detect the ISA and fall back to fp32. All 6 AMP subgraph
  tests pass (were failing). *Backward through AMP subgraph ops is not yet
  validated — see OPEN_ISSUES.*
- **ONNX export/import** (PR #38) — defaults to the tested opset 13, exposes
  `opset_version`, emits explicit opset imports, fixes float16 constant encoding and
  `pooling_convention='full'` output-shape parity. Validated with ONNX 1.21 / ONNX
  Runtime 1.24 (`tests/python/onnx`: 10525 passed). *Note: the published wheels are
  built ONNX-free; enabling ONNX requires a source build — see OPEN_ISSUES.*

## 3. CUDA Graphs revival

CUDA Graphs work again on CUDA 13 and are **default-on for hybridized cached-ops
with `static_alloc=True` and `static_shape=True`** (eager execution unchanged).
Implemented and validated across phases (`src/imperative/cuda_graphs.h`,
`tests/python/gpu/test_cuda_graphs*.py`):

- Capture of conv/deconv/pooling/batchnorm/layernorm/instancenorm/activation/
  elementwise/broadcast, plus FullyConnected via the cuBLASLt path, and
  `batch_dot`/`matmul` rerouted to a capture-safe strided cuBLASLt gemm.
- Differential-replay validation: bitwise-identical outputs; non-deterministic
  segments correctly skipped, then RNG ops (`kParallelRandom`, cuDNN dropout)
  made replay-safe and admitted.
- Measured speedups: tiny-op chain ~4.95×; transformer-ish net 1493→976 µs/iter
  (1.53×); unrolled LSTM 3053→1312 µs (2.33×); resnet18 inference bitwise-identical
  graphs-on vs -off. End-to-end MLP+SGD training bitwise-identical across 30 steps.

## 4. Packaging — self-contained wheels

- **Linux CUDA wheel**: built `USE_OPENCV=ON`; bundles the OpenCV shared libraries
  **and their full transitive closure** (codecs, GDAL/GDCM, OpenEXR, tbb, jpeg/png/
  tiff/webp/openjp2, …) into `mxnet/lib/` with an `$ORIGIN` RUNPATH, so it imports
  on a clean host with no `libopencv-dev`. CUDA/cuDNN/NCCL come from pip `nvidia-*`
  wheels + the system CUDA toolkit (see [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md)).
- **macOS arm64 CPU wheel**: the same bundling on Mach-O — `tools/build_cleanup_wheel.sh`
  auto-detects macOS, walks `libmxnet.dylib`'s OpenCV closure with `otool`, copies
  the cores + full transitive closure into `mxnet/lib/`, rewrites every install name
  to `@loader_path/<sibling>` (the Mach-O analog of ELF `$ORIGIN`) and re-signs each
  dylib (`codesign -s -`, required on Apple Silicon). Ships Accelerate BLAS/LAPACK +
  float oneDNN; `OPENCV=True`; self-contained on a clean host.
- A **provenance gate** (`tools/release_provenance.py`) re-derives the feature set
  from the compiled binary + wheel contents and fails the build unless the promised
  features (CUDA/cuDNN/NCCL/oneDNN/OpenCV) are actually present.

## 5. Engine, concurrency & lifecycle

- **Cold-start engine deadlock** on first data-dependent op (`np.where`/`nonzero`)
  under load — fixed (PR #58): `LazyAllocArray::Get()` no longer holds
  `create_mutex_` across CUDA stream init; ops dropped during shutdown now finish
  inline. Validated under load; the quarantined test was un-skipped.
- **Shutdown ordering (A13)** — `MXNotifyShutdown` now calls `Engine::Stop()` so
  worker threads are joined before Python teardown (`test_engine_shutdown.py`).
- **CUDA event-pool recycling (T1)** — lapped event slots under-synchronized across
  streams; added `CUDAEventPool::IsLapped()` and grew the default pool 64→1024
  (`test_event_pool_recycling_gpu.py`).
- **Async callback errors** that were silently dropped now reach `WaitForAll` for
  read-only ops, no-var ops, and `NaiveEngine` (proof tests in the gtest suite).
- **Idempotent `ThreadedEnginePerDevice::Start()`**, rejection of negative bulk
  size / negative push counts before allocation, always-on allocation-free
  `CheckDuplicate`, and `Engine.RandSumExpr` race fix.
- **Profiler singleton** — replaced a hand-rolled double-checked-lock with a C++17
  magic-static (`test_threaded_init.py`).

## 6. Memory safety / use-after-free

- **Async UAF in numpy linalg cusolver wrappers (B2)** — QR/solve allocated device
  memory, launched async cusolver with no sync, then freed on CPU; switched to the
  ephemeral-GPU-storage pattern.
- **cuBLAS math-mode not restored on GEMM-fallback throw (B3)** — added an RAII
  guard so a thrown fallback no longer leaves the long-lived handle stuck in TF32.
- **cuBLASLt workspace `cudaMalloc` failure left a sticky CUDA error (B4)** — now
  cleared with `cudaGetLastError()` so it isn't misattributed to the next kernel.
- **Solver-handle ownership flag never reset (B5)** — one-line fix in mshadow.
- **19 unchecked `cudaMemsetAsync`/`cudaMemcpyAsync` returns (T3)** wrapped in
  `MSHADOW_CUDA_CALL`; **16 throwing destructors (T4)** given non-fatal CUDA-call
  variants so a failure in a dtor can't `std::terminate`.

## 7. Numeric correctness

- **fp16 reduction overflow (T5, OI-3)** — `np.mean`/`np.var`/`np.std` over a large
  axis reduced into fp16 before dividing and overflowed to `inf`. The mean/sum path
  reduces into fp32 scratch then casts; the variance/std *moments* path now runs the
  ENTIRE fp16 computation in fp32 — including the per-element `(data - mean)^2`, which
  a single large-magnitude element can overflow even when the final variance is O(1) —
  and casts only the finished result back to fp16 (`test_linalg_reduce_safety_gpu.py`).
- **Integer `dot` / `matmul` / `tensordot` (OI-1)** — these float-only kernels
  rejected integer inputs, which NumPy and PyTorch both compute. The eager frontend
  now follows PyTorch promotion: integer (non-bool) operands run in float64 and the
  result is cast back to the promoted integer dtype (`int @ int -> int`); mixed
  int/float keeps the float operand's width. Exact while products/sums stay within
  float64's 2**53 integer range. bool still rejects (PyTorch errors on bool matmul),
  and the symbolic path is unchanged (it cannot promote without an eager dtype).
- **`np.mean` / `np.average` of integers (OI-2)** — confirmed to follow NumPy
  semantics: an integer array returns the default float dtype with the *unrounded*
  true mean. (NumPy returns float64; PyTorch errors; neither rounds to int — the
  original "should round to int" framing was mistaken.) Pinned by a regression test.
- **flip / flipud / fliplr / rot90 are copies (OI-4, partial)** — these return
  independent copies, matching PyTorch (`torch.flip` & friends copy; they are not
  views). The values were always correct; the NumPy view-aliasing the regression
  sweep expected is a NumPy-only contract. (Positive-stepped-slice and axis-moving
  views, which PyTorch *does* return as views, still copy here — the narrowed OI-4.)
- **float64 accumulation** for LayerNorm / GroupNorm / BatchNorm training variance
  on large-finite float32 inputs (avoids precision loss / overflow).
- **PyTorch-convention type promotion** — int×float binary ops now promote to
  float-width (not NumPy's float64); `result_type`, weak-scalar promotion,
  `linalg.norm(int)`, `unique(float16)`, and `cross(int)` aligned to PyTorch.
- **np.cross backward** — four distinct bugs fixed (wrong `req` index for grad_b,
  missing broadcast reduction for lower-rank `b`, wrong output rank, wrong dtype
  dispatch).
- **`_npi_tril_indices` partial-output bind** returned unwritten (garbage) output;
  switched to per-output `KERNEL_ASSIGN` (commit `716370508`).

## 8. Robustness / bounds checking

- Out-of-bounds CPU index validation for `_contrib_index_copy`, `sparse.retain`,
  and ROI ops (`ROIAlign`, `PSROIPooling`, `DeformablePSROIPooling`, `RROIAlign`)
  batch ids / spatial dims.
- CPU RNN with `use_sequence_length` now raises `NotImplementedError` (hybridized
  CPU path isn't feasible) instead of producing wrong results.
- Custom-op profiler begin/end is balanced after exceptions.

## 9. Performance

- **CPU reductions** — global sum/mean via `FlatGlobalSum` + OMP reduction
  (float32 17→~48 GB/s ~2.8×, float64 ~91 GB/s ~5×); outer/strided-axis via
  cache-friendly per-thread double accumulators ((4096,4096) axis=0 ~22 ms →
  ~0.67 ms, ~33×).
- **GPU reductions** — global sum routed to `cub::DeviceReduce::Sum` (#44):
  float32 ~190→385 GB/s; axis reductions improved via a `calc_num_load` heuristic.
- **bf16 GPU FullyConnected** — added the missing `mshadow_type_info` case + RTC
  bfloat16 type and dispatch (`test_bf16_gpu_ops.py`).

## 10. Legacy-issue backlog & downstream compatibility

- **Apache open-issue audit** — scanned ~750 open `apache/mxnet` issues + 69 PRs and
  added **223 runtime-verified regression tests**; **61+ issues** confirmed fixed in
  this fork (e.g. #21176, #20936, #19422, #18300, #13945, …). 6–10 NumPy view/stride
  contract cases remain xfail (see OPEN_ISSUES).
- **d2l.ai book** — `np.argmax` GPU size-1 axis, GPU OOM retry-with-backoff
  (`MXNET_GPU_MEM_POOL_OOM_RETRIES`), stale `mxnet.__version__` regeneration, storage
  banner gated behind `MXNET_LOG_STORAGE_INIT=1`, and a learning-rate scheduler
  `epoch_size=` kwarg so `MultiFactorScheduler`/`CosineScheduler` count epochs, not
  minibatch steps. The two convergence gaps (#6 scheduler `epoch_size`, #7 FCN
  `trainer.step`) are resolved book-side (OI-29). All D2L notebook compatibility is
  now green, including the `train_ch13` multi-GPU path (OI-28).
- **NumPy 2.x operator shape/axis params (toward OI-26)** — tuple params containing
  NumPy scalars (e.g. from `rand_shape_nd` / `np.random.randint`) rendered as
  `(np.int64(3), …)` under NumPy 2.x — the scalar-`repr` change, since `str(tuple)`
  uses `repr` per element — which the C++ Shape/Tuple parser rejected with "Invalid
  Parameter format for shape". `base.param_str` now coerces NumPy scalars in tuple/list
  params to plain Python scalars before stringifying (the ndarray imperative-invoke and
  both symbol attr paths). Byte-identical to NumPy 1.x for previously-working inputs, so
  it is non-breaking. Full NumPy 2.x ABI validation across the suite is still open
  (OI-26).

---

*For the build/release recipe see [`docs/cuda_wheel_build.md`](docs/cuda_wheel_build.md)
and [`BUILDING.md`](BUILDING.md). For installation see [`README.md`](README.md).*
