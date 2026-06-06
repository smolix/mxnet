# Plan: Revive and harden CUDA Graphs in MXNet

Status: proposal. CUDA Graphs are implemented but **off by default**
(`MXNET_ENABLE_CUDA_GRAPHS=0`). They worked in MXNet 1.x and were backported to
2.0 (commit `5ab7f64c2`, 2022). This plan brings them back to a reliable,
default-on-for-hybridized state, with a correctness story at each step.

---

## 1. What exists today

- **`src/imperative/cuda_graphs.h`** — the whole machinery, namespace
  `mxnet::cuda_graphs`:
  - `CudaGraphsExec::RunAll()` is the entry point, called from the engine
    closure in `src/imperative/imperative_utils.h:~1310` (`CreateEngineOp`),
    so graphs wrap **bulked op segments** of a hybridized/cached-op execution.
  - Capture uses `cudaStreamBeginCapture(..., cudaStreamCaptureModeThreadLocal)`
    → `cudaGraphInstantiate` → `cudaGraphLaunch`. Captured graphs are cached
    per `(cudaStream_t, is_train)`.
  - **Warm-up protocol**: the first execution of a segment runs conventionally
    (`has_been_run_conventionally`), so tempspace pointers settle before capture.
  - **Tempspace tracking**: `GetGPUTempspacePtrs()` snapshots the temp-space
    base pointers; if they change between runs, the graph is *updated in place*
    via `cudaGraphExecUpdate` (CUDA 12 path) instead of recaptured. If a *new*
    tempspace pointer appears *during* capture, it `LOG(FATAL)`s
    (`cuda_graphs.h:~446`) — a guard we must respect.
  - **Segment splitting**: `OpOK()` walks the segment; runs of capturable ops
    become graphs, non-capturable ops run conventionally between them.

- **Gating — `OpOK()` (`cuda_graphs.h:~525`)** rejects an op if any of:
  1. `FIsCUDAGraphsCompatible` (attr in `include/mxnet/op_attr_types.h:382`)
     returns false;
  2. it is stateful (`FStatefulCompute`/`FStatefulComputeEx`);
  3. it dispatches via `FComputeEx` (storage-fallback / DNNL path);
  4. it requests any resource other than `ResourceRequest::kTempSpace`
     (e.g. `kRandom`, `kParallelRandom`, cuDNN dropout state).

- **Per-op opt-outs**: ~31 `.cu` files already set `FIsCUDAGraphsCompatible`
  to false (or conditionally), e.g. `leaky_relu.cu`, `instance_norm.cu`,
  `dropout.cu` (only when `is_train`), `la_op.cu` (linalg), several
  `matrix_op.cu`, contrib `adamw.cu`, RNG ops.

- **Test**: `tests/python/gpu/test_gluon_gpu.py::test_cuda_graphs` exists,
  runs ~19 gluon blocks with `MXNET_ENABLE_CUDA_GRAPHS=1` + `MXNET_USE_FUSION=0`,
  hybridized `static_alloc=True, static_shape=True`, and compares against the
  non-graph result in both inference and training.

- **Env vars**: `MXNET_ENABLE_CUDA_GRAPHS` (default 0), `MXNET_CUDA_GRAPHS_VERBOSE`,
  `MXNET_CUDA_GRAPHS_DBG_FILE[_FLAGS]`, `MXNET_CUDA_GRAPHS_MAX_LOG_ENTRIES`.

**Conclusion:** the framework is ~90% there. The gap is (a) it's gated off,
(b) the most valuable ops (anything cuBLAS/cuDNN-based: FC, conv, matmul) are
excluded by rule 2/3/4, and (c) we lack evidence it's currently correct on this
CUDA 13 / sm_89 toolchain.

---

## 2. Why it's disabled / what actually breaks

Two classes of problem:

1. **Illegal operations during stream capture.** During capture you may not:
   `cudaMalloc`/`cudaFree`, synchronize, or do most library handle setup.
   - **Storage allocation**: any op that grows a storage/tempspace pool mid-exec
     calls `cudaMalloc` → capture aborts. The warm-up + tempspace-snapshot
     protocol handles the *common* tempspace case, but only for `kTempSpace`;
     anything else is excluded.
   - **cuBLAS / cuBLASLt**: `cublasSetStream`, workspace allocation, and the
     first call that lazily allocates internal buffers are capture-unsafe unless
     the handle+workspace are configured **before** capture and a fixed algo is
     used. Today these ops are stateful → excluded wholesale by `OpOK()` rule 2.
   - **cuDNN**: similar — algo search / workspace allocation must happen before
     capture; the conv/RNN ops are excluded today.

2. **Correctness hazards specific to graphs.**
   - **RNG**: ops using `kRandom`/`kParallelRandom` reuse a resource whose
     internal counter advances per call. A replayed graph must still advance RNG
     state, or every replay produces identical "random" output. These are
     excluded today (correct but conservative).
   - **Pointer stability**: a captured graph bakes in device pointers. If the
     cached-op re-plans memory (different tempspace base, different IO buffers),
     the graph must be updated. The `cudaGraphExecUpdate` path handles topology-
     identical updates; a topology change must trigger full recapture.

---

## 3. Revival steps (incremental, each independently shippable)

### Phase 0 — Prove the current state (no code changes)
- Build with graphs and run the existing test:
  `MXNET_ENABLE_CUDA_GRAPHS=1 MXNET_CUDA_GRAPHS_VERBOSE=1 MXNET_USE_FUSION=0`
  `pytest tests/python/gpu/test_gluon_gpu.py::test_cuda_graphs`.
- Capture the verbose log: which segments capture, which split, which ops are
  rejected and why. This is the ground truth for everything below.
- Add a microbenchmark (small MLP / conv stack, hybridized static) measuring
  per-iteration wall time graphs-on vs graphs-off, to quantify the dispatch win.
  Expectation: the win scales with op count and shrinks with op size — biggest
  for many-small-op inference (the dispatch-bound regime, cf. the 1024² reduce
  at 8% peak that is launch-bound).

### Phase 1 — Make the gating observable and safe-by-default
- Keep `MXNET_ENABLE_CUDA_GRAPHS=0` default.
- Add a one-line capture summary per segment (behind verbose): `N ops, K graphs,
  R rejected (reasons)`. Makes regressions and coverage visible.
- Add an assertion build mode: after a captured replay, in debug, re-run the
  segment conventionally into separate buffers and compare (bitwise for
  integer/copy ops, tolerance for float). This is the in-framework correctness
  net used throughout the later phases.

### Phase 2 — cuBLAS / cuBLASLt capture safety (the big payoff)
The goal is to let FC / matmul / dot participate.
- Move all capture-unsafe setup **before** capture:
  - Bind the stream to the cuBLAS handle once per stream (already done in
    `mshadow::Stream` init), never inside capture.
  - **Pre-allocate a fixed cuBLAS workspace** with `cublasSetWorkspace` (and a
    cuBLASLt workspace) sized once, owned by the stream/context — so no
    allocation happens during capture.
  - **Pin the algorithm**: use a fixed cuBLASLt algo (no heuristic search during
    capture) or run one warm-up call before capture so any lazy init is done.
- Mark these ops capturable via `FIsCUDAGraphsCompatible` returning true *only
  when* the capture-safe path is active (gate on a flag set after warm-up).
- Because FC/conv are currently `FStatefulCompute`, rule 2 in `OpOK()` must be
  relaxed for ops that explicitly declare graph compatibility: change the order
  so an explicit `FIsCUDAGraphsCompatible==true` overrides the stateful veto
  (stateful is a *heuristic* for "probably unsafe", not a proof).
- Validate with the Phase-1 comparison net across dtypes (fp32/fp16/bf16) and
  shapes, plus the existing gluon test extended with Dense/Conv-heavy blocks.

### Phase 3 — cuDNN ops (conv, pooling, norm, optionally RNN)
- Same recipe: algo selection + workspace allocation before capture (cuDNN algo
  cache already warms on first call — the warm-up run covers this), then declare
  compatible. Conv is the highest-value cuDNN op.
- Keep dropout-in-training and RNN excluded until Phase 4.

### Phase 4 — RNG correctness under replay
- For ops using `kRandom`: a replayed graph re-runs the *same* kernel with the
  *same* seed pointer, so we must advance the philox/curand counter outside the
  graph (host-side bump of the offset before each launch) or capture the counter
  update inside the graph against a device-resident counter that the kernel
  increments. Prefer the device-resident counter so replay is self-contained.
- Until proven, leave RNG ops excluded (current behavior) — they split the graph
  but stay correct.

### Phase 5 — Default-on for the safe regime
- Flip the default to on **only** for hybridized cached-ops with
  `static_alloc=True && static_shape=True` (stable shapes/pointers — the regime
  graphs were designed for), GPU context, and a CUDA/driver version we've
  validated. Imperative and dynamic-shape paths stay off.
- Keep `MXNET_ENABLE_CUDA_GRAPHS=0` as an explicit kill switch.

---

## 4. Testing / correctness strategy (the core ask)

Correctness is enforced at three layers, cheapest-first:

1. **In-framework differential replay (primary net).** A debug/opt-in mode
   (Phase 1) that, for every captured segment, executes the graph **and** a
   conventional run into shadow buffers and asserts equality (exact for
   integer/index/copy ops; `rtol/atol` for float, matching op tolerances).
   This catches *any* divergence — pointer staleness, RNG, stale workspace —
   at the segment granularity, with the exact op names in the failure. Run it
   over the whole gluon model zoo subset in CI.

2. **Op-level equivalence tests.** Extend
   `tests/python/gpu/test_gluon_gpu.py::test_cuda_graphs` into a parametrized
   suite: for each newly-enabled op family (Dense, Conv, Pool, BN/LN, matmul,
   dot) × dtype (fp32/fp16/bf16) × {inference, training}, build a tiny
   hybridized static block and assert graphs-on == graphs-off within tolerance,
   over N>=10 iterations (so a replay that "sticks" on stale data is caught by
   varying the input each iter). Include:
   - **Changing inputs each iteration** (catches a graph that captured a data
     pointer instead of replaying the copy).
   - **A tempspace-pointer-change scenario** (force a re-plan between iters) to
     exercise the `cudaGraphExecUpdate` path.
   - **An RNG block** (dropout in train) to assert outputs differ across iters
     once Phase 4 lands, and that the graph splits correctly before then.
   - **A segment with a deliberately non-capturable op in the middle** to
     verify the split logic still produces correct end-to-end results.

3. **Numerical end-to-end.** Train a small model a few steps graphs-on vs
   graphs-off; assert identical (fp64) / within-tolerance (fp16/bf16) loss
   trajectory. This is the integration-level guard against subtle ordering bugs.

CI matrix: run layers 1–2 on every change to `cuda_graphs.h` or any op's
`FIsCUDAGraphsCompatible`; layer 3 nightly. Always run with
`MXNET_CUDA_GRAPHS_VERBOSE=1` in CI so the capture/reject summary is in the log.

Determinism note: the differential-replay net must seed RNG and fix
`MXNET_ENABLE_CUDA_GRAPHS` per-process; capture is per `(stream, is_train)`, so
tests must not share a stream across the on/off comparison.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Relaxing the stateful veto re-enables a genuinely unsafe op | Opt-in per op via explicit `FIsCUDAGraphsCompatible==true`; never blanket-relax. Differential-replay net catches divergence. |
| cuBLAS/cuDNN lazy allocation during capture aborts capture | Mandatory warm-up call + pre-sized workspace before capture; verbose log asserts no capture abort. |
| Stale device pointers after memory re-plan | Existing tempspace snapshot + `cudaGraphExecUpdate`; add the IO-pointer-change test; `LOG(FATAL)` guard already trips on new tempspace during capture. |
| RNG replay produces repeated values | Keep RNG excluded until device-counter approach lands + test asserts cross-iter variation. |
| Driver/toolkit version differences (we're on CUDA 13/sm_89; graphs added on CUDA 11) | Phase 0 re-validates on current toolchain before anything else; gate default-on by validated version range. |
| Graphs help only dispatch-bound workloads, regress nothing but add capture overhead for short-lived graphs | Capture cost is amortized by the cache; for one-shot segments the warm-up path runs conventionally anyway. Microbenchmark in Phase 0 quantifies break-even. |

## 6. Expected payoff

Graphs collapse per-op launch + engine-dispatch overhead into a single
`cudaGraphLaunch`. The win is largest exactly where dispatch dominates: many
small ops (inference, small batch, elementwise/reduction chains). The 1024²
reduce at 8% of bandwidth (launch-bound) is representative — graphs would let
back-to-back small kernels run without per-launch CPU overhead. For large-op
training the relative win is smaller but nonzero. This is why graphs and the
"dispatch overhead" item are the same project: graphs are the concrete
mechanism that addresses dispatch without a bespoke scheduler rewrite.
