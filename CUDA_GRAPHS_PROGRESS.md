# CUDA Graphs revival — empirical progress log

Companion to `CUDA_GRAPHS_PLAN.md`. Records measured results per phase on this
fork's hardware. Host: 4× RTX 4090 (sm_89), CUDA 13.3 (V13.3.33), driver
595.71.05, cuDNN 9.x. Build: `config/build_gpu.cmake` (USE_CUDA, USE_CUDNN,
USE_ONEDNN, MXNET_CUDA_ARCH=8.9).

---

## Phase 0 — ground truth (2026-06-09, no code changes)

### Functional state
`MXNET_ENABLE_CUDA_GRAPHS=1 MXNET_USE_FUSION=0 pytest
tests/python/gpu/test_gluon_gpu.py::test_cuda_graphs` → **PASS** (5.8 s).
CUDA Graphs are functional on this CUDA 13 / sm_89 toolchain.

### What captures vs what is bypassed (from `MXNET_CUDA_GRAPHS_VERBOSE=1`)
- **Already captured (cuDNN + elementwise):** Convolution, Deconvolution,
  Pooling, BatchNorm, LayerNorm, InstanceNorm, Activation, LeakyReLU, and all
  the elementwise/broadcast ops. cuDNN capture works because the warm-up run
  primes the algo cache before capture — the plan doc was pessimistic here.
- **Bypassed (run conventionally between graphs):** `FullyConnected` /
  `_backward_FullyConnected`, explicitly `FIsCUDAGraphsCompatible -> false`
  (src/operator/nn/fully_connected.cu:80) because cuBLAS gemm is capture-illegal
  (cudaError 900). The dot / matmul / batch_dot family is excluded the same way.
- **Excluded, correct:** `Pad` (stateful), `Dropout` in train (RNG).

So the **cuBLAS gemm family is the one remaining prize** (Phase 2).

### Microbenchmark (bench_cuda_graphs.py, per-iter wall time, 4090)
| Model | graphs=off | graphs=on | speedup | note |
|---|---|---|---|---|
| chain (64 tiny tanh, all capturable) | 605.9 µs | 122.4 µs | **4.95×** | full dispatch win |
| mlp (8× Dense, FC bypassed) | 360.1 µs | 236.7 µs | 1.52× | ceiling set by FC/cuBLAS bypass |
| convnet (8× cuDNN conv) | 342.2 µs | 235.4 µs | 1.45× | conv captures already |

`mlp` is the Phase-2 yardstick: lifting the FC bypass should push it toward the
`chain` regime. The 4.95× on `chain` confirms graphs collapse per-op launch +
engine-dispatch overhead as designed (most valuable for many-small-op work).

---

## Phase 1 — capture summary + differential-replay net (2026-06-09)

Implemented in `src/imperative/cuda_graphs.h`; CI tests in
`tests/python/gpu/test_cuda_graphs_replay.py`.

### Capture summary (`MXNET_CUDA_GRAPHS_VERBOSE=1`)
One line per segment after capture:
`CUDA graph segment summary [<ops>]: N subsegs -> K graphs (M nodes), R bypassed ops.`
Makes graph coverage / regressions visible at a glance.

### Differential-replay net (`MXNET_CUDA_GRAPHS_VERIFY=1`)
For each captured subseg: snapshot all touched buffers, run the ops twice
conventionally (determinism self-check) and once as the graph, all from the
identical pre-segment state, and compare graph-vs-conventional outputs
(numpy-style `|a-b| <= atol + rtol*|b|`; fp16/bf16 widened; non-float byte-exact).
`LOG(FATAL)` on divergence. Knobs: `_VERIFY_EVERY`, `_VERIFY_RTOL` (1e-3),
`_VERIFY_ATOL` (1e-4).

Key design point — **determinism self-check**: run the conventional path twice;
if it isn't self-consistent (RNG, cuDNN dropout state, nondeterministic atomics)
graph-vs-conventional equality is not a valid test, so the segment is SKIPPED
(logged) rather than falsely flagged.

### Results (build-g, sm_89)
- `chain` (64-tanh) and `convnet` (conv+relu): captured as single graphs,
  **78 replay-OK each, 0 mismatch, bitwise-exact (max_rel=0)**.
- The net **caught a real divergence** before the determinism fix: cuDNN
  `Dropout(train)` output (its reserve/state buffer + fresh RNG mask) gave
  `max_abs=3.37e38` — confirming the comparison + FATAL path works. Now correctly
  SKIPPED as nondeterministic.
- CI: `test_cuda_graphs_replay.py` — 2 passed.

### Known limitation (acceptable)
VERIFY perturbs RNG state (extra conventional runs), so it cannot be combined
with tests that assert bitwise graphs-on == graphs-off for RNG ops (e.g. the
Dropout case in `test_gluon_gpu.py::test_cuda_graphs`). VERIFY is an opt-in debug
tool for deterministic segments — which is exactly the regime Phase 2 (cuBLAS
gemm) and Phase 3 (cuDNN conv/pool/norm) operate in.

---

## Phase 2 — cuBLAS gemm capture safety (2026-06-09)

Goal: let `FullyConnected` (and the gemm family) participate in CUDA graphs.

### Root cause (empirically isolated on CUDA 13.3 / sm_89)
Re-enabled FC capture behind `MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=1` and bisected the
err-900 with per-op capture-status probes (`MXNET_CUDA_GRAPHS_DEBUG_OPS=1`):
- The crash is at mshadow `Stream::Wait` (`stream_gpu-inl.h:92`) **during** op #1
  (FullyConnected); Activations capture fine.
- Hypothesis matrix (use_bias × MXNET_USE_CUBLASLT) was decisive:
  - legacy `cublasSgemmEx` path (`USE_CUBLASLT=0`): **crashes** (err 900), bias
    irrelevant.
  - **cuBLASLt path (`USE_CUBLASLT=1`): works** once the workspace is persistent.
- The real capture-illegal op was the cuBLASLt **per-call `cudaMalloc`/`cudaFree`
  for the matmul workspace** (`cublaslt_gemm.cc`); `cudaMalloc` during capture is
  illegal and silently *invalidates* the graph, cascading to the later `Wait`.
  (The plan had attributed blocker #1 to the engine's per-op `Wait`; on this
  toolchain that path isn't hit — the segment is one engine op, captured inline.)

### Fix (carefully)
1. **Persistent per-stream cuBLASLt workspace** (`cublaslt_gemm.cc`): allocate one
   buffer per `cudaStream_t` (sized to the 32 MiB cap), reuse it, never free
   per-call. Capture-safe (no alloc during capture) and concurrency-safe (matmuls
   on one stream are serialized; distinct streams get distinct buffers).
   `StreamWorkspace` refuses to allocate while capturing (returns nullptr →
   caller falls back) so a cold first-use during capture can't corrupt the graph.
2. **Auto-force cuBLASLt** when `MXNET_ENABLE_CUDA_GRAPHS && ALLOW_CUBLAS`
   (`UseCuBlasLt()`): warm-up (conventional) and captured runs use the SAME
   backend, so the captured graph matches normal execution and the
   differential-replay net is a valid check.
3. **No unsafe fallback during capture** (`linalg_impl.h`,
   `AssertGemmCaptureSafe`): all four scalar gemm variants (fp32/fp16/bf16/fp64)
   `LOG(FATAL)` with an actionable message if a legacy cuBLAS call is reached
   while capturing, instead of a cryptic err-900. Gated on `MXNET_ENABLE_CUDA_GRAPHS`
   so the default path pays nothing.

### Validation
- **Correctness:** mlp (7× Dense) now captures as **one 28-node graph, 0 bypassed
  ops**; differential-replay net **0 mismatch** (graph == conventional, bitwise).
- **cuBLASLt-vs-legacy fp32 drift = 0.000e+00 (bitwise identical)** across 6
  shapes × {dot, FC}; confirmed cuBLASLt served all 12 fp32 gemms, 0 fallbacks
  (`MXNET_CUBLASLT_VERBOSE=1`). Risk of numerical drift is empirically zero for
  fp32 on this toolchain (both paths use TF32; `CUBLAS_COMPUTE_32F_FAST_TF32`).
- Scope shipped: **fp32 FullyConnected**. fp16/bf16/fp64 FC capture is gated by
  the FATAL guard (clean failure) until separately validated; dot/batch_dot keep
  their existing `FIsCUDAGraphsCompatible=false` (untouched).

### Flags
- `MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=1` — opt FC into graph capture (off by default).
- `MXNET_CUBLASLT_VERBOSE=1` — log cuBLASLt served/fallback per gemm.
- `MXNET_CUDA_GRAPHS_DEBUG_OPS=1` — per-op capture-status trace during MakeGraph.

### fp16 coverage (update)
FC routes through cuBLASLt for **both fp32 and fp16** (the pseudo-fp16 path,
`MaybeCublasLtHgemm`: fp16 I/O, fp32 accumulate). Re-checked the earlier "fp16
didn't use Lt" reading — that was `dot`/`batch_dot` (a separate non-Lt gemm path,
not opted into capture), not FC. With the persistent workspace + forced-cuBLASLt:
- **fp16 FC captures correctly**: 6× Dense fp16 MLP → 28 replay-OK, 0 mismatch,
  0 guard-FATALs, FC inside the captured graph.
- **fp16 FC cuBLASLt-vs-legacy drift = 0.000e+00** (bitwise identical) across 4
  shapes — same as fp32.
- CI: `test_cuda_graphs_fc_cublaslt_capture[float32]` and `[float16]` both pass
  (fp16 at rtol/atol 1e-2).

So **fp32 and fp16 FullyConnected capture are both shipped and validated.**
bf16/fp64 FC and dot/batch_dot remain gated (clean FATAL / `FIsCUDAGraphsCompatible
=false`) pending their own validation.

### dot / batch_dot / matmul — investigated, deferred (operator reroute needed)
Explored extending capture to the rest of the gemm family. Finding: these ops do
**not** use the `linalg_gemm` cuBLASLt path that FC uses — they go through
**mshadow's legacy `BLASEngine`** (`cublasSgemm` / `cublasSgemmStridedBatched`,
`include/mshadow/dot_engine-inl.h`) and/or the tensordot machinery:
- `batch_dot`, `_npi_matmul` → `mshadow::BatchGEMM` (strided batched legacy gemm).
- `_npi_dot` → `TensordotImpl` (single legacy gemm); `dot`/`_backward_dot` are
  additionally excluded by `FComputeEx` dispatch.

All crash capture identically (`cublasSgemmStridedBatched` invalidates the
capture → err-900 at the next `Stream::Wait`), exactly analogous to FC's legacy
`cublasSgemmEx`. Tested `cublasSetWorkspace` on the mshadow blas handle — it did
**not** make the legacy path capture-safe (confirms the plan's earlier note);
reverted.

**Conclusion:** the proven fix is the same cuBLASLt route, but it requires
**operator-level rerouting** of these ops from `mshadow::BatchGEMM` to
`linalg_batch_gemm` (which already has a cuBLASLt strided path), plus a capture
guard on `linalg_batch_gemm`'s legacy fallback — OR routing mshadow's
`BLASEngine<gpu>` gemm through cuBLASLt (one place, captures the whole family,
but a deeper mshadow change). Deferred as a dedicated follow-up; `batch_dot`/
`matmul`/`dot` remain `FIsCUDAGraphsCompatible=false`. Reference for the reroute:
`src/operator/tensor/la_op-inl.h:89` (existing `linalg_batch_gemm` caller).

---

## Phase 2b — reroute dot family to linalg_batch_gemm (cuBLASLt capture) (2026-06-09)

Extended capture to the gemm family beyond FC. These ops used mshadow's legacy
`BLASEngine` (capture-unsafe), not the `linalg_gemm` cuBLASLt path.

### Rerouted (GPU) to the capture-safe cuBLASLt batched path
- **batch_dot** (`BatchDotForward_`, dot-inl.h) and **matmul** (`MatmulImpl`,
  np_matmul_op-inl.h): GPU gemm now goes through `linalg_batch_gemm`
  (cuBLASLt strided, capture-safe) instead of `mshadow::BatchGEMM`. Verified
  mapping: `BatchGEMM<tL,tR>(dst,lhs,rhs) == linalg_batch_gemm(lhs,rhs,dst,a,b,
  tL,tR)`. CPU path unchanged (mshadow), selected via `if constexpr`.
- Backward is covered: batch_dot's grad is built from `batch_dot` ops;
  matmul's backward reuses `MatmulImpl`.
- Attrs flipped to opt-in under `MXNET_CUDA_GRAPHS_ALLOW_CUBLAS`: `batch_dot`,
  `_npi_matmul`, `_backward_np_matmul`.

### Precision: TF32 *and* full fp32 (both supported)
batch_dot/matmul were full-fp32 via mshadow (`cublasSgemmStridedBatched`, default
math); the TF32 cuBLASLt path broke `test_batch_dot` (rtol 1e-3). Fix:
- Added `linalg_batch_gemm_fullfp32` (cuBLASLt `CUBLAS_COMPUTE_32F`, capture-safe)
  and `allow_tf32` flag on `MaybeCublasLtSgemmStrided`.
- Added fp16 (`linalg_batch_gemm<gpu, half>`) and capture guards on the
  float/double batched fallbacks.
- fp32 path honors `MXNET_CUDA_ALLOW_TENSOR_CORE` (fresh read, like cudnn_ops):
  **TF32 by default (fast), full fp32 when 0** — matching FC/la_op convention.
- `test_batch_dot` wrapped in `environment('MXNET_CUDA_ALLOW_TENSOR_CORE','0')`
  (cf. test_depthwise_convolution) to assert strict fp32.

### np.dot (_npi_dot) — excluded (latent crash fixed)
`_npi_dot`/`_backward_npi_dot` dispatch through tensordot (legacy mshadow gemm),
had no capture attr → would crash if captured. Marked
`FIsCUDAGraphsCompatible=false` (runs conventionally). Tensordot reroute is
future work.

### Results (build-g, sm_89)
- batch_dot: captures (TF32 + fp32), 0 mismatch; `test_batch_dot` PASS; correct
  vs numpy (fp32 ~1e-7, fp16 ~5e-4).
- matmul: captures, 0 mismatch; `test_np_matmul` 64/64 combos PASS on GPU.
- **Perf (batch_dot fp32, no regression):** graphs-off NEW ~207µs ≈ OLD mshadow
  ~220µs; captured TF32 ~100µs (2.2×), captured fp32 ~176µs.

### Regression check (Phase 2b)
- `test_cuda_graphs_replay.py` (FC fp32/fp16 + replay net): 4/4 pass.
- Gluon GPU suite: 391 pass, 10 skip, **1 fail = `test_dtype`** — confirmed
  PRE-EXISTING: reproduces on the old build (no Phase-2 changes) with the same
  seed (`MXNET_TEST_SEED=262818427`); a seed-dependent fp64 `cublasGemmEx`
  `CUBLAS_STATUS_NOT_INITIALIZED` quirk on CUDA 13, unrelated to CUDA graphs.
- matmul captured perf: graphs-off ~266µs → captured ~102µs (**~2.6×**).

## Status summary
Captured & validated (opt-in `MXNET_CUDA_GRAPHS_ALLOW_CUBLAS`): **FullyConnected,
batch_dot, matmul** (fwd+bwd), fp32 (TF32 default / full-fp32 via
`MXNET_CUDA_ALLOW_TENSOR_CORE=0`) and fp16. cuDNN conv/pool/norm already captured
(Phase 0). Excluded (safe): dropout-train (RNG), np.dot/tensordot, bf16/fp64 FC.
Remaining (separate efforts): Phase 3 cuDNN RNN, Phase 4 RNG-under-replay, Phase
5 default-on; tensordot reroute.

---

## Full GPU operator suite validation + stability fixes (2026-06-09)

Ran the full `test_operator_gpu.py` (13,189 tests incl. numpy ops on GPU),
sharded across 4 GPUs (rerunfailures disabled to dodge a pre-existing
pytest-socket-thread segfault). Triaged every non-pass:

1. **`test_sldwin_selfatten_operators` — REGRESSION I introduced, FIXED.**
   Sliding-window attention uses batch_dot; my "TF32-by-default" choice shifted
   its precision past rtol 1e-3. Fix: **batch_dot/matmul default to full fp32**
   (preserves the legacy mshadow precision — zero regression); **TF32 is opt-in
   via `MXNET_CUDA_ALLOW_TENSOR_CORE=1`**. This also let me revert the
   test_batch_dot / test_np_matmul tensor-core-off wrappers (no test changes
   needed). Capture works in both precisions (the speedup is from graphs, not
   TF32).

2. **`test_laop_2` segfault — PRE-EXISTING (cuBLAS-13 `syrk`), FIXED.**
   Reproduces on the old build (no Phase-2 changes); compute-sanitizer shows 0
   GPU errors → a host-side crash in legacy `cublas<t>syrk` on cuBLAS 13.5.x
   (same legacy-API breakage family as `cublasDgemm`→NOT_INITIALIZED). Crash
   surface mapped: only `syrk` (gemm2/trmm/trsm/potrf/gelqf all fine). Fix:
   compute the symmetric rank-k update as a **full gemm** (`B=α·op(A)·op(A)ᵀ`),
   numerically equivalent, using the working capture-safe gemm path
   (linalg_impl.h `LINALG_GPU_SYRK`). Verified: syrk now matches numpy and
   `linalg.gemm2` precision (incl. the pre-existing ~1e-8 fp64 gemm level, which
   is identical old-vs-new — a separate pre-existing CUDA-13 fp64 characteristic,
   not a regression).

Net: no regressions from the CUDA-graphs work; one pre-existing crash fixed for
stability.

3. **`test_dtype` (gluon) fp64 NOT_INITIALIZED — PRE-EXISTING, FIXED.**
   resnet18 cast to fp64 (fwd+bwd) hit `CUBLAS_STATUS_NOT_INITIALIZED` from the
   legacy `cublasGemmEx(fp64)` for certain shapes — deterministic (3/3), not
   seed-flaky; reproduces on the old build. cuBLASLt's fp64 path is reliable, so
   `linalg_gemm<gpu,double>` now ALWAYS tries `MaybeCublasLtDgemm` first (not only
   under MXNET_USE_CUBLASLT), falling back to cublasGemmEx. Full precision
   preserved (CUBLAS_COMPUTE_64F; gemm2 fp64 max_rel unchanged 4.46e-8) and now
   capture-safe. test_dtype: 3/3 pass.

### Final stability state
Full GPU operator suite (13,189 tests, 4-GPU shard): **13,124 passed, 0 failed,
0 segfaults**. All three triaged issues fixed (1 mine, 2 pre-existing). System is
stable: no crashes, no flaky failures.

### Final full validation (all fixes) — GREEN
Operator suite (4-GPU shard): 640+5496+6332+656 = **13,124 passed, 0 failed,
0 segfaults**. Gluon suite: **392 passed, 0 failed**. System stable.

PyTorch behavioral reference: fp32-default / TF32-opt-in for batch_dot/matmul
matches PyTorch's `torch.backends.cuda.matmul.allow_tf32 = False` default.

### End-to-end training validation (Phase-5 numerical guard)
A 4-layer MLP classifier + SGD, hybridized static, trained 30 steps with graphs
OFF vs ON (ALLOW_CUBLAS): loss trajectories are **bitwise identical** at every
step (FC fwd+bwd captured + optimizer step). Confirms the captured training path
matches conventional with zero drift. Committed on branch
`cuda-graphs-gemm-capture` (not merged). la_op gemm/gemm2 + standard optimizers
are already capturable (no host sync) and capture-safe via the linalg fix; adamw/
multi_lamb stay excluded (dynamic-scale host readback — cudaStreamSynchronize).

### Real-model validation + performance characteristics
- **resnet18 inference**: graphs-on == graphs-off **bitwise** (max_rel 0.0) — capture
  correct on a real conv+BN+FC model. The replay net SKIPS conv segments (cuDNN
  conv is non-deterministic across repeated calls; the determinism self-check
  catches it and avoids false failures) and verifies the deterministic ones.
- **Where capture pays off** (per-iter, graphs off → on):
  | workload | regime | off | on | speedup |
  |---|---|---|---|---|
  | chain (64 tiny elementwise) | dispatch-bound | 606 µs | 122 µs | 4.95× |
  | **transformer block** (B=1,T=32; FC+batch_dot) | dispatch-bound | 319 µs | 191 µs | **1.68×** |
  | matmul stack (B=8) | mixed | 266 µs | 102 µs | 2.6× |
  | batch_dot stack (B=8) | mixed | ~220 µs | ~100 µs | 2.2× |
  | resnet18 (B=8/B=1) | compute-bound (big convs) | ~1170/927 µs | ~1145/920 µs | ~1.02× |
  Capture collapses per-op launch+dispatch overhead → biggest win for many-small-op
  inference (attention/transformer, small batch); negligible for conv-bound nets.
  The transformer win is **enabled by this work** (FC+batch_dot capture; previously
  they fragmented the graph).

### la_op capture edge case — investigated, not an issue in practice
Concern: linalg.gemm/gemm2/syrk/trmm/trsm have no capture-exclusion attr, so under
graphs-on-without-ALLOW_CUBLAS they could hit the legacy gemm path during capture
and trip the AssertGemmCaptureSafe FATAL. Investigated empirically: (a) the numpy
gluon frontend (the one that produces capturable hybridized graphs) does not expose
these legacy linalg ops at all (np.linalg has no gemm2/syrk); (b) even via the
legacy symbol+CachedOp path, _linalg_gemm2 is NOT bulked into captured segments
(only the surrounding elementwise ops capture) → it runs conventionally → no FATAL.
So the guard is a true safety net that normal usage does not trip; no gating needed.
Documented for a future hardening pass if legacy linalg ever bulks into graphs.

### No default-path (graphs-off) regression
The changes compile into the normal path, so verified graphs-off perf old-vs-new:
mlp 365→347 µs, resnet18 1052→1033 µs (both within noise / marginally faster).
AssertGemmCaptureSafe is a single cached-bool fast-out when graphs are off (no
per-gemm CUDA call); the fp64 cuBLASLt-preference is fp64-only and fixes a crash.
Net: capture capability when graphs are enabled, zero cost to default execution.

---

## Phase 3 (cuDNN RNN capture) — investigated, blocked at executor level (deferred)
Attempted RNN capture. Findings:
1. **RNN never reaches the cuda_graphs capture path.** Adding an explicit
   `FIsCUDAGraphsCompatible` to RNN (to override OpOK's stateful veto) had NO
   effect: a hybridized static gluon LSTM shows ZERO capture activity (no
   "Capturing"/segment-summary/bypass lines). The gluon RNN layer
   (`rnn_layer.py`) uses a custom `forward`/`_forward_kernel` (Python shape
   checks + `to_device` + `npx.rnn`) that does not engage the static cached-op
   bulk segments (`CreateEngineOpSeg` → `CreateEngineOp` → cuda_graphs) the way
   conv/FC nets do. Root cause is at the frontend/executor routing layer, BEFORE
   the op attr matters.
2. Even once routed, the cuDNN RNN compute has capture-illegal ops: a per-call
   H2D seq-length memcpy (`rnn-inl.h:~887`, `cudaMemcpyAsync` from pageable host
   needed by cudnnRNNForward on cuDNN 8+); reserve/workspace/devSeqLengths are
   warm-up-allocated so those are fine for static shape.
3. The differential-replay net would also need to handle RNN's stateful I/O.

Conclusion (fused path): capturing the single fused `cudnnRNNForward` op is a
multi-layer effort (frontend/executor routing + cuDNN compute capture-safety +
state-aware verification), much larger than the gemm family. Deferred.

### Phase 3 — RESOLVED via the unrolled-RNN path (2026-06-09)
The reason MXNet RNNs are slow is **dispatch overhead**, and the fused
`cudnnRNNForward` op is *already* a single kernel launch — capturing it would
save almost nothing. The dispatch-bound case is the **unrolled RNN** (an
`LSTMCell` stepped over T timesteps = many small FullyConnected gemms +
elementwise ops), which is exactly the regime CUDA-graph capture accelerates.

That path **already captures** through the validated Phase-2 cuBLASLt gemm work:
an unrolled `LSTMCell` (states passed explicitly so it hybridizes into a static
cached-op) runs its i2h/h2h FC gemms through the capture-safe cuBLASLt path with
no new code.

Measured (unrolled LSTM, T/N/C/H = 10/4/32/32, sm_89):
- graphs OFF: 3053 µs/iter → graphs ON: 1312 µs/iter = **2.33× speedup**.
- attribution: OFF 4940 → FC-bypassed 2676 → FC-captured (Phase 2) 1533 µs —
  i.e. capturing the FC gemms is what unlocks the RNN speedup.

Verified for correctness by `test_cuda_graphs_unrolled_rnn_capture` (differential
replay: captured graph == conventional, FC inside a captured graph, no
capture-unsafe legacy cuBLAS fallback).

Net: the high-value RNN case (unrolled, dispatch-bound) is **done** and yields
~2.3×; the low-value fused-cudnnRNN op capture stays deferred. Recommend Phase 5
(default-on for the validated gemm+conv capture) for further immediate value.

---

## Phase 5 — default-on for the static-shape regime (2026-06-09)
CUDA-graph capture (incl. the Phase-2 cuBLASLt gemm family: FC / batch_dot /
matmul, fwd+bwd, fp32+fp16) now defaults **ON** for hybridized cached-ops with
`static_alloc=True && static_shape=True`, with **zero change to eager execution**.

### Why gated on static_shape (not just static_alloc)
The capture cache key is `{stream, is_train}` — no shape. A `static_alloc`-only
cached-op can change input shapes between calls, so capturing it would replay a
stale-dimension graph. Capture is only reached via `StaticInitExec`
(`CreateEngineOpSeg`, the static path), so gating the default on
`static_alloc && static_shape` is both necessary and sufficient for safety.

### Two coupled gates, two mechanisms
1. **Master switch** (`is_enabled_`): plumbed a `default_enable` flag from
   `CachedOp` (`static_alloc && static_shape`) → `CreateEngineOpSeg` →
   `CreateEngineOp` → `CudaGraphsExec`. `is_enabled_ =
   GetEnv("MXNET_ENABLE_CUDA_GRAPHS", default_enable)` — env still overrides
   (force-on for everything, or force-off).
2. **cuBLASLt backend** (`UseCuBlasLt`): the global static flag would, if simply
   defaulted on, force cuBLASLt for *all* gemms including eager — an unvalidated
   eager change. Instead a process-global runtime flag
   (`EnableCuBlasLtForGraphs`) is set from `CudaGraphsExec` the first time a
   capture-enabled segment is built (BEFORE warm-up, so the persistent per-stream
   cuBLASLt workspace is allocated conventionally, not during capture). Result:
   - graph-using processes → cuBLASLt for warm-up+capture (capture-safe);
   - **pure-eager processes never set the flag → legacy gemm, byte-identical to
     pre-Phase-5.**
   `MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=0` still opts gemm capture out
   (`AllowGemmCapture()`, default true). gemm `FIsCUDAGraphsCompatible` attrs now
   default true (OpOK eligibility only; no eager effect).

### Validation
- CI: `tests/python/gpu/test_cuda_graphs_replay.py` — 10/10 pass, incl. two new:
  - `test_cuda_graphs_phase5_default_on_static_regime`: a hybridized static net
    captures + FC-gemm-via-cuBLASLt with **no** ENABLE/ALLOW_CUBLAS env set.
  - `test_cuda_graphs_phase5_eager_unaffected`: a non-hybridized (eager) net
    builds **no** capture segments (summary never emitted).
- Speedup (transformer-ish hybridized static net, 6× attn+FFN block, sm_89):
  graphs OFF 1493.5 → **Phase-5 default 976.0 µs/iter = 1.53×**; explicit opt-in
  905.7 µs (same path). **Checksums bitwise-identical** across off/default/opt-in
  (25914.0137) — correct results.
- No eager regression: the d2l RNN notebooks run eager (no `hybridize()`), so the
  runtime cuBLASLt flag is never set and behavior is unchanged (see the RNN
  notebook A/B below).

### d2l RNN notebook A/B (eager no-regression gate, 4×4090)
Ran the d2l-neu RNN training notebooks (rnn-scratch, rnn-concise, lstm, gru,
deep-rnn, seq2seq) one-per-GPU. These run **eager** (no `hybridize()`), so they
test that the Phase-5 code changes don't regress the default eager path. They
also validate that the Phase-0–3 operator reroutes match the published reference.

Wall time (s), build-g pre-Phase-5 vs Phase-5 build, default env (no graph flags):

| notebook    | pre-P5 | Phase-5 | Δ      |
|-------------|--------|---------|--------|
| deep-rnn    | 471.8  | 472.9   | +0.2%  |
| lstm        | 309.7  | 303.2   | −2.1%  |
| gru         | 285.3  | 277.3   | −2.8%  |
| rnn-scratch | 240.1  | 234.2   | −2.5%  |
| rnn-concise | 163.6  | 158.4   | −3.2%  |
| seq2seq     |  19.1  |  18.2   | −4.7%  |

All within run-to-run noise (wall times include dataset download + kernel
startup + plotting), no regression. Separately, released-wheel (pre-CUDA-graphs)
vs our branch was within ±1.5% on the four substantial workloads.

Correctness vs the d2l.smola.org reference (executed under the released wheel):
- rnn-scratch perplexity 7.5 (ref 7.2); rnn-concise 7.4 (ref 7.0) — within RNG
  variance; generated "time traveller …" text the same char-level quality.
- seq2seq BLEU 0.687/1.0/1.0 (ref 1.0/0.658/0.0) — comparable, ours no worse.

## Status summary (updated)
gemm family (FC/batch_dot/matmul, fwd+bwd, fp32+fp16) and conv/pool/norm capture
**default-on** for the static_alloc+static_shape regime (Phase 5). Eager
execution unchanged. RNN: unrolled/dispatch-bound path captures (~2.3×); fused
cudnnRNN op capture deferred (low value). Remaining: Phase 4 RNG-under-replay
(dropout-train) — tractable (mxnet parallel RNG is device-resident philox);
tensordot/np.dot reroute.

---

## Phase 4 — RNG correct under replay (2026-06-09)
RNG ops now capture: `OpOK` admits ops whose only resources are capture-safe —
`kTempSpace`, `kParallelRandom`, and `kCuDNNDropoutDesc` (`ResourceCaptureSafe`).
Previously *any* non-tempspace resource vetoed capture, splitting the graph at
every dropout / random op.

### Why these are replay-safe (and `kRandom` is not)
- **`kParallelRandom`**: MXNet's per-thread Philox states live in a
  device-resident buffer; the kernel loads → advances → stores them on device, so
  each graph *replay* advances the RNG exactly as a conventional run would.
- **`kCuDNNDropoutDesc`**: `cudnnDropoutForward` advances a device-resident
  counter in the dropout state buffer; the descriptor is *restored* (not
  re-seeded) per call (`cudnnRestoreDropoutDescriptor`, no per-call alloc).
- **`kRandom`** (legacy curand host generator: np.random.*, shuffle, image aug,
  RNN's cuDNN dropout) stays excluded — its offset is bumped host-side, which a
  captured graph would bake once and repeat on every replay.

Note: cuDNN dropout was *already* `FIsCUDAGraphsCompatible` (and that attr
short-circuits `OpOK` before the resource check), so Phase 5's default-on already
captured it — this phase **verifies** that path is correct and additionally
admits the `kParallelRandom` ops (rrelu, legacy sample_*).

### Verification (build-g, sm_89) — replay advances RNG, sequence matches eager
Hybridized static net, training, per-iteration output sum, graphs-off vs
graphs-default (captured), seed fixed:
- **cuDNN dropout** (p=0.5): off 24/24 distinct sums; captured 24/24 distinct and
  **byte-identical sequence to off** (`first6=[422.92,360.1,188.78,264.23,293.46,
  170.3]`). No repeated mask; replay sequence == conventional.
- **rrelu** (`kParallelRandom`): off 20/20 distinct; captured 20/20 distinct,
  **byte-identical to off**.

CI: `test_cuda_graphs_dropout_rng_under_replay` (replay tail must not collapse to
one value AND graphs-on sequence == graphs-off). Full replay suite 11/11; eager
dropout/leaky_relu operator tests 8/8 (OpOK change doesn't affect eager).

Remaining: `kRandom` host-generator ops (would need a device-resident offset or a
host-side per-replay bump); tensordot/np.dot reroute.
