# Upstream apache/mxnet unresolved issues review (2026-05-17)

Background. apache/mxnet was archived 2023-11-17. All issues are now
read-only. This document mines the still-open issues to surface candidate
work items for the Blackwell port (`smolix/mxnet`). Nothing here was
posted or commented on apache/mxnet — read-only triage.

Queries used (all via `gh issue list --repo apache/mxnet`):

- 100 open `Bug` issues sorted by `reactions-desc`.
- 200 open issues (any label) sorted by `reactions-desc`.
- 100 open issues sorted by `updated-desc`.

Cross-referenced with `/workspace/mxnet/issues.md` and `/workspace/mxnet/CHANGELOG.md`.

---

## Already addressed in our fork

| #     | Title                                                                            | Why we already handle it                                                          |
| ----- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 21190 | Support building source against CUDA 12.1 (`cudaGraphExecUpdate` signature drift) | Ported to CUDA 13 in commit `dd7553781`; the same API drift fixed at that layer.  |
| 21208 | "no CUDA-capable device" on cu101 docker                                          | We ship cu13 wheel; user's environment was CUDA 10.1 driver vs lib mismatch.      |
| 21179 | cuDNN lib mismatch on `mxnet-cu112==1.9.1`                                       | We bundle cuDNN 9.22 in the wheel via `nvidia-cudnn-cu13>=9.22` dep.              |
| 21165 | MXNet does not support NumPy 1.24 (`np.bool` removed)                            | Tracked as our issues.md #44; numpy-2.x test fixes already landed (`fa51581cb`). Partial. |
| 13915 | `test_activation` softrelu backward flake                                        | Resolved in `8f6cc19ad` + cuDNN 9.22 bump; 4/4 PASS across 4 seeds.               |
| 18099 | AMP/RNN `waitall()` failure                                                      | Tracked as our issues.md #9 — cuDNN-v8 RNN rewrite (`817f5bea1`) is the substrate; AMP retest is the remaining piece. |
| 18564 | `test_gpu_memory_profiler_symbolic` tensordot attr mismatch                      | Tracked as issues.md #12; trivial — rename or restore the kernel tag.             |
| 20251 | AArch64 wheels                                                                   | Out of scope (we target x86_64 Blackwell). Listed for completeness.               |
| 18800 | `[RFC] v1.8.0 release` / archived-upstream concerns                              | Strategic — our fork is the answer (smolix/mxnet preview release shipped).        |

---

## A — Likely still broken in our fork (worth fixing)

These are real correctness or quality-of-life bugs that the Blackwell
port has not touched and that have a strong prior of still reproducing
on our code.

### A1. #21199 — Computation fails with oneDNN "could not create a primitive descriptor for a reorder" on 5D conv input (recent)
- 1×1 Conv with `num_filter=200` fed a 5D input (`[1,2,200,1,1]`) via `CachedOp` fails inside oneDNN with reorder pd-creation. 200→200 fails; 199 channels works — looks like a v2-vs-v3 layout / blocking edge case.
- Why this likely affects our fork. We are on oneDNN v3.11, the same path; reorder semantics changed in v3 and we already filed correctness item #4 (int8 quantized concat) against the same primitive family. 5D tensors flowing into a 4D conv via implicit reshape is exactly the kind of layout-blocking corner case oneDNN v3 made stricter.
- Suggested fix. Reproduce; run with `DNNL_VERBOSE=2` and inspect the rejected reorder format. Almost certainly fix is in `src/operator/nn/dnnl/dnnl_convolution.cc` or `dnnl_base.cc` — explicit `to_default_format` reorder before primitive create, or force `format_tag::any` and let oneDNN pick. Effort: S (1 day with verbose trace).
- Link: https://github.com/apache/mxnet/issues/21199

### A2. #17335 — Excessive GPU memory usage with dynamic shape input (13 reactions — most-felt open bug)
- Highest reactions among open bugs. With ThreadedEngine + dynamic shape Gluon DataLoader, memory pool grows unboundedly. Max batch 16 vs 256 for PyTorch on the same workload; NaiveEngine works fine, so it is a pool / cache leak in the threaded path.
- Why this likely affects our fork. We have not touched the memory pool; same code path. `pooled_storage_manager.h` allocates indefinitely on novel shapes. RTX PRO 4000 has only 24 GB — this will bite anyone running variable-length NLP / dynamic-image pipelines.
- Suggested fix. (a) Set `MXNET_GPU_MEM_POOL_TYPE=Round` as the default — already tracked as issues.md #22 — and document it; (b) instrument the pool to evict bucket entries older than N seconds; (c) cap per-bucket retention. Effort: M (need a benchmark harness to verify no regression on static shapes).
- Link: https://github.com/apache/mxnet/issues/17335

### A3. #18751 — `gluon.nn.BatchNorm` swaps `running_mean` and `running_var` on GPU (5 reactions, 20 comments)
- Reproduces on multiple CUDA versions (10.1, 10.2). All-ones input → on CPU `running_mean=1, running_var=0` (correct); on GPU `running_mean→0, running_var→1`. Trains a ResNet head with two BatchNorms → NaN within a few steps. Critical for any custom-head transfer learning.
- Why this likely affects our fork. The cuDNN BatchNorm path goes through `BNStatFinalize` / `BatchNormFwInferenceKernel`; we have not audited running-stat update order. Argument-binding bug in `cudnnBatchNormalizationForwardTraining` (running mean vs var swapped) survives across cuDNN versions because cuDNN does what it's told.
- Suggested fix. Grep `src/operator/nn/cudnn/cudnn_batch_norm-inl.h` for `runningMean` / `runningVariance` parameter order; compare to upstream cuDNN sample. Add a unit test mirroring the reporter's gist. Effort: S (probably one-line fix once located, 1 day with test).
- Link: https://github.com/apache/mxnet/issues/18751

### A4. #16686 — `grad_req='add'` numerical inconsistency vs manual accumulation
- Found while debugging BERT-divergence on GluonNLP. Doubling gradient-accumulation steps without changing LR causes divergence; manually accumulating into a side buffer does not. Differences localised to embedding & dense weights, observed on Mac (NaiveEngine) and G4 (ThreadedEngine).
- Why this likely affects our fork. Gradient accumulation is a first-class training pattern. The bug is in `WriteInplace` vs `AddTo` paths in `imperative/imperative_utils.h` and possibly in how the gradient buffer is initialised (one path zeros, the other reuses). NaN-on-divergence is hidden in big models — silent correctness bug.
- Suggested fix. Add a deterministic test: train a tiny MLP for N steps with `grad_req='write'` accumulating manually vs `grad_req='add'`; compare bitwise. If they differ, bisect through `Imperative::Backward` `AddTo` reduce path. Effort: M.
- Link: https://github.com/apache/mxnet/issues/16686

### A5. #11314 — `AddTakeGradLargeBatchCaller` (Embedding backward) non-deterministic NaN on GPU
- Embedding backward over large batch produces NaN at random positions in the weight gradient. Reproduces consistently on CUDA 9.2/p3 (Volta+), rarer on CUDA 9.0/p2. Old (2018) but never fixed.
- Why this likely affects our fork. Anyone training a model with a wide Embedding (NLP, recommender, sparse classifier) will hit this. The kernel is `src/operator/tensor/indexing_op-inl.cuh`; uses atomics — likely a missing zero-init of the temp buffer, or a race on partial scattered writes. CUDA 13 has stricter behaviour, may be worse not better.
- Suggested fix. Inspect `AddTakeGradLargeBatchKernel`. Typical pattern: `__shared__` accumulator not initialised to zero or `atomicAdd` to uninitialised memory. Effort: M (need a stress harness).
- Link: https://github.com/apache/mxnet/issues/11314

### A6. #19994 — `MXNDArraySyncCopyToCPU()` hangs after hours of inference
- `ThreadedEngine::WaitForVar` waits on a futex that never wakes. ARM (`aarch64-linux-gnu`) reproducer; after running for hours the engine deadlocks on var-dependency.
- Why this likely affects our fork. The engine code is unchanged. Long-running inference is a real use case (serving). Deadlock under load is unacceptable for the "frozen production stacks" audience that the port targets. Related to issues.md #18090 (CI deadlock) — possibly the same root cause.
- Suggested fix. Add a `WaitForVar` timeout-with-diagnostic mode; long-term, audit `ThreadedEngine::Push` for missing notify-on-dep-complete edges. Effort: L (engine work, hard to reproduce locally without a stress harness).
- Link: https://github.com/apache/mxnet/issues/19994

### A7. #18090 — Deadlock with ThreadedEngine (CI flake — same family as A6)
- CI jobs hang for 3 hours after the last test completes; no shutdown path. Specifically `unix-gpu` Python 3: GPU test step. Almost certainly the engine teardown / static-dtor sequencing.
- Why this likely affects our fork. Will bite us when we stand up CI on smolix/mxnet (issues.md #32). We already see fewer hangs because we removed `mx.test_utils.set_default_context` overuse, but the underlying engine teardown path is intact.
- Suggested fix. Force a `ShutdownEngine()` API and have it called from `atexit` in Python. Possibly also implicates #11163 (Windows DLL unload deadlock) — same shape, different OS. Effort: M.
- Link: https://github.com/apache/mxnet/issues/18090

### A8. #19353 — `linalg_impl.h` temp-buffer use without GPU synchronization  **FIXED**
- `potri`, `potrf`, `getri`, `getrf`, `gelqf`, `orglq`, `syevd`, `gesvd` all allocate a temp buffer, enqueue cuBLAS / cuSOLVER calls that use it, free it on CPU side while the kernels are still in flight. The freed buffer can be reassigned, then trashed by an unrelated stream → infrequent flaky NaN.
- Why this likely affects our fork. We added cuBLAS Lt scope (not adopted yet); none of these `linalg_impl.h` call sites have been audited. Linalg ops are used by VAEs, GPs, RL value-functions, attention with `det()` terms — niche but real. Plus CUDA 13's stricter ordering could expose this more often.
- Fix applied. Replaced `EPHEMERAL_GPU_STORAGE_ALLOC` macro + manual `Storage::Get()->Free()` with a RAII class `LinalgEphemeralGPUStorage` that calls `cudaStreamSynchronize()` before freeing the buffer in its destructor. 11 call sites patched (potrf×2, potri×2, gelqf, orglq, gelqf_workspace_query, syevd, gesvd, batch_getrf, batch_getri). No function signature changes. Stress test added: `tests/python/gpu/test_linalg_temp_sync.py` (5 tests × 200 iterations, all pass).
- Link: https://github.com/apache/mxnet/issues/19353

### A9. #18584 — `batch_dot` fp16 GPU precision vs `dot`
- `nd.dot` and `nd.batch_dot` give materially different answers on fp16 GPU inputs. With contrived bit-pattern inputs the difference is visible; on real attention QK^T workloads this means transformer logits are inconsistent across reshape boundaries.
- Why this likely affects our fork. The fp16 GEMM path goes through `cublasGemmStridedBatchedEx`; cuBLAS in CUDA 13 made the default accumulator selection different. Anyone training transformer-style models in fp16 on our wheel may see silent quality regressions vs cuDNN-attention paths.
- Suggested fix. Audit `src/operator/numpy/np_tensordot_op-inl.h` and `src/operator/tensor/dot-inl.h`. Force fp32 accumulator on batch_dot when input dtype == fp16. Add parity test between dot and batch_dot. Effort: S.
- Link: https://github.com/apache/mxnet/issues/18584

### A10. #19019 — AMP not reusing weights on recursive networks → GPU OOM
- AMP creates a fresh fp16 copy of weights every time a layer is invoked recursively. Long sequence RNN-style models OOM the GPU. Reporter's repro is a custom decoder iterating over a shared cell.
- Why this likely affects our fork. AMP is already item issues.md #8 (test_amp_subgraph.py 6 failures). This is a different AMP bug. Our 24 GB Blackwell card will trip on this faster than the reporter's 32 GB V100.
- Suggested fix. In `python/mxnet/contrib/amp/amp.py`, cache the fp16 cast NDArray keyed by `id(weight)`; check the cache before allocating a new cast op. Effort: M.
- Link: https://github.com/apache/mxnet/issues/19019

### A11. #18865 — `mx.random.seed` is order-dependent on multi-CPU
- Setting per-context seed twice in a row (one for cpu(0), one for cpu(1)) gives different results than setting one, sampling, then setting the next. Order dependency in seed propagation through the engine queue.
- Why this likely affects our fork. Reproducibility is a stated value of this port. Tests rely on `MXNET_TEST_SEED`; any pollution undermines retry-on-flake reasoning. The bug is in `Resource::ResetSeed` plus how the engine snapshots the RNG state on dispatch.
- Suggested fix. Make `mx.random.seed(ctx=...)` enqueue a `barrier_op` instead of a direct write. Effort: S–M.
- Link: https://github.com/apache/mxnet/issues/18865

### A12. #17495 — Singleton thread-safety (Engine, Storage, Resource managers)
- **STATUS: FIXED (2026-05-18) — see `singleton_thread_safety.md`**
- Full audit performed. All singletons in scope (Engine, Storage, ResourceManager, CpuEngine, OpenMP, TmpMemMgr, DNNLStream) already use C++17 magic-statics or `thread_local` and are safe.
- One genuine class-(c) issue found and fixed: `Profiler::Get()` in `src/profiler/profiler.cc` used a hand-rolled double-checked lock on a non-atomic `std::shared_ptr` (data race; UB). Replaced with a single function-local-static `std::shared_ptr<Profiler>` — compiler-emitted initialisation guard guarantees thread-safe one-time construction.
- New smoke test: `tests/python/unittest/test_threaded_init.py` — 8 simultaneous threads exercising singleton init. Passes 2/2.
- FC subgraph check: 387/0/16.
- Link: https://github.com/apache/mxnet/issues/17495

### A13. #11163 — Deadlock during DLL unload (Windows, but the pattern is real on Linux too)
- The Engine destructor calls `condition_variable::notify_all` from a static-local destructor under loader-lock. On Windows this deadlocks; on Linux it works mostly but can race with `dlclose` (e.g. unloading the C++ Python extension during interpreter shutdown).
- Why this likely affects our fork. Python-level `import mxnet` followed by interpreter exit occasionally segfaults; we've seen this in `test_custom_op_fork` audits and in CI cleanup. Hard to confirm without a clean repro but the diagnosis is sound.
- Suggested fix. Replace `static Engine` local with an explicit `Engine::Shutdown()` that the Python `atexit` calls; do nothing in the destructor. Effort: M.
- Link: https://github.com/apache/mxnet/issues/11163

### A14. #20447 — `[v2.0]` in-place ops change dtype (array-api spec violation)
- `a1 *= b1` with mixed dtypes raises "Type inconsistent" instead of casting. Array API spec says in-place must not change dtype/shape — current MXNet behaviour is neither casting nor silently keeping; it errors.
- Why this likely affects our fork. mx.np will diverge from numpy more as users adopt 2.x. Our `gluon.data` batchify fix (`7934d40d7`) is exactly this category. Right answer per array-api: cast `b1` to `a1.dtype` before the multiply.
- Suggested fix. In `python/mxnet/numpy/multiarray.py`'s `_wrap_mxnp_np_ufunc` for `__imul__` / `__iadd__` / `__isub__` / `__itruediv__`, pre-cast the rhs. Add a numpy-parity test. Effort: S.
- Link: https://github.com/apache/mxnet/issues/20447

### A15. #14264 — `nd.reshape` to a smaller shape silently truncates
- `mx.nd.arange(10).reshape((1,2))` returns `[[0,1]]` instead of raising. NumPy raises `ValueError`. Has been open since 2019.
- Why this likely affects our fork. Silent data loss is the worst class of bug. The C++ `Reshape` op tolerates unequal element counts; this should be a shape-check guard.
- Suggested fix. In `src/operator/tensor/matrix_op-inl.h::ReshapeShape`, add a `CHECK_EQ(in_size, out_size)` for the case where neither dim is `-1` / `0`. Mind keeping legacy `Reshape(0, ...)` placeholder semantics. Effort: S.
- Link: https://github.com/apache/mxnet/issues/14264

---

## B — Maybe already fixed by our port, needs validation

These look like they should have been caught by our cuDNN-9 / oneDNN-v3 /
CUDA-13 work, but no test has confirmed it. Each needs an explicit
verification before being closed out.

### B1. #17231 — Quantization example (`imagenet_gen_qsym_mkldnn.py`) segfaults
- Test plan. Run `python example/quantization/imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --calib-mode=entropy --num-calib-batches=10`. If it completes and produces a quantized symbol, this is closed by the oneDNN v3 fixes (`46ada1129`, `1df0ff579`, `740165f04`). Effort to verify: 1 hour.

### B2. #20675 — MXNet 2.0 up to 10x slower than 1.x on Windows — **VERIFIED N/A** (2026-05-18)
- The Linux CMakeLists.txt configures cleanly with `USE_ONEDNN=OFF -DUSE_INT64_TENSOR_SIZE=ON -DUSE_CUDA=OFF`; no error, no warning needed. The Windows-only slowdown is hardware/OS-specific and is out of scope for this Blackwell Linux port.

### B3. #19994 / #18090 deadlock pair
- Test plan. Run `tests/python/gpu/test_operator_gpu.py` overnight in a loop (>= 8 h). If it never hangs, our cuDNN-9 rewrite + the engine code path *may* have eliminated the deadlock. (Listed in A6/A7 too because a single passing run does not prove anything.)

### B4. #18121 — Sparse NDArray model load
- Test plan. Load a known-good `__storage_type__=1` (RowSparse) symbol via `mx.model.load_checkpoint`. We did not exercise sparse parameter load in this port; CHANGELOG says sparse-conversion was touched in `7934d40d7`. Effort: 1 hour to construct a 1.4-era saved model and load it.

### B5. #18923 / #13138 / #15540 — ONNX import paths
- Test plan. We already know ONNX module path was never updated for MXNet 2.0 (issues.md #14), so these import-from-ONNX bugs *will* still fail. They are reclassified here only because the fix is upstream (resolve our #14 first); if we fix #14, retest these as part of that.

### B6. #16933 — `setup.py bdist_wheel` lays out files wrong — **VERIFIED FIXED** (2026-05-18)
- `f8b0c7125` (wheel tag) + `83718e389` (pip-deps wheel) closed this.
- Confirmed: `mxnet/libmxnet.so` is in the wheel, `Root-Is-Purelib: false`, tag = `cp311-cp311-linux_x86_64`.

### B7. #19159 — GPU memory grows forever in Flask debug mode (multi-thread)
- Test plan. Same root cause as A2 (#17335) and A12 (#17495 thread-safety). Once those land, retest with the Flask repro.

### B8. #19218 — CPU inference very slow for some checkpoints (~110ms per conv)
- Test plan. Build resnet50 checkpoints at iteration 3 and iteration 23, run CPU inference on each, profile. If our oneDNN v3 + per-OC scale path is now used, this should be uniform. Effort: 2 hours.

---

## C — Not applicable to Blackwell port

One-line dismissals.

- #17887 — `mxnet-cu92` Python 3.8 import fails on Windows. CUDA 9.2 / Windows; we target CUDA 13 / Linux.
- #21208 — CUDA 10.1 docker container + driver mismatch. cu101; we target cu13.
- #21125 — Windows Server 2016 install. Windows.
- #21180 — Build a wheel for mxnet 1.9.1 on Windows. Windows + 1.x.
- #21209 — ppcle64 build. Wrong arch.
- #21170 — Install from source on macOS M1 Ventura. ARM macOS.
- #20992 — MxNet + GluonCV on Mac M1. macOS / ARM.
- #18774 — No mxnet binaries on CRAN for R 4.0.1. R bindings; out of scope.
- #18657 — "MXNET for cuda 11.0 Nvidia 450" — request for a cu110 wheel. Obsolete.
- #21210 — `mxnet-cu118` python package. Obsolete CUDA target.
- #20118 — Fail to install MXNet with cu110. Obsolete CUDA target.
- #20471 — Wrong gradients on Windows-GPU. Windows; we are Linux. (May still be a bug on Linux but reporter only saw it on Windows.)
- #17935 — Windows CI CUDA intermittent error C2993. Windows CI.
- #14203 — Cannot compile mxnet on Windows. Windows build.
- #17246 — R-package broken. R bindings.
- #17920 — R docs generation error. R bindings.
- #18774 — No mxnet R binaries on CRAN. R bindings (duplicate).
- #14589 — convert from mxnet to onnx failed. Old, ONNX-export path; covered by our issues.md #14.
- #19111 / #19583 / #16449 / #16552 / #15440 / #15438 / #13466 / #19369 / #18217 / #18962 — documentation / website bugs. Out of scope unless someone reorganises our docs.
- #21219 / #20897 — Graphviz version dependency bumps. Cosmetic.
- #19623 — Out of memory during compilation on CI. apache CI is gone.
- #21206 / #19139 / #18931 / #16167 / #17676 / #17783 / #18800 — RFCs (roadmap, JVM, etc.). Strategic, no concrete bug.
- #11163 — Windows DLL-unload deadlock. Pattern listed in A13 because it almost certainly recurs on Linux; the Windows-specific repro is dismissed.
- #18493 — 3D Upsampling missing. Feature request, not a bug.
- #14373 — Passing parameters to HybridBlocks and not using them. Confusing API behaviour; not a correctness bug.
- #16431 — RFC for multithreaded inference. Strategic, fed by A12.
- #11865 — `attach_grad` of intermediate variables loses gradient graph. 2018-era autograd corner case; very low traffic.
- #15540 — leaky_relu cannot broadcast (ONNX import). Out of scope while ONNX import is broken at collect time.

---

## Headline triage

If the next port milestone is "1.0 of the fork", the highest-leverage
items above (correctness > performance, breadth-of-impact > niche) are
**A1, A2, A3, A4, A5**. They each affect a separate axis: oneDNN
reorder edge case, GPU memory pool, BatchNorm correctness on GPU,
gradient-accumulation correctness, embedding-backward determinism. None
require breaking changes; each is a 1–3 day fix with a clear repro from
upstream.

A6 / A7 (engine deadlock) and A8 (linalg sync) are the next tier — real
but harder to repro / land safely.

A11–A15 are quality-of-life issues that are nearly trivial to fix once
prioritised; bundle them as a single "API-correctness pass" PR.

## Methodology

- Queried apache/mxnet via `gh issue list` three ways: top 100 open
  `Bug`-labelled by reaction count, top 200 open across all labels by
  reaction count, top 100 open by `updated-desc`.
- Fetched body + label set + last comment for the top ~35 candidates.
- Cross-referenced against `issues.md` (45 known-open items) and
  `CHANGELOG.md` (Blackwell-port commit-by-commit history) to discard
  already-fixed.
- Applied the requested filters: dismissed ancient OS / CUDA-9 / R /
  Windows-build / docs / RFC items as "not applicable" unless the
  underlying pattern (e.g., DLL-unload deadlock) generalises.
- Categorised remaining items A / B / C. Inside A, prioritised by
  (severity × user-impact × ease of repro).
- Output: 15 actionable A items, 8 B items, ~30 C dismissals.
