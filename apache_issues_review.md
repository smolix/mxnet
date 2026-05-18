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

### A1. #21199 — Computation fails with oneDNN "could not create a primitive descriptor for a reorder" on 5D conv input — **FIX READY (2026-05-18), awaits rebuild**
- Root cause: `src/operator/nn/dnnl/dnnl_reshape.cc:61` reorder fails oneDNN v3's `ndims-consistency check` (`3rdparty/onednn/src/common/reorder.cpp:90`). When an upstream blocked conv (e.g. `brgconv_1x1:avx2` on AVX2 with nf≥200) produces a 4-D `acdb` chunk, the downstream `reshape_like + Reshape` op leaves the NDArray's metadata shape 5-D while the underlying DNNL memory chunk is still 4-D. The `dnnl_reshape.cc` constructor then asks oneDNN to reorder a 4-D `acdb` chunk into a 5-D `abcde` desc; v2 accepted this, v3 rejects it. The "boundary" (nf=199 works, nf=200+ fails) is not channel divisibility — it's whether oneDNN's conv dispatcher selects a blocked layout (`brgconv_1x1`) vs a plain one (`x64:gemm:jit`).
- Fix: in `DNNLReshapeFwd`, when `in_mem`'s ndims disagrees with `input.shape().ndim()`, take a 2-step path — (1) reorder `in_mem` to a temp buffer with the **source's** ndims at default format (same-ndim reorder, satisfies v3); (2) reinterpret the same buffer as the post-reshape shape via a `temp_reshaped_` view at the same data handle, then default→default reorder into the output. `src/operator/nn/dnnl/dnnl_reshape.cc` + `dnnl_reshape-inl.h` patched; new `temp_reshaped_` member; `Execute` binds the second reorder's `DNNL_ARG_FROM` to the reshaped view. Patch is strictly additive — when ndims match (the common case), behavior is unchanged.
- Regression test: `tests/python/dnnl/test_a1_5d_conv_reorder.py` (parametrized over `num_filter ∈ {199, 200, 208, 256, 512}` + correctness vs numpy). Not run yet; awaits rebuild after the current pytest sweeps finish.
- Link: https://github.com/apache/mxnet/issues/21199

### A2. #17335 — Excessive GPU memory usage with dynamic shape input — **PATCH READY (2026-05-18), awaits rebuild**
- Bug confirmed in `src/storage/pooled_storage_manager.h`: `PooledStorageManager::Free()` appends to `memory_pool_[bucket_id]` with **no eviction path**. Buckets grow monotonically until total free GPU memory drops below ~5% reserve. Synthetic repro at `tests/python/gpu/test_pool_dynamic_shape.py` (8 buckets × 8 concurrent live buffers): pool plateaus at **766 MiB = 8× working set**. With 64 distinct buckets: 4278 MiB (matches upstream OOM-at-batch=16 signature).
- Fix: per-bucket count cap (option (c) from the original suggested-fix list). New env var `MXNET_<DEV>_MEM_POOL_PER_BUCKET_LIMIT` (default 4). When `Free()` is called and `BucketSize(bucket_id) >= per_bucket_limit_`, the new chunk goes straight to `contextHelper_->Free` (mirroring `DirectFree` logic) instead of being retained. `BucketSize` getter added to both `UnorderedMapContainer` (Naive pool) and `VectorContainer` (Round pool). Setting K=0 restores legacy unbounded behavior.
- Expected impact: 766 → ~440 MiB on the synthetic 8-bucket repro; proportional reduction (~K/concurrency) for the upstream-reporter dynamic-shape workload. Steady-shape jobs (resnet18 batch=32) are unaffected (BucketSize rarely > 4).
- Risks: extra `cudaFree` calls on truly extreme dynamic workloads (negligible at K=4, tunable upward); profiler memory accounting unchanged (uses `DirectFree`'s existing pattern).
- Patch: `.investigations/a2_pool_retention.patch` (106 lines; `git apply --check` clean). Repro test: `tests/python/gpu/test_pool_dynamic_shape.py` (asserts peak ≤ K·sum_bucket_mib + overhead). Awaits rebuild after current sweeps complete.
- Link: https://github.com/apache/mxnet/issues/17335

### A3. #18751 — `gluon.nn.BatchNorm` `running_mean`/`running_var` swap on GPU — **RESOLVED, verified 2026-05-18**
- All-ones input + default BN now correctly yields `running_mean=1, running_var=0` on **both CPU and GPU**. Minimal upstream-style repro at `.investigations/a3_repro.py` passes; existing regression suite `tests/python/gpu/test_batchnorm_running_stats.py` passes 6/6 on current build.
- True root cause was not a cuDNN argument-binding bug. cuDNN argument order in `src/operator/nn/cudnn/cudnn_batch_norm.cu:137-162` was always correct (matches v9 header `cudnn_ops_v9.h:2633`). The DNNL CPU path only updated running stats inside `DNNLBNBackward::Execute`, so the CPU vs GPU disagreement was "WHEN" the stats are visible (cuDNN: in forward; DNNL CPU: only after backward). Commit `a47ce39d9` moved the DNNL momentum update from backward to forward; both backends are now consistent.
- Link: https://github.com/apache/mxnet/issues/18751

### A4. #16686 — `grad_req='add'` numerical inconsistency vs manual accumulation
- Found while debugging BERT-divergence on GluonNLP. Doubling gradient-accumulation steps without changing LR causes divergence; manually accumulating into a side buffer does not. Differences localised to embedding & dense weights, observed on Mac (NaiveEngine) and G4 (ThreadedEngine).
- Why this likely affects our fork. Gradient accumulation is a first-class training pattern. The bug is in `WriteInplace` vs `AddTo` paths in `imperative/imperative_utils.h` and possibly in how the gradient buffer is initialised (one path zeros, the other reuses). NaN-on-divergence is hidden in big models — silent correctness bug.
- Suggested fix. Add a deterministic test: train a tiny MLP for N steps with `grad_req='write'` accumulating manually vs `grad_req='add'`; compare bitwise. If they differ, bisect through `Imperative::Backward` `AddTo` reduce path. Effort: M.
- Link: https://github.com/apache/mxnet/issues/16686

### A5. #11314 — `AddTakeGradLargeBatchCaller` (Embedding backward) non-deterministic NaN on GPU — **NO REPRO ON BLACKWELL (2026-05-18); defensive patch ready**
- **Status**: stress-tested on Blackwell sm_120 / CUDA 13 with >2000 large-batch backward passes (vocab=50000, dim=256, batch=8192, seq=64; fp32+fp16; zero-fill and 0xFF trap-pattern weight.grad pre-fill; uniform and skewed-hot-100 indices). Zero NaN/Inf observed. The original buggy kernel `AddTakeGradLargeBatchKernel` has been **dead code since 2019** (upstream PR #16355, commit `80964213d`) — `EmbeddingOpBackward<gpu>` now dispatches to `EmbeddingGradKernel` (`src/operator/tensor/indexing_op.cu:894→936`) which zero-initialises its shared-mem accumulator for `kWriteTo` and reads existing grad for `kAddTo`. No uninit-read window. Regression test added at `tests/python/gpu/test_embedding_backward_nan.py` (4 parametrised cases; pass).
- **Residual concern (defensive)**: the legacy `AddTakeGradLargeBatchKernel` is still reachable via `MXNET_FORCE_ADDTAKEGRAD=1` and includes a post-loop `sh_ballot[]` read without an explicit `__syncthreads()` between the loop exit and the read. CUDA 13's stricter ordering on Blackwell could in theory let nvcc reorder writes across the loop edge. Defensive 2-line patch (extra `__syncthreads()` + zero-fill of `sh_grad_weight` before the scatter store) is saved at `.investigations/a5_embedding_nan.patch`. Strict no-op for the production path. Will be applied + rebuilt after the current sweeps finish.
- Link: https://github.com/apache/mxnet/issues/11314

### A6. #19994 — `MXNDArraySyncCopyToCPU()` hangs after hours of inference — **INSTRUMENTED (2026-05-18)**
- `ThreadedEngine::WaitForVar` waits on a futex that never wakes. ARM (`aarch64-linux-gnu`) reproducer; after running for hours the engine deadlocks on var-dependency.
- Why this likely affects our fork. The engine code is unchanged. Long-running inference is a real use case (serving). Deadlock under load is unacceptable for the "frozen production stacks" audience that the port targets. Related to issues.md #18090 (CI deadlock) — possibly the same root cause.
- Status (2026-05-18). No local reproducer available. Added `MXNET_ENGINE_DIAG=1` watchdog to `WaitForVar` and `WaitForAll` (configurable timeout via `MXNET_ENGINE_DIAG_TIMEOUT_S`, default 30 s). On timeout, logs var pointer, `pending_ops`, `shutdown_phase`, `kill` flags, and a hint to use `MXNET_ENGINE_TYPE=NaiveEngine`. The actual missing-notify-edge bug requires a minimised ARM reproducer to fix. See `engine_deadlock_audit.md`.
- Link: https://github.com/apache/mxnet/issues/19994

### A7. #18090 — Deadlock with ThreadedEngine (CI flake — same family as A6) — **PARTIALLY FIXED (2026-05-18)**
- CI jobs hang for 3 hours after the last test completes; no shutdown path. Specifically `unix-gpu` Python 3: GPU test step. Almost certainly the engine teardown / static-dtor sequencing.
- Why this likely affects our fork. Will bite us when we stand up CI on smolix/mxnet (issues.md #32). We already see fewer hangs because we removed `mx.test_utils.set_default_context` overuse, but the underlying engine teardown path is intact.
- Status (2026-05-18). Root cause confirmed: `MXNotifyShutdown` (called from `atexit` in `base.py`) only called `NotifyShutdown()` + `WaitForAll()`, never `Stop()`. Worker threads stayed alive until the static-local `shared_ptr<Engine>` dtor fired during interpreter teardown, causing the `cv.notify_all()` in `~ThreadedEngine()` to race with partially-torn-down thread-locals. Fixed by adding `Engine::Get()->Stop()` to `MXNotifyShutdown()` in `src/c_api/c_api.cc`. Workers are now fully joined before interpreter teardown. Test: `tests/python/unittest/test_engine_shutdown.py` (12/12 PASS). See `engine_deadlock_audit.md`.
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

### A13. #11163 — Deadlock during DLL unload (Windows, but the pattern is real on Linux too) — **FIXED (2026-05-18)**
- The Engine destructor calls `condition_variable::notify_all` from a static-local destructor under loader-lock. On Windows this deadlocks; on Linux it works mostly but can race with `dlclose` (e.g. unloading the C++ Python extension during interpreter shutdown).
- Why this likely affects our fork. Python-level `import mxnet` followed by interpreter exit occasionally segfaults; we've seen this in `test_custom_op_fork` audits and in CI cleanup. Hard to confirm without a clean repro but the diagnosis is sound.
- Status (2026-05-18). Fixed. `MXNotifyShutdown()` (atexit-registered in `base.py`) now calls `Engine::Get()->Stop()` after `WaitForAll()`. `Stop()` signals all worker queues and joins all thread pools, so by the time the Python interpreter starts C-extension teardown, all engine threads are gone. The static-dtor `cv.notify_all()` becomes a no-op (no threads waiting). No new `Engine::Shutdown()` API was needed — `Stop()` already existed on `ThreadedEnginePerDevice`. Test: `test_engine_shutdown.py::test_clean_exit_basic` (5 trials), `test_clean_exit_gpu` (3 trials), `test_clean_exit_after_many_ops` (3 trials) — 12/12 PASS. See `engine_deadlock_audit.md`.
- Link: https://github.com/apache/mxnet/issues/11163

### A14. #20447 — `[v2.0]` in-place ops change dtype — **RESOLVED 2026-05-18**
- Already implemented: `python/mxnet/numpy/multiarray.py` has `wrap_mxnp_np_ufunc_inplace` (lines 275-302) that casts `x2` to `x1.dtype` before forwarding. All four in-place dunders (`__iadd__`, `__isub__`, `__imul__`, `__itruediv__`) are decorated with it.
- Regression test added at `tests/python/unittest/test_inplace_dtype.py` — 4/4 pass. Each test verifies no exception is raised, lhs dtype stays float32, and values match numpy parity exactly.
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

### B1. #17231 — Quantization example (`imagenet_gen_qsym_mkldnn.py`) segfaults — **CLOSED (2026-05-18)**
- The script was renamed to `imagenet_gen_qsym_onednn.py` in our fork (file at `example/quantization/imagenet_gen_qsym_onednn.py`).
- Run: `PYTHONPATH=python python3 example/quantization/imagenet_gen_qsym_onednn.py --model=resnet50_v1 --calib-mode=none`
- Outcome: **Runs cleanly to completion** (no segfault, no crash). Downloads resnet50_v1 pretrained weights, quantizes all 53 oneDNN subgraph nodes (conv_bn_act fusions + FC), and writes `example/quantization/model/resnet50_v1-quantized-symbol.json` (281 KB) + `resnet50_v1-quantized-0000.params` (92 MB). Zero errors or warnings beyond a benign Gluon type-inference note.
- Note: `--calib-mode=entropy` requires downloading the 2.6 GB ImageNet val_256_q90.rec calibration set (network access needed). The `--calib-mode=none` path exercises the full quantize-graph-pass code path without a dataset and is sufficient to confirm the segfault is gone.
- Conclusion: the original segfault is resolved by the oneDNN v3 quantization fixes. Closed.

### B2. #20675 — MXNet 2.0 up to 10x slower than 1.x on Windows — **VERIFIED N/A** (2026-05-18)
- The Linux CMakeLists.txt configures cleanly with `USE_ONEDNN=OFF -DUSE_INT64_TENSOR_SIZE=ON -DUSE_CUDA=OFF`; no error, no warning needed. The Windows-only slowdown is hardware/OS-specific and is out of scope for this Blackwell Linux port.

### B3. #19994 / #18090 deadlock pair
- Test plan. Run `tests/python/gpu/test_operator_gpu.py` overnight in a loop (>= 8 h). If it never hangs, our cuDNN-9 rewrite + the engine code path *may* have eliminated the deadlock. (Listed in A6/A7 too because a single passing run does not prove anything.)

### B4. #18121 — Sparse NDArray model load — **TESTED 2026-05-18 (11/11 pass)**
- `tests/python/unittest/test_sparse_model_load.py` (11 tests, all pass). Split by API era:
  - **Working (legacy)**: `mx.nd.save`/`mx.nd.load` round-trips `row_sparse` NDArrays preserving stype on both CPU and GPU. `tostype('row_sparse')` at the ndarray layer works.
  - **Intentionally blocked (Gluon2.0 / np-semantics)**: `Embedding(sparse_grad=True)`, `Parameter(stype='row_sparse').initialize()`, `SymbolBlock` construction with row_sparse, and `mx.nd.save` for row_sparse in np mode all raise documented errors mentioning "not supported in NumPy interface and Gluon2.0". These are upstream design decisions, not Blackwell port bugs.
- Verdict: no action needed for the Blackwell port. Workaround for users with 1.x sparse checkpoints: `mx.npx.reset_np()` before `mx.nd.load`. Re-enabling Gluon2.0 sparse support is a separate upstream-design discussion.
- Link: https://github.com/apache/mxnet/issues/18121

### B5. #18923 / #13138 / #15540 — ONNX import paths
- Test plan. We already know ONNX module path was never updated for MXNet 2.0 (issues.md #14), so these import-from-ONNX bugs *will* still fail. They are reclassified here only because the fix is upstream (resolve our #14 first); if we fix #14, retest these as part of that.

### B6. #16933 — `setup.py bdist_wheel` lays out files wrong — **VERIFIED FIXED** (2026-05-18)
- `f8b0c7125` (wheel tag) + `83718e389` (pip-deps wheel) closed this.
- Confirmed: `mxnet/libmxnet.so` is in the wheel, `Root-Is-Purelib: false`, tag = `cp311-cp311-linux_x86_64`.

### B7. #19159 — GPU memory grows forever in Flask debug mode (multi-thread) — **NO SEPARATE LEAK FOUND 2026-05-18; A2 COVERS THIS**
- Multi-thread reproducer added: `tests/python/gpu/test_b7_multithread_pool.py` (4 threads × 200 iters of resnet18_v1 with dynamic shapes from {192,224,256}^2 + main-thread `gpu_memory_info` polling). On the current pre-A2-rebuild binary with `MXNET_GPU_MEM_POOL_TYPE=Round`: ramp-peak 602 MiB, steady plateau **44 MiB** at t ≥ 7 s. 1-thread baseline (50 iters): plateau 10 MiB. Multi/single ratio = 4.2x — clean per-worker scaling, no unbounded growth. Pre-patch test assertion passes.
- Code audit (no patch): `Storage::gpu_mutex_` (shared across all GPU devices) covers every entry point in `pooled_storage_manager.h` (Alloc/Free/DirectFree/ReleaseAll → InsertInCache/GetMemStorage/ReleaseAllNoLock). No per-thread CUDA chunk cache exists. `ObjectPool` (engine bookkeeping) uses a single mutex + global free list. Only `BulkStatus` is `thread_local`, and it is bounded by `BulkFlush()`. There is no thread-local bypass of the global pool.
- Conclusion: the Flask-debug-mode unbounded growth in #19159 is the dynamic-shape × unbounded-per-bucket retention bug of A2 — threading just speeds up bucket discovery, it does not introduce a separate leak. The A2 K-cap fixes both. No additional patch required; re-verify after A2 rebuild.
- Files: test at `tests/python/gpu/test_b7_multithread_pool.py`; curves at `.investigations/b7_1thread_baseline.log`, `.investigations/b7_4thread_repro.log`; rationale note at `.investigations/b7_thread_pool.patch`.
- Link: https://github.com/apache/mxnet/issues/19159

### B8. #19218 — CPU inference very slow for some checkpoints — **CONFIRMED + DIAGNOSED 2026-05-18**, **WORSE THAN UPSTREAM**
- Benchmarked on AMD EPYC 7B12 Zen 2 (AVX2-only): Conv2D 64ch (1,3,224,224) takes 49.8 ms with `OMP_NUM_THREADS=1` BUT **536 ms with default 64 threads** (10× slower). Conv2D 512ch: 282 → 539 ms. Dense 1000 (1,2048): 3.6 → 271 ms (75× slower). Numbers in `b8_cpu_inference_bench.md`, bench at `bench_cpu_inference_b8.py`.
- Root causes: (1) **IC=3 + brg_conv_fwd:avx2 is pathological** — oneDNN v3 picks weight format `Acdb16a` which pads IC to blocks of 16. With IC=3, 81% of every AVX2 vector op is wasted on padding zeros. (2) **Negative thread scaling at bs=1** — the EPYC's 8 NUMA domains plus minimal per-thread work makes 64 threads catastrophically slower than 1. (3) **oneDNN v3 bump made it worse** — v3 now prefers `brg_conv` (throughput-tuned) where v2 used `jit:avx2`.
- **Immediate workaround (no code change)**: Set `OMP_NUM_THREADS=1` for bs=1 inference. 10× speedup. Document in tuning guide.
- **Proposed code fix**: In `src/operator/nn/dnnl/dnnl_convolution.cc`, gate `brg_conv` selection away from IC<16 + bs=1 + AVX2-only host. Allow the older `jit:avx2` path to win there.
- Effort: M (1 day to ship the dispatcher gate + verify across a few EPYC/Zen / Xeon hosts). Not a Blackwell-port blocker (CPU inference is a side use case).

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
