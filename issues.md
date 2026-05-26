# MXNet Port Issues

Canonical work tracker for the fork. Episodic build session detail belongs in
git log, not here. Open / partial / deferred / external items are on top;
the **RESOLVED / HISTORICAL** section below the divider holds everything
already closed (kept for traceability and to avoid re-doing closed audits).

**Branch:** `master`
**Latest merged PRs:** #33 (dropout/license/wheel `.3`), #34 (NCCL
single-machine kvstore fix + release-wheel timeout/thread update), #35
(ctypes ABI / floor-divide / release-provenance guard), #36 (skip audit +
WarpCTC optional-plugin contract cleanup)
**Latest tag on GitHub:** `v2.0.0+cu13.bw.20260526.3` (release published with
CUDA wheel)
**Release wheel:** <https://github.com/smolix/mxnet/releases/download/v2.0.0%2Bcu13.bw.20260526.3/mxnet-2.0.0%2Bcu13.bw.20260526.3-cp312-cp312-linux_x86_64.whl>
**Local wheel:** `dist/mxnet-2.0.0+cu13.bw.20260526.3-cp312-cp312-linux_x86_64.whl`
(522 MiB, release artifact and local copy; smoke-tested locally;
`rt.get_commit_hash()` reports
`cfe0fbd6881be1834e96dcd638aa95de5dc8856d`.)
Runtime deps are pulled from NVIDIA pip wheels; `libmxnet.so`
RUNPATH covers `$ORIGIN/lib`, venv-side `nvidia/cudnn/lib`,
`nvidia/nccl/lib`, `nvidia/cu13/lib`, and local CUDA install paths. The
wheel repairs cuDNN pip-wheel exact-version aliases on import before
`libmxnet.so` is loaded.
**Validation host:** 4× RTX 4090 (sm_89), CUDA 13.x driver line, cuDNN 9, NCCL
**macOS release tag:** `macos-arm64-slim-wheel-20260520`

Status labels:

- **Open** — known issue; no verified fix on the current branch.
- **In progress** — local work exists but is not committed and verified.
- **Partial** — first slice fixed; broader audit/coverage still pending.
- **Deferred** — cannot be verified here or out of scope for the current pass.
- **External** — owned by D2L, notebook infra, or another project.
- **Informational** — retained context, not a work item.

---

## Active Queue (Open / Partial / In-Progress)

| Priority | Tracker | Status | Issue | Next action |
|---|---|---|---|---|
| P0 | B4 / XOP18 | Deferred (architectural) | Real `_backward_sg_onednn_*` for QAT needs an NNVM/CachedOp framework refactor (multi-week scope). 20-test coverage in `test_quantized_backward.py` (14 passed, 6 xfailed) is the truthful production state. | Reopen with a concrete framework-refactor proposal. |
| P1 | XOP18 | Partial | Quantized self-attention subgraph forward contract pinned. Backward zero-grad behavior remains under B4. | Close alongside B4 framework refactor. |
| P2 | CN9 / L6 | Open (track upstream) | Bundled dmlc concurrent queue still assigns `-1` into a `uint32_t` sentinel under NVCC; oneDNN's vendored ITT assembly still lacks a non-executable-stack note. | Do not commit private submodule-local fixes; track as upstream/submodule policy. |
| P2 | C4 | Open | CUDA build matrix CI for Ada/Hopper/Blackwell + CUDA 12.x compatibility. | Validate `sm_89` here; leave CUDA 12.x and dedicated Blackwell to later runners. |
| P2 | L7 | In progress | Target load envelope: 48-64 runnable tasks; cap `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2-4`, `MKL_NUM_THREADS=1` for xdist lanes. | Keep one heavy CPU lane at a time. |
| P2 | O1, O4, O7 | Partial / Open | Linux wheel now has a working NVIDIA-pip runtime layout and published release artifact, but it is still not a self-contained CUDA/cuDNN/NCCL wheel (O1). The current `.20260526.3` release wheel reports `NCCL` disabled even though the source NCCL single-machine path was validated in PR #34. Release publication was manual; GitHub release automation is still absent (O4). No conda/system package story exists (O7). The build-metadata freshness guard landed in PR #35: cleanup-wheel packaging refreshes/rebuilds `build/`, and `release_provenance.py` rejects CMake metadata whose `MXNET_COMMIT_HASH` differs from `git HEAD`. | Strategic; defer expensive CUDA release automation until the release cadence is clearer. Decide whether release wheels should enable NCCL for single-machine multi-GPU collectives, then decide whether to keep the pip-runtime dependency model or produce a bundled runtime wheel. |
| P2 | T2, T3, T4, T5, T6, T11 | Open / Partial | T2 GluonNLP/Sockeye/AutoGluon (DGL out of scope); T3 multi-machine ps-lite rendezvous; T4 Python 3.13+; T5 NumPy 2.x ABI; T6 DLPack PyTorch/JAX/CUDA interop; T11 cross-platform lifecycle coverage. | Strategic; revisit when needed. |
| Strategic | O8, O9, O12 | Informational / Deferred | Apache MXNet archived 2023-11-17 — all fixes live in this fork (O8). Future oneDNN major releases will require porting (O9). ONNX Runtime 1.26 / opset 26 refresh is out of scope for current Linux/CUDA cleanup (O12). | — |
| Strategic | P1, P3, P4, P5 | Deferred / Hardware | cuBLASLt default-on / stride-aware / INT8 (P1); topk K-independence (P3); softmax / LayerNorm small-op kernel pipelines (P4); BF16 CPU validation on AVX-512-BF16 hardware (P5). | Defer; benchmark harness driven. |
| Deferred | GH7, GH8, GH9 | Deferred | Horovod KVStore barrier API (GH7); FlexiBLAS / THP / `parallel_for` grain (GH8); TensorRT upgrade (GH9). | Out of scope until specific drivers exist. |
| Remote | FP16 smoke | Remote | `tools/run_fp16_remote_smoke.sh` ready for a Zen 4+ host. | Run on target when available. |

### Cross-Platform Lifecycle Coverage TODO (T11)

- [ ] Linux x86 CPU: DataLoader, ThreadedEngine, KVStore, custom-op lifecycle tests with and without oneDNN.
- [ ] Linux CUDA: CUDA analogues for engine exception propagation, KVStore lifetime, custom-op forward/backward failure isolation.
- [ ] Linux CUDA: re-run lifecycle tests against `NaiveEngine`, `ThreadedEnginePooled`, `ThreadedEnginePerDevice`.
- [ ] Sanitizers: C++ engine/OpenMP/KVStore subset under TSAN; C++/Python lifecycle subset under ASAN/UBSAN.
- [ ] CI: add a quick job that builds `mxnet_unit_tests` and runs the focused lifecycle filters before any expensive full-suite job.

---

# RESOLVED / HISTORICAL

Everything below this divider is closed work or historical record. Rows are
kept for traceability and to avoid re-doing closed audits.

## Resolved In Current Fork

Compact summary; commit hashes, test paths, and detail-level reasoning live in
git log. Latest entries at the top.

| Date | Tracker | Resolution |
|---|---|---|
| 2026-05-26 | **V1 release-wheel acceptance sweep** | Closed for release wheel `v2.0.0+cu13.bw.20260526.3` under its actual feature set (CUDA/cuDNN/oneDNN/OpenCV enabled, NCCL disabled). Fresh-venv wheel acceptance passed: CPU broad unittest `13219 passed, 35 skipped, 1 xfailed`; CPU `test_operator.py` `1096 passed, 1 skipped`; CPU `test_random.py` `37 passed`; DNNL smoke/quantization/AMP/BF16/batch-dot/layer-norm/subgraph shards all passed; QAT backward shard reported the expected B4 state `14 passed, 6 xfailed`; GPU operator NumPy shard `11005 passed, 58 skipped`; GPU operator classic shard `1932 passed, 5 skipped`; GPU quantization wrapper `25 passed, 6 xfailed`; cuBLASLt, cuDNN fallback, TF32 deconv, reducer, fork-safety, AMP, and AMP-init shards all passed. D2L is currently treated as clean per the latest d2l-neu build status. The acceptance harness was also fixed locally so strict xfail summaries are not mislabeled as failures and empty shard selectors are caught. |
| 2026-05-26 | **FS8 stale skip audit** | Closed on merged `master`. The repaired profiler/NCCL/KVStore stale-skip batch passes `6 passed, 1 skipped` (the skip is expected because the current release wheel was built without NCCL); the old Gluon issue-11164 dynamic reshape/slice group is active and passes `23 passed`; higher-order-gradient plus GPU quantization wrapper validation passes `56 passed, 6 xfailed`; and the FS13 skip-reason lint passes `1 passed`. No current FS8 umbrella work remains: new broad skips should be rejected by `test_fs13_skip_reason_tracker_id.py`, and fresh failures should get concrete tracker rows instead of reopening FS8. |
| 2026-05-26 | **FS13 skip-reason lint / XOP26 optional plugin cleanup** | Closed in PR #36 commit `7ff3e7f6e`. The skip audit now covers runtime `pytest.skip()` / `pytest.xfail()` calls as well as decorator marks, and previously vague runtime skips cite concrete structural/capability reasons. WarpCTC plugin-side contracts are repaired: CMake now requires/resolves `ctc.h` + `libwarpctc` clearly, the sentinel label shape is correct, and async backward errors are waited before assertion. Real third-party `libwarpctc` validation is external/deferred. |
| 2026-05-26 | **FS12 numpy-shape SIGBUS** | Closed locally. Root cause was a Python/C ABI mismatch, not a thread_local storage bug: `MXIsNumpyShape(int* curr)` wrote four bytes into a one-byte `ctypes.c_bool` allocated by `mxnet.util.is_np_shape()`. `MXNDArrayIsDeferredCompute(int* curr)` had the same wrapper bug. Both wrappers now use `ctypes.c_int` and return `bool(curr.value)`. Regression coverage: `tests/python/unittest/test_fs12_np_shape_bus_error_repro.py` now checks the actual out-parameter object size passed through `ctypes.byref`. Validation: focused FS12 suite `4 passed, 1 skipped`; full `test_numpy_op.py -x` cleared the old ~21% SIGBUS point and ran to 99% before exposing the separate mixed `floor_divide` numeric bug below. |
| 2026-05-26 | **Mixed int/float16 `np.floor_divide` boundary** | Closed locally. The full `test_numpy_op.py` run reached `11080 passed, 1 skipped` and then failed on `floor_divide(int16, float16)`: the mixed half path computed `7 / 2.334` in float16, rounded the quotient to `3`, then floored it. The mixed `mshadow_op::mixed_floor_divide` and reverse path now compute the quotient in float before flooring and casting back to half. Regression coverage: `test_np_floor_divide_mixed_int_float16_boundary`; the original parametrized `test_np_standard_binary_funcs[...]` case passes. |
| 2026-05-26 | **Release-wheel build timeout/thread policy** | Closed in PR #34 commit `cfe0fbd68`. The release CUDA wheel job timeout and build-step timeout are now 90 minutes, and the build step uses `getconf _NPROCESSORS_ONLN` instead of a fixed small `-j` value. |
| 2026-05-26 | **XOP23 NCCL kvstore merge-root handling** | Closed in PR #34 commit `529677e7f`. Root cause was a mixed single-device then multi-GPU NCCL push path where the merge root changed devices after optimizer setup; pull then failed `root == buf.merged.ctx().dev_id`. `KVStoreNCCL` now copies local values into the merged buffer's context whenever the contexts differ, not only on CPU-to-GPU transitions. Validation: focused single-process repro, `tests/python/gpu/test_nccl.py`, `tests/python/gpu/test_nccl_singleproc.py` (`13 passed`), and `tests/python/gpu/test_nccl_multiproc.py` (`3 passed`) in a `USE_NCCL=ON` build. Multi-machine ps-lite rendezvous remains under T3. |
| 2026-05-26 | **D2L channels tolerance report** | External/D2L-side, not an MXNet regression. The 1x1-convolution notebook compared two mathematically equivalent float32 paths with an unseeded sample and a too-tight L1 `<1e-6` threshold; 93/1000 MXNet seeds exceeded it while max elementwise error stayed at normal float32 roundoff. The d2l-neu `channels.md` tolerance was relaxed to `<1e-5`; no MXNet code change is required. |
| 2026-05-25 | **XOP9 Dropout `out=` contract** | Closed. Legacy `mx.nd.Dropout(..., out=...)` already wrote through the caller-provided output on CPU/native-or-MKL and cuDNN paths, but `_npx.dropout(..., out=...)` silently ignored `out` in both Python wrappers and the packed C++ API. `_npx.dropout` now forwards the output handle through `Invoke()` and returns the caller's output object. Regression coverage: `tests/python/unittest/test_xop9_dropout_out.py` for CPU/native-or-MKL plus NumPy-extension `out=`, and `tests/python/gpu/test_xop9_dropout_out_gpu.py` for the cuDNN path. Existing RNN reserve-space coverage remains in `tests/python/unittest/test_xop9_rnn_dropout_req.py`. |
| 2026-05-25 | **Release `v2.0.0+cu13.bw.20260525.1`** | Published tag, GitHub release, wheel asset, and PR #30. Local wheel provenance was checked against commit `b2cb03ac2`; clean-wheel smoke confirmed `mxnet` imports, reports CUDA/cuDNN enabled, sees 4 GPUs, creates cuDNN alias symlinks in the NVIDIA pip-wheel layout, and runs a GPU convolution. |
| 2026-05-25 | **D2L cuDNN sublibrary loading** | Closed in commit `b2cb03ac2` (`Fix cuDNN runtime library layout`). Root cause was the NVIDIA cuDNN pip wheel exposing `libcudnn_*.so.9` while cuDNN dynamically requests exact filenames such as `libcudnn_ops.so.9.22.0`; the loader then mixed venv and system cuDNN sublibraries. `python/mxnet/_cuda_runtime.py` repairs exact-version aliases before `libmxnet.so` loads, and the wheel RUNPATH now includes venv-side `nvidia/cudnn/lib`, `nvidia/nccl/lib`, and `nvidia/cu13/lib`. Regression coverage: `tests/python/unittest/test_cuda_runtime.py`. D2L AlexNet, NiN, and VGG all passed after the fix. |
| 2026-05-25 | **D2L sentiment-analysis-rnn GPU leak** | Closed in commit `846e24d0e` (`Fix API output ownership leak`) with follow-up coverage in `cda356a97`. The confirmed leak was not a scheduler leak or cuDNN reserve-space leak; API wrappers that returned runtime ADTs copied freshly allocated `NDArray*` outputs into Python handles and dropped the original heap handles without deleting them. The Gluon LSTM path leaked one output tensor per forward call. `CreateADTFromOutputVector` now transfers ownership correctly, and the same ownership helper was applied across representative multi-output NumPy/NPX wrappers. Regression coverage: `tests/python/gpu/test_api_output_memory_leak.py`, `tests/python/unittest/test_api_output_ownership.py`, and the sharded unit-test runner added under `tests/run_unit_test_shards.sh.in`. |
| 2026-05-25 | **Similar leak-pattern audit** | Closed with the API-output fix. The pervasive pattern was ADT output ownership, not a repeated `Storage::Alloc`/scheduler-cache leak. The fixed callsites include RNN state outputs plus representative multi-output wrappers (`topk`, layer norm/group norm/batch norm/pooling/control-flow/linalg/histogram/unique/moments/unravel and related NPX paths). The earlier stateful-operator audit remains useful context: cuDNN dropout uses shared state, cuDNN conv/deconv and linalg paths use temp-space/workspace resources, oneDNN and TensorRT caches do not hold per-iteration output chunks. |
| 2026-05-25 | **AMP/cache and release-wheel stability** | Closed in commit `cda356a97` (`Fix AMP cache leak and stabilize test wheel build`). AMP weight cache cleanup is pinned by `tests/python/gpu/test_amp_weight_cache.py`. The same commit removed stale C API denormal-flush plumbing, added `src/common/denorms.h`, made profiler/NCCL tests more robust, and tightened build cleanup for wheel production. |
| 2026-05-25 | **Batched image augmentation / hotdog pipeline** | Closed in commit `9d33107cc` (`Fix batched image augmentation paths`). `random_flip_*`, `random_resized_crop`, crop, and related image paths now accept batched HWC/NHWC input rather than only a single HWC image. Regression coverage: `tests/python/unittest/test_numpy_gluon_data_vision.py` and `tests/python/gpu/test_perdevice_image_pipeline.py`, which runs subprocesses per GPU under `MXNET_ENGINE_TYPE=ThreadedEnginePerDevice` to catch crashes or GPU-memory growth in the image pipeline. |
| 2026-05-23 PM (session 3) | **D2L-Issue-3** narrowed further — engine-layer hypothesis | Bisection refined to STAGE=1 in `d2l-issues/repro_aug_chain.py`: ANY single CPU random image-aug op triggered the multi-GPU SEGV.  `NaiveEngine` and `MXNET_ENGINE_TYPE=ThreadedEngine` both passed; default `ThreadedEnginePerDevice` crashed.  Added `std::mutex` to `mshadow::Random<cpu>` and 10 image-aug FCompute sites (commit `74084a529`) as defense-in-depth. This row is historical: the later batched image-augmentation fix and PerDevice image-pipeline regression coverage supersede the workaround/root-cause hypothesis from this intermediate stage. |
| 2026-05-23 PM | **D2L Issues 5/6/7 + D5/D6/D7/D8 retired** | At this checkpoint all d2l items were closed on the MXNet side except `D2L-Issue-3`; that exception was later superseded by the 2026-05-25 image-pipeline row above. The d2l-side notebook fixes for `D2L-Issue-6` (`epoch_size=num_batches`) and `D2L-Issue-7` (`trainer.step(1)` vs PyTorch's no-rescale; or per-notebook `lr*N` bump) were handed off via `~/d2l-neu/MXNET-FIXES-ISSUES-6-AND-7.md`. Legacy d2l diagnostics D5 (DeadKernel batch, BERT NLI verified alive), D6 (use-gpu stale stamp), D7 (import-time GPU probing), D8 (artifact-quality signal) are external d2l notebook-infra ownership; reopen here only on a fresh MXNet-side crash repro. |
| 2026-05-23 PM | **D2L-Issue-6** lr-scheduler step semantics | Measurement (`tests/python/unittest/test_d2l_lr_scheduler_epoch_size.py`) confirmed MXNet `MultiFactorScheduler` / `CosineScheduler`, PyTorch `MultiStepLR`, and `optax.piecewise_constant_schedule` are semantically equivalent — all count caller-supplied steps; none has an intrinsic epoch concept. The d2l-mxnet 2× train-loss gap is from `Trainer.step()` advancing per-minibatch while the d2l notebook passes epoch-scale milestones. Added `epoch_size=` kwarg to `FactorScheduler` / `MultiFactorScheduler` / `PolyScheduler` / `CosineScheduler` so callers can pass epoch indices and get PyTorch-equivalent decay. 10 regression tests. **D2L-side notebook fix outstanding** — note delivered to `~/d2l-neu/MXNET-FIXES-ISSUES-6-AND-7.md`. Commit `399a081a4`. |
| 2026-05-23 PM | **D2L-Issue-7** FCN reduction gap | Measurement confirmed `gluon.loss.SoftmaxCrossEntropyLoss(axis=1)`, `F.cross_entropy`, and `optax.softmax_cross_entropy_with_integer_labels` produce **bit-identical** mean loss on FCN-shaped inputs. The real cause of the 3× higher d2l-mxnet train loss is `gluon.Trainer.step(N)` rescaling gradient by `1/N` while PyTorch's `optimizer.step()` does not — for `lr=0.001, batch_size=32`, MXNet's effective LR is 32× smaller. `Trainer.step()` docstring updated with explicit PyTorch comparison. 3 regression tests in `test_d2l_trainer_rescale_semantics.py`. **D2L-side notebook fix outstanding** — note delivered to `~/d2l-neu/MXNET-FIXES-ISSUES-6-AND-7.md` (Option A: `trainer.step(1)` global; Option B: bump FCN lr to `0.032`). Commit `399a081a4`. |
| 2026-05-23 PM | **D2L-Issue-5** storage banner suppression | Banner gating (already in place at `src/storage/storage.cc:201-209` behind `MXNET_LOG_STORAGE_INIT=1`) pinned by `test_d2l_storage_banner_suppression.py` (2 tests: silent by default, visible on opt-in). Commit `399a081a4`. |
| 2026-05-23 PM | **D2L-Issue-1** argmax regression (legacy nd.) | Extended `test_d2l_argmax_size_one_axis_regression.py` with 6 additional cases covering legacy `mx.nd.argmax` / `mx.nd.argmin` on size-1 GPU axes (the original report flagged checking whether legacy API shares the broken kernel; it does, and is also fixed by the `reduce_kernel_M1` shadow). 19 tests total, all green. Commit `399a081a4`. |
| 2026-05-23 PM | **D2L-Issue-3** fine-tuning DeadKernel triage | Empirically confirmed NOT in autotune — same flaky DeadKernel reproduces under `MXNET_CUDNN_AUTOTUNE_DEFAULT=0` and under `MXNET_CUDNN_FORCE_NO_HEURISTIC_PLANS=1`. Single-GPU runs intermittently pass; multi-GPU is the flaky path. Synthetic two-net repro on 4 GPUs runs clean. Added defense-in-depth `std::mutex` around `SelectPlan` in `src/operator/cudnn_ops.cc` (disable via `MXNET_CUDNN_AUTOTUNE_SERIALIZE=0`). Root cause of the d2l-notebook-only multi-GPU crash still under investigation — likely in `d2l.train_ch13` / KVStore interaction, not in MXNet itself. Commit `399a081a4`. |
| 2026-05-23 | **FS3 / FS5** | Sweeps green. Focused C++ sweep (`Engine.*:CAPI*.*:ThreadLocal.*:OMPBehaviour.*:EngineShutdown.*`) **21/21 passed**. GPU shard (8 files: batchnorm-running-stats, deconv-TF32, device-pushpull, fork-safe-dnnl, pool-dynamic-shape, reducer-regressions, d2l-bug-2, d2l-argmax) **51 passed** in 34.6s. Commit `175e3ed7b`. |
| 2026-05-23 | **D2L-Bug-3** (BERT NLI dead kernel) | Closed by the D2L-Bug-2 retry path. Solo BERT NLI on `.20260523.5` survived past the original 1095s death point and ran for 1400s before hitting the cell timeout (`CellTimeoutError`, kernel alive). The original symptom was an OOM `LOG(FATAL)` from an engine worker (which exits via `abort()` without traceback). Commit `0f52a3d18`. |
| 2026-05-23 | **D2L-Bug-2** (GPU OOM) | Bounded retry-with-backoff in `PooledStorageManager::Alloc` for `cudaErrorMemoryAllocation`. Default 4 retries × 50/100/200/400 ms (≤750 ms wall before FATAL), gated by `MXNET_GPU_MEM_POOL_OOM_{RETRIES,BACKOFF_MS}`. FATAL diagnostic now includes requested bytes / pool used / device free/total / retries. Coverage in `tests/python/gpu/test_d2l_bug_2_gpu_oom_retry.py` (8 tests, incl. 4 GB × 2-process concurrent smoke). Commit `1e5cb5019`. |
| 2026-05-23 | **XOP12 expansion** | Aux-state + inplace dimensions added to the contract harness. BatchNorm inference-mode aux-state preservation, BatchNorm training-mode aux-state update, Activation kWriteInplace-chain correctness. 51 contract checks + 3 new dimensions. Commit `0f52a3d18`. |
| 2026-05-23 | **XOP19** primary outputs | Audit-closed. Cached `dnnl::memory` + `set_data_handle` is functionally equivalent to `CreateDNNLMem + CommitOutput` for the supported reqs (`kWriteTo`, `kWriteInplace`) without the per-call alloc/copy overhead. `kNullOp` early-returns; `kAddTo` is `CHECK_NE`'d. Float-accumulate would be wrong for quantized output anyway. |
| 2026-05-23 | **XOP22 tail** | Two user-facing asserts in `mxnet.symbol.contrib` (foreach body format mismatch; while_loop cond shape) converted to `raise ValueError`. Internal-invariant asserts annotated as `# mxnet invariant, not user-facing`. Coverage in `test_xop22_symbol_contrib_user_validation.py` (4 tests). Commit `6a3ab8138`. |
| 2026-05-23 | **GH1 tail** | NDK r19c Dockerfile SHA256 pin via `sha256sum -c`; ARMv6 toolchain Dockerfile gets `--fail --proto '=https' --tlsv1.2` + `ARMV6_TOOLCHAIN_SHA256` build arg; `get-pip.py` wget enforces TLSv1.2 + retries + size check; `deploy.sh` defensive wget + cleanup-on-failure; `link_check.yml`, `os_x_mklbuild.yml`, `os_x_staticbuild.yml` bumped to `actions/checkout@v4` + `setup-python@v5`; `tools/diagnose.py` bounds DNS resolve via `socket.setdefaulttimeout` save/restore. Coverage in `test_gh1_dockerfile_and_workflow_hardening.py` (6 tests). Commit `6a3ab8138`. |
| 2026-05-23 | **GH2** | Audit-closed. Independent scan of `src/c_api/`, `include/mxnet/c_api.h`, `cpp-package/include/mxnet-cpp/` confirms no remaining concrete C/C++ API parity gaps. cpp-package stubs (`Reshape`, optimizer, `FeedForward`) are feature-work, not API parity. |
| 2026-05-23 | **GH6 tail** | `ImageListDataset` skips blank lines (was crashing on `int("")`); uses int keys for both file-loaded and list-loaded variants (was str on list path → two keyspaces); `MNIST` raises `ValueError` on empty/truncated label files. Coverage in `test_gh6_dataset_edge_cases.py` (3 tests). Commit `0f52a3d18`. |
| 2026-05-23 | **FS12** | Diagnostic anchor + ASAN runbook pinned in `test_fs12_np_shape_bus_error_repro.py`. Bisected to `test_np_sum[False-int64-int64-int64-False-1-shape1]` SIGBUS in `MXSetIsNumpyShape` thread_local at ~21% through file; passes in isolation. Root-cause still requires ASAN. Commit `0f52a3d18`. |
| 2026-05-23 | **XOP19** quantized FC primary output | `quantized_fully_connected.cu` rejects `kAddTo` before write (cuBLAS GEMM had hardcoded `beta=0`). Commit `032fcea0d` and earlier. |
| 2026-05-23 | **XOP19** selfatt gates | `SgDNNLSelfAttQKForward` + `DNNLSelfAttValAttForward` kNullOp early-return + kAddTo CHECK_NE; source-grep regression in `test_xop18_quantized_subgraph_req.py`. Commit `032fcea0d`. |
| 2026-05-22 | **XOP21** large-shape truncation | GroupNorm `N`, ROIAlign / PSROIPool count-launch counters, BilinearSampler / SpatialTransformer block counts, dnnl_dot bigDim products, multi_sum_sq chunks-per-tensor all INT_MAX-guarded. Tests in `test_xop21_large_shape_validation.py`. |
| 2026-05-22 | **XOP22** second wave | Gluon Parameter / KVStore base / BytePS asserts raise. Vocab/embedding/contrib.quantization/symbol.contrib `_flatten`/`_regroup`/`check_input` converted. `python -O` subprocess suite at 39 passed (was 19). |
| 2026-05-22 | **XOP23** engine invariants | Assert→CHECK conversions in `threaded_engine.cc`. `Engine.WriteAfterReadChainTermination`, `Engine.RapidVarAllocDelete`, `Engine.ShutdownRaceCreateUseDeleteCycle` in `threaded_engine_invariants_test.cc` (8 threads × 256 cycles). |
| 2026-05-22 | **CN2 / CN9** policy | `RelWithDebInfo` retains `-Wno-error=array-bounds` / `-Wno-error=stringop-overflow` for GCC ≥ 13 with documenting comment. CN9 dmlc queue u32 sentinel + oneDNN ITT executable-stack are documented as upstream/submodule policy in `BUILDING.md`. |
| 2026-05-22 | **CL-11 / GH1** legacy shell | `tools/dependencies/make_shared_dependencies.sh` carries a deprecation header pointing at `tools/build_cleanup_wheel.sh` as the current build path; SHA256 policy documented. |
| 2026-05-22 | **D2L-Bug-1** argmax GPU size-1 | `reduce_kernel_M1` in `src/operator/tensor/reduce_rtc.cc` used the outer flat-output index as the `index` referenced by `FUNC = AType(OP(...), index)`. Block-scope `const index_t index = 0;` shadow fix. 13 tests in `test_d2l_argmax_size_one_axis_regression.py`. Commit `cb2cdff7a`. |
| 2026-05-22 | **D2L-Bug-4** stale `mxnet.__version__` | `setup.py` writes `mxnet/_build_info.py` from resolved `MXNET_PACKAGE_VERSION`; `libinfo.py` imports from it. Commit `667498cb8`. |
| 2026-05-22 | **D2L-Bug-5** storage banner | Banner gated behind `MXNET_LOG_STORAGE_INIT=1`. Commit `667498cb8`. |
| 2026-05-22 | **BoxNMS** add segfault | `_backward_contrib_box_nms` declares `FResourceRequest{kTempSpace}` so `kAddTo` branch doesn't dereference past empty resource vector. Commit `667498cb8`. |
| 2026-05-22 | **XOP7** DNNL deconv weight-grad | `CreateDNNLWeightGrad` handles `kNullOp`; `DNNLDeconvBwd::WeightsGradMem`'s kWriteTo fast path nullptr-checks `CreateDNNLData` with helper fallback. Tests in `test_xop7_dnnl_deconv_req.py`. |
| 2026-05-22 | **XOP8** quantized range outputs | All quantized reshape/quantize/quantize_v2 + DNNL variants + dnnl_fc + `_contrib_quantized_embedding` route through `AssignQuantizedRangeOutput`. |
| 2026-05-22 | **XOP14** cuBLAS req→beta | cuDNN activation/pooling/softmax/LRN/bilinear/spatial-transformer wrappers + LRN backward honor kNullOp/kAddTo; 11 direct-cuBLAS sites audited; `quantized_fully_connected.cu` gated. Commit `a77daea17`. |
| 2026-05-22 | **XOP16** quantized embedding | Storage contract pinned (shape+dtype+range-output value) in `test_xop16_quantized_embedding_storage.py`. |
| 2026-05-22 | **XOP19** MaskedSoftmax | `DNNLMaskedSoftmaxForward` early-returns on kNullOp + `CHECK_NE`s kAddTo; pinned by `test_xop19_masked_softmax_req.py`. |
| 2026-05-22 | **FS3** image_random_crop | kAddTo rejection unconditional (was only on resize path). Commit `f5e1aa7dd`. |
| 2026-05-22 | **D2L-1 / D2L-3 / D2L-4** oneDNN 0-dim | `LOG(FATAL)` on 0-dim/view-path/GetDefaultFormat/GetPermutedFormat all fixed; 5+2 tests in `test_d2l_zero_dim_dnnl_regression.py`. Cleared 24 d2l notebook failures. |
| 2026-05-22 | **D2L-2** OpenCV bundling | `USE_OPENCV=ON` build bundles `libopencv_*.so.4.6.0` at `python/mxnet/lib/` with RUNPATH `$ORIGIN/lib`; `opencv-python>=4,<5` declared in setup metadata. 4 tests in `test_d2l_opencv_image_io_regression.py`. |
| 2026-05-22 | **D2L-5** batch_dot underflow | Defense-in-depth: 0-dim batch_dot raises early instead of falling into a 0-dim DNNL primitive. |
| 2026-05-22 | **GH4** trailing leak | `base.py` operator-module signature generation wraps both `get_module_file()` opens in the same try/finally; `tools/rec2idx.py` closes index file via try/finally. |
| 2026-05-22 | **P2** Blackwell wheel | sm_80/86/89/90/100/120 SASS + compute_120 PTX wheel built (`v2.0.0+cu13.bw.20260522.1`); guard test `test_cuda_arch_policy.py`. |

## Cross-Operator Audit (XOP1-XOP27) — Resolved

All cross-operator audit rows are closed. Detailed per-row resolutions remain
in the appendix below for traceability.

| ID | Area | Resolution |
|---|---|---|
| XOP1 | Norm state semantics | Native CUDA BN with `cudnn_off=True` + SyncBN update moving mean/var in forward, not backward. |
| XOP2 | Norm affine semantics | BN / SyncBN no longer mutate gamma for `fix_gamma=True`; native CUDA, cuDNN, SyncBN paths preserve `grad_req` semantics. |
| XOP3 | oneDNN LayerNorm | Publishes `std = sqrt(var + eps)`; backward converts back to oneDNN variance; gamma/beta `grad_req='add'` accumulation. |
| XOP4 | fp16 reductions | CUDA non-last-axis LayerNorm / GroupNorm keep fp16 forward mean/std/scratch in fp32 until final cast. |
| XOP5 | Shape inference | GroupNorm/InstanceNorm/SyncBN reject caller-provided bad gamma/beta shapes via checked shape assignment. |
| XOP6 | Hidden-output metadata | InstanceNorm hidden mean/var canonical names; CTCLoss `FListOutputNames` spelling fixed. |
| XOP7 | oneDNN output req/copyback | LayerNorm, deconv weight-grad fast path, softmax/log-softmax fwd+bwd, activation fwd+bwd, batch_norm fwd, quantized batch_norm, masked_softmax — audit complete via `CreateDNNLMem` / `CommitOutput` or explicit gates. |
| XOP8 | Quantized range outputs | All native + oneDNN quantized variants route range outputs through `AssignQuantizedRangeOutput`. |
| XOP9 | Stochastic / resource ops | Dropout backward + MKL/cuDNN paths; RNN dropout reserve-space contract (12 cases); direct `out=` CPU/native-or-MKL, NumPy-extension, and cuDNN forward paths. |
| XOP10 | Aux-state timing | `IdentityAttachKLSparseReg` updates moving averages in forward, uses output request for forward assignment. |
| XOP11 | Gluon affine flags | `LayerNorm(center=False/scale=False)` substitutes zero beta and one gamma. |
| XOP12 | Contract harness | 51 contract checks + aux-state (BN) + inplace (Activation) dimensions. |
| XOP13 | General output-request semantics | Resize / random-crop / crop-resize, TopK mask, native LRN backward, `_npi_average(returned=True)`, BoxNMS, `_npi_unique`, `sample_unique_zipfian`, empty-input NumPy reductions — all gated. |
| XOP14 | cuDNN/library beta mapping | cuDNN wrappers + 11 direct-cuBLAS sites + `quantized_fully_connected.cu` honor kNullOp/kAddTo. |
| XOP15 | Quantized primary-output req | `_contrib_quantized_elemwise_mul`, native + oneDNN quantize/quantize_v2/dequantize honor kNullOp/kAddTo for primary, shared helpers for ranges. |
| XOP16 | Quantized inference contracts | Quantized embedding storage contract (shape+dtype+range value) pinned. |
| XOP17 | Quantized metadata | Quantized RNN lists `statecell_output` when `state_outputs=True`. |
| XOP18 | Quantized subgraph forward | Forward contract anchor (registration + shape) for `_sg_onednn_selfatt_qk{,_split,_valatt}`. (Backward remains under B4.) |
| XOP19 | oneDNN descriptor/output handling | Reducer, softmax, batch-dot, deconv weight-grad, dnnl_reshape, `DNNLMaskedSoftmax`, BF16 fallback paths in selfatt + conv all converted or gated. Primary writes in quantized subgraphs use cached dst pattern (audit-closed). |
| XOP20 | Image dtype validation | `resize-inl.h` int32/int64 guard fixed; image resize preserves `kNullOp` and rejects `kAddTo`. |
| XOP21 | Large-tensor size truncation | LayerNorm, GroupNorm, ROIAlign, PSROIPool, BilinearSampler, SpatialTransformer, dnnl_dot, multi_sum_sq all INT_MAX-guarded. |
| XOP22 | Python validation via assert | AMP, KVStore, RecordIO, RTC, schedulers, Gluon Parameter, optimizers, contrib text/vocab/embedding/quantization, symbol/ndarray contrib, foreach/while_loop user input → `raise ValueError`. `python -O` suite at 39 passed. |
| XOP23 | Engine/runtime invariant | Assert→CHECK conversions + 3 race-stress tests. (NCCL stress open — see active queue.) |
| XOP24 | CUDA/NCCL unchecked status | `cudaMemcpyPeerAsync` checked; KVStore P2P / `KVStoreNCCL` NCCL/init calls checked; healthy-path P2P coverage. |
| XOP25 | Storage/profiler UB | `SET_GPU_PROFILER` null-checks profiler ptr; Linux CPU memory info multiplies by `mem_unit`. |
| XOP26 | Plugin/output contracts | WarpCTC kNullOp/kAddTo gate; OpenCV plugin buffer-bytes fix. The optional WarpCTC build path now requires/resolves `ctc.h` + `libwarpctc` clearly, and the sentinel test skips unless the plugin is built. Real third-party `libwarpctc` validation is external/deferred. |
| XOP27 | Visualization metadata | `plot_network()` forms shape/type keys from consumed output index. |

## Compiler Noise Triage — Resolved

All CN clusters closed except CN9 (submodule boundary, in active queue).

- **CN1** Build throughput — non-tuning builds skip `operator_tune.cc`.
- **CN2** Tuple/runtime allocation — overflow guards + `-Wno-error=array-bounds`/`-Wno-error=stringop-overflow` for GCC ≥ 13.
- **CN3** dmlc optional — fixed in dmlc-core commit `d610d79`.
- **CN4** Half/bfloat reductions — residual initialization independent of reducer setup.
- **CN5** CUDA unsigned guards — bincount/delete/nan-to-num/np_random with type-trait guards.
- **CN6** Sentinel conversions — `np_cross` / `np_matmul` value-initialize vectors.
- **CN7** Half param packing — fused optimizers use typed assignment instead of `memcpy`.
- **CN8** Local cleanup — CUDA resize/transformer unused vars, KVStore NCCL overload noise, pointwise fusion init, CTC moderngpu, mshadow packet alloc, einsum init, half max-pool init.

## D2L Diagnostics Import — Resolved

- **D1** Runtime deps (OpenCV/CUDA/cuDNN/NCCL) — release-wheel workflow
  configures `-DUSE_OPENCV=OFF` by default; primary
  metadata/runtime-bundling guard committed; `MXNET_SETUP_ENABLE_OPENCV_DEPS=0`
  opt-out; legacy `tools/pip/setup.py` CD path bundles `libopencv_*` when
  enabled. Current `v2.0.0+cu13.bw.20260525.1` wheel uses NVIDIA pip runtime
  dependencies and a RUNPATH that covers venv-side cuDNN/NCCL/CUDA library
  directories; fully self-contained runtime bundling remains tracked under
  O1/O4/O7 in the active queue.
- **D2** CUDA arch coverage — `sm_89` rebuild clears the no-kernel-image gate on Ada host. Release matrix coverage tracked under O2/C4.
- **D3** GPU scalar host sync — standalone scalar-to-host probes OK on rebuilt binary.
- **D4** Transformer native crash — oneDNN `batch_dot` reorders MXNet buffers into primitive-selected descriptors; `_sg_onednn_batch_dot` adds temp-space request. Coverage in `test_batch_dot_attention_regression.py`.

## Linux/CUDA Execution Queue — Resolved

- **L0** Build setup; **L1** Apple fixes on x86 (DataLoader, DLPack, quantization, oneDNN, NumPy drift); **L2** CUDA smoke; **L3** CUDA regression batch (cuDNN/TF32/cuBLASLt/fp16/linalg/deferred-compute/reducer/NumPy/sparse/KVStore/NCCL/histogram/Proposal); **L5** Tracker cleanup; **L8** Build freshness; **L9** Host GPU driver state (NVIDIA 580.126.20 → 580.159.03 after reboot).

## Linux/CUDA Validation Backlog — Resolved

- **C1** Histogram CUDA parity; **C2** Proposal/MultiProposal checked arithmetic; **C3** cuBLASLt GEMM/FC/dtype/strided + same-process threaded stress; **C5** cuDNN frontend autotune + no-plan fallback (`MXNET_CUDNN_FORCE_NO_HEURISTIC_PLANS=1`); **C6** cuDNN multi-stream against deterministic oracle; **C7** Targeted CUDA edge shard (split/reshape/reducer/kernel-error/zero-size/concat); **C8** TF32 deconvolution.

## Blackwell / CUDA Correctness Backlog — Resolved

- **B1** oneDNN INT8 subgraphs — `test_matmul_subgraph.py` 64 passed; **B2** Quantized Gluon — `test_quantization_dnnl.py` 26 passed; **B3** Mixed dtype quantization — AMP treats quantize_v2/dequantize as FP32 boundary; **B5** Mixed dtype matrix — `test_mixed_dtype_matrix.py` 5 passed.

## Full-Sweep Findings — Resolved

- **FS1** CPU unittest (1752 passed); **FS2** oneDNN Python (DNNL 30 / subgraphs 935 / quantization 26+26); **FS4** NCCL bandwidth (metric, not assertion); **FS6** GPU operator (3131 + 1905 NumPy + classic shards); **FS7** DNNL quantized conv+sum (4 focused checks); **FS9** Gluon BatchNorm crash (local-stats training routed to native CPU); **FS10** Higher-order gradients (test-source `variables=x` misuse); **FS11** Quantization GPU wrapper.

## GitHub Delta — Resolved

- **GH1** Security/tooling (zip extraction, ipynb2md subprocess, kill-mxnet, build_openmp, SHA256 pins, OpenCV downloader, ci.util.download_file, EC2 metadata, S3 key normalization, NDK + ARMv6 + get-pip + deploy.sh, 3 stale workflow versions, diagnose.py DNS).
- **GH2** C/C++ inference API parity (CppExecutor grad alignment, autograd-correct inference rerun, `MXSymbolGetInputSymbols` `ListInputs(kAll)`, `CutGraphInputs` + `MXSymbolGetChildren` NodeEntry dedup, `MXInvokeCachedOp` thread-safe).
- **GH3** Autograd/Gluon semantics (`autograd.grad` shape, attach_grad add, Gluon export dtype).
- **GH4** Resource hygiene (RecordIO, logger lifetime, base.py first/second-file leak, rec2idx).
- **GH5** Operator correctness (argsort fp16, linspace int floor, RReLU mask, NumPy min/max infinity, ModulatedDeformableConvolution slice, Gluon GroupNorm disabled affine).
- **GH6** Dataset/transforms (RecordFileDataset reader reset, ImageFolderDataset classes=, image-path separator, MultiBoxPrior coverage, RandomRotation skip-path labels, ImageListDataset blank-line + key-type, MNIST empty-label).

## Apple Silicon / macOS Wheel — Resolved

- **A6** Resource shutdown; **A7** macOS multiprocessing (`cpu_shared` probe + pickle-transport fallback); **A8** macOS slim optimized wheel built.
- **T12/T13/T14** C++ oneDNN pooling / convolution / Apple Silicon fallbacks.
- **T15-T17** Apple Silicon C++ + Python sweeps + macOS wheel smoke.

## Test Coverage and Integrations — Resolved

- **T1** GPU operator shards; **T7** Data/image tests; **T8** ONNX opset 18 reductions; **T9** Gluon model zoo 34/34; **T10** Custom C++ operators 9/9.

## Build, Release, And Operations — Resolved

- **O2** CUDA arch matrix (sm_80/86/89/90/120 + compute_120 PTX); **O3** Lightweight GitHub Actions workflow; **O5** CHANGELOG; **O6** README/docs; **O10** macOS arm64 slim wheel artifact.

## Follow-Up Branch Patches (R1-R48)

These resolved items from the follow-up branch are retained for traceability;
each has a commit hash and focused test references in git log. R1-R36 cover
DataLoader cleanup, image short-buffer reads, libjpeg-turbo RecordIO,
histogram validation, fused optimizers, quantized flatten, Proposal CPU
sizing, oneDNN quantized transpose, Azure option, oneDNN generated headers,
plugin lifetime, no-python DataLoader reset, OpenMP atomics, custom-op
exception isolation, KVStore async lambdas, threaded engine exception refs,
oneDNN C++ test helpers, lifecycle test coverage, engine shutdown, DLPack
CPU interop, NumPy API drift, AArch64 oneDNN fallbacks, quantized
transpose/requantize CPU fallback, C++ oneDNN pooling/convolution tests,
BatchNorm stochastic test, AArch64 float fallbacks, Gluon model-zoo
NumPy semantics, C++ stochastic shape helper, Apple Silicon smoke manifest,
C++ oneDNN AArch64 helpers, stale XPASS markers, binomial test dtype,
oneDNN JIT fallbacks, DataLoader `cpu_shared` fallback, ONNX-free wheel.
R37-R48 cover Linux oneDNN batch-dot, Linux BF16 tests, GPU pytest harness,
Linux DataLoader fork path, Linux wheel OpenCV guard, Linux OpenMP fork
test, optional extension artifacts, DNNL adaptive-pooling timeout, dmlc
optional lifetime, product reducer warning cleanup, two compiler warning
cleanup batches.

## Historical Highlights

Major Blackwell/CUDA port findings, kept as context.

| Area | Outcome |
|---|---|
| Adaptive average pooling | DNNL adaptive avg-pool backward disabled in favor of correct CPU fallback; 72/72 passed. |
| Quantize asym | oneDNN v3 attr-on-reorder issue fixed with `set_scales_mask(DNNL_ARG_DST, 0)`. |
| INT8 conv concat/relu/u8 | Runtime + property gates avoid the oneDNN small-channel u8 post-op bug. |
| Softrelu backward | `test_activation` softrelu resolved and unskipped. |
| Random seeding | CPU random generators are per logical CPU dev_id, matching GPU. |
| fp16 batch dot | `cublasGemmStridedBatchedEx` with fp32 accumulation. |
| CUDA linalg temp storage | Ephemeral GPU scratch synchronizes before free. |
| AMP subgraph | BF16-on-AVX2 fallback upcasts to fp32 for unsupported oneDNN primitives. |
| AMP RNN conversion | Repaired; original cuDNN 9 waitall failure did not reproduce. |
| NCCL single-process | 2-GPU NCCL KVStore tests passed; multi-process DDP-style is outside MXNet KVStore design. |
| Test-source bugs | Several stale numpy/op tests fixed or correctly skipped. |
| GPU profiler symbolic test | oneDNN v3 node-name expectation updated. |
| ONNX | Opset 18 reduction API change handled; broader ORT 1.26 / ONNX 1.21 refresh deferred under O12. |
| cuDNN 9.22 bump | Depthwise conv perf improved on Blackwell. |
| cuDNN frontend autotune | Env-gated v9 frontend autotune path added; default conservative. |
| sm_120 SASS | Confirmed `12.0+PTX` emits sm_120 SASS. |
| Sparse ops | CUDA 13 / Thrust 3 sparse benchmarks: no port regression. |
| fp16 tensor cores | Large dense/conv fp16 near PyTorch parity on Blackwell. |

## Tracker Reconciliation Notes

| Source | Current interpretation |
|---|---|
| Removed `FOLLOW_UPS.md` FU-1 | AVX2 int8 conv+relu tail gate implemented; covered by `test_fu1_int8_ic_lt8_gate.py`. |
| Removed `FOLLOW_UPS.md` FU-2 / `github-issues.md` G10 | Mixed fp16/int8 quantization tracked as B3. |
| Removed `FOLLOW_UPS.md` FU-4 | Fork-safe oneDNN/DataLoader behavior implemented; Linux validation under L1/T11. |
| Removed `FOLLOW_UPS.md` FU-6 | QAT subgraph backward bodies are not present on current `master`; **B4** is canonical. 2026-05-19 handover named local-only branch `fix/fu6-qat-subgraph-backward`, source `src/operator/subgraph/dnnl/dnnl_qat_backward.cc`, env gate `MXNET_QAT_SUBGRAPH_BACKWARD=1` — those refs/files are not present here. |
| Removed `FOLLOW_UPS.md` FU-8 | Legacy A6/A7 labels refer to old engine-deadlock audit IDs, not current rows; lifecycle work tracked by T11. |
| Removed `FOLLOW_UPS.md` FU-11 | Wide oneDNN stack/concat fallback implemented; covered by `test_fu11_large_stack_concat.py`. |
| Removed root CUDA tracker markdown | `nccl_status.md`, `cudnn_autotune_v9.md`, `fp16_perf_bench.md`, `sparse_thrust3_bench.md`, `storage_pool_bench.md`, `quantized_backward_status.md` were historical reports; active work canonical under FS4/T3, C5/C6, C4, sparse/storage rows, and B4. |
| `issues.md` T12-T14 | Resolved Apple/local oneDNN test-harness entries; remaining Linux x86 oneDNN validation under L1/T11. |
