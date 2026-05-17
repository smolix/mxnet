# MXNet Blackwell port — open issues

Snapshot: 2026-05-17 on branch `onednn-v3-port` at HEAD `f8b0c7125` (49 commits since start of port).

### Autonomous-session changes (after `f5934f094`)

| Commit | Change | Headline |
|---|---|---|
| TBD | fix apache#18751: BatchNorm running_mean/var swap (DNNL CPU path) | Move running-stats update from backward to forward in `dnnl_batch_norm.cc`; both CPU+GPU now show mean≈1/var≈0 after forward on all-ones input. Regression test at `tests/python/gpu/test_batchnorm_running_stats.py` (6/6 PASS). Note: residual CPU/GPU running_var discrepancy ~4e-4 is pre-existing (DNNL biased-N vs cuDNN unbiased-N-1 variance estimator). |
| TBD | fix apache#18584: batch_dot fp16 precision parity | `cublasHgemmStridedBatched` (fp16 accum) → `cublasGemmStridedBatchedEx(CUBLAS_COMPUTE_32F)` in mshadow; max rel-err 5.36→<0.005 |
| `cedeb2f9b` | re-enable 21 upstream-disabled tests | 22 unskips (incl. test_activation in `7e4231da5`) |
| `8a47e5a9a` | issues.md + cublaslt_scope.md | ~1130 LOC adoption scope documented |
| `7934d40d7` | `gluon.data.batchify` legacy NDArray handling | unblocks `test_gluon_data.py` (30/30) |
| `783cfa133` | TF32 default ON for FP32 conv on cuDNN 9 | **2.87×** speedup on sm_120 (3×3 256→256) |
| `a912fcdab` | bugreport_gluon_data.md | follow-up doc |
| `2feaef7c1` | issues.md post-rebuild results | verify table |
| `bd09b1a7b` | scope `reset_np()` to per-test fixture | clean test_image + test_gluon_data combined run |
| `f103c5491` | cuDNN 9.14 → 9.22 (local wheel install) | depthwise 3×3 256→256: **0.16 → 1.14 TFLOPS (~7×)** |
| `2b5b29085` | issues.md #1, #17 resolved | adaptive_pool 72/72, cuDNN bump |
| `7e4231da5` | unskip + #2,#3,#6 resolved | softrelu 4/4 seeds, quantize_gluon, quantize_asym |
| `ed26be03f` | issues.md #11 resolved | numpy test-source fixes confirmed |
| `f8b0c7125` | wheel platform-tag fix | `cp311-cp311-linux_x86_64` (was `py3-none-any`) |
| `e0eb106ea` | issues.md #18 (task #35): cuDNN frontend autotune MODE_A+B | `UseFrontendAutotune()` + `GetCombinedPlans()` in `cudnn_ops.cc`; env-gated (`MXNET_CUDNN_AUTOTUNE_FRONTEND`); parity with legacy on sm_120 canonical shape |

Verification on post-cuDNN-9.22 build:

| Surface | Pass | Fail | Skipped |
|---|---|---|---|
| `test_fc_subgraph.py` (full) | **387** | 0 | 16 |
| `test_conv_subgraph.py` (full) | **426** | **2** | — |
| `test_dnnl.py` (full) | **97** | 0 | — |
| `test_quantization.py` (single file) | **25** | 0 | — |
| Adaptive pool (`test_adaptive_pooling`) | **72** | 0 | 0 |
| Gluon RNN unskipped batch | **108** | 0 | 0 |
| Operator + sparse unskipped batch | **7** | 0 | 0 |
| numpy_op unskipped batch | **6** | 0 | 0 |
| Gluon data + image combined | **44** | 0 | 4 |
| test_activation × 4 seeds | **4** | 0 | 0 |
| **Total verified post-bump** | **1176** | **2** | **20** |

The 2 remaining failures are both the same int8 quantized concat bug (item #4 below) — a real DNNL v3 reorder-semantics issue, not order-dependent.

## Status at this snapshot

Final clean rebuild + 3-way parallel test sweep complete:

| Surface | Pass | Fail | Skipped |
|---|---|---|---|
| GPU0 — conv subgraph + FC subgraph + INT8 (`test_conv_subgraph.py` + `test_fc_subgraph.py` + `test_quantization_dnnl.py`) | **837** | 3 | 16 |
| GPU1 — `test_gluon_gpu.py` | **255** | 0 | 142 |
| CPU 8-way xdist — `test_gluon_rnn.py` + `test_gluon.py` | **232** | 0 | 138 |
| **Total** | **1324** | **3** | **296** |

99.8% pass rate. The 3 remaining failures and their root causes are documented below (items 4, 2, plus the test-order flake).

### Additional surfaces validated 2026-05-17 (post-TF32 + post-batchify rebuild)

| Surface | Pass | Fail | Skipped |
|---|---|---|---|
| FC subgraph (`test_fc_subgraph.py`, full file) | **387** | 0 | 16 |
| Unskipped operator + sparse (7 tests) | **7** | 0 | 0 |
| Unskipped numpy_op (6 tests) | **6** | 0 | 0 |
| Unskipped gluon_rnn (108 parametrized) | **108** | 0 | 0 |
| Unskipped batch 1 (6 tests: deconv2d_16c, CSVIter, aggregator x2, adamax, profile_create_domain_dept) | **6** | 0 | 0 |
| `test_gluon_data.py` (after `7934d40d7` batchify fix) | **30** | 0 | 0 |
| **Additional** | **544** | **0** | **16** |

The TF32 default change (commit `783cfa133`) was independently benchmarked: **2.87x** speedup on a 3×3 conv, 28×28, 256→256, batch 32 on sm_120 (14.46 → 41.48 TFLOPS).

libmxnet.so is 792 MB with all 5 SASS variants + PTX 120 fallback. Wheel is `mxnet-2.0.0+cu13.bw.20260517-py3-none-linux_x86_64.whl`, 1.9 GB self-contained.

## Motivation / context

This port exists because:

1. **Primary use case** — execute existing MXNet notebooks on Blackwell (RTX PRO 4000 / sm_120) hardware with CUDA 13 + cuDNN 9 + oneDNN v3 + NCCL 2.28. Apache MXNet upstream was archived on 2023-11-17 with no Blackwell or CUDA 13 support; this fork (`smolix/mxnet`) is the only place that work will land.
2. **Secondary** — provide a working Blackwell wheel for the residual MXNet user community. There are still teams running legacy MXNet pipelines (research code, frozen production stacks, niche operators like `_contrib_quantize_*`) who cannot easily migrate to PyTorch/JAX. A correct Blackwell port lets those pipelines run on current hardware without a full rewrite.

The bar therefore is "real notebooks run correctly and respectably fast on Blackwell", not "feature-parity with PyTorch". Performance gaps are acceptable when correctness is in place; correctness gaps are not.

This file lists everything still open at this snapshot. Items are grouped by severity. Each has a one-line "what" and a hint at "what to do".

---

## CORRECTNESS gaps (block "the port is done")

1. ~~**`adaptive_avg_pool` backward correctness** (task #33).~~ **RESOLVED 2026-05-17** via commit `1d2198862` (force CPU-reference fallback in `SupportDNNLAveragePooling`) — the CPU kernel correctly normalises by pool-window overlap, so 36→0 failures. Re-verified post-cuDNN-9.22 rebuild: `test_adaptive_pooling` is 72/72 PASS across all shape×stype combos. A latent DNNL backward bug remains in `dnnl_pooling.cc::GetPoolingBwd` (in_data aliased to out_grad for adaptive) but is unreachable while SupportDNNL returns false; a partial attempt to re-enable DNNL adaptive pooling failed smoke and was reverted.

2. ~~**`test_quantize_gluon_with_forward`** (gluon `quantize_net` of resnet18).~~ **RESOLVED 2026-05-17** — confirmed PASSING on post-cuDNN-9.22 rebuild (`HEAD=f103c5491`). The combination of the FC saturation fix + B2 NaN guard (`f5934f094`) + TF32 selection on cuDNN 9 (`783cfa133`) closed this together. 1/1 PASS in 0.25s.

3. ~~**`_contrib_quantize_asym`** v3 attr-on-reorder bug.~~ **RESOLVED** via agent #45 commit — `src/operator/quantization/dnnl/dnnl_quantize_asym-inl.h:141-155` now uses `set_scales_mask(DNNL_ARG_DST, 0)` instead of the removed v2 `set_rnn_data_qparams`. (Asymmetric quantization is not exercised by any in-tree test, so verification remains by code inspection only.)

4. **`test_pos_single_concat_pos_neg[int8/auto-data_shape1]`** — re-tested 2026-05-17 against HEAD `cedeb2f9b`: failure is **NOT** order-dependent. Test fails in isolation with `MXNET_TEST_SEED=11`. Failure mode: max int8 quantization error 5.0 on output (atol=0.12, so 40× exceeded) where most error locations have `a=0, b≈1.2` — entire output channels are being zeroed. data_shape1 is `(4, 3, 24, 24)`; channels 3-6 of output (the `relu(conv0(x))` half of the concat) are corrupted. Suspect: oneDNN v3 reorder uint8→int8 with `set_scales_mask(DNNL_ARG_SRC, 0)` in `src/operator/quantization/dnnl/dnnl_quantized_concat.cc:88-99` — scale math looks right (`out_scale / i_scale`) but observed behavior is zeros. Either reorder primitive caches a stale desc, the rescaled memory is being dropped before Submit(), or v3 reorder semantics for uint8→int8 differ from what the code assumes. Needs DNNL `dnnl_verbose=2` trace to pin down.

5. **Backward through quantized ops** — untested. If anyone fine-tunes a quantized model this likely blows up. Forward inference is solid; backward through `_sg_onednn_fully_connected`, `_sg_onednn_conv` is unvalidated.

6. ~~**`test_activation` softrelu backward** (#13915).~~ **RESOLVED 2026-05-17**

46. ~~**`batch_dot` fp16 precision divergence from `dot`** (apache#18584).~~ **RESOLVED** — mshadow `BLASEngine<gpu, half_t>::batched_gemm` was calling `cublasHgemmStridedBatched` (true fp16 accumulators) while the 2-D `dot` path calls `cublasSgemmEx` (fp32 accumulators, pseudo-fp16). On CUDA 13 / Blackwell the divergence is visible with max relative error >500% on random inputs. Fix: replaced `cublasHgemmStridedBatched` with `cublasGemmStridedBatchedEx(CUBLAS_COMPUTE_32F)` in `3rdparty/mshadow/mshadow/dot_engine-inl.h`. Regression test at `tests/python/gpu/test_batch_dot_fp16_parity.py`. Commit SHA: TBD — B2 SoftReLU/LogSigmoid α=±1 fix (`8f6cc19ad`) + cuDNN 9.22 build, verified 4/4 PASS across 4 different `MXNET_TEST_SEED` values (11, 17, 23, 31). The upstream "intermittent flake" no longer reproduces. Unskip committed.

---

## FUNCTIONALITY gaps (operations broken or not validated)

7. **bf16 path on this host** — Zen 2 EPYC 7B12 lacks AVX-512 BF16. oneDNN falls back to fp32 emulation. Not fixable in software. Test on Intel SPR / Zen 4 / Granite Rapids.

8. ~~**AMP (Automatic Mixed Precision) subgraph** — 6 `test_amp_subgraph.py` failures with `inner_product` primitive creation errors.~~ **RESOLVED 2026-05-17** via bf16→fp32 fallback in `dnnl_base-inl.h` + `dnnl_fully_connected.cc` + `dnnl_conv.cc` + `dnnl_transformer.cc`. `DNNLISASupportsLowpFloat()` detects AVX2 hosts at runtime and upcasts bf16 operands to fp32 for the DNNL primitive call, then reorders the fp32 result back to bf16. All 6 AMP subgraph tests now PASS (25s); FC subgraph 387/0/16 and test_dnnl.py 97/0 unchanged. See `amp_subgraph_fix.md` for full writeup.

9. **AMP-RNN conversion** — `test_amp_conversion_rnn` fails with "Error during waitall()" tracked at upstream #18099. Untested whether our cuDNN-9 v8 RNN path interacts with AMP's autocast hooks at all.

10. **NCCL multi-process not validated** — 1-process / 2-GPU `kv.create('nccl')` push/pull works. Standard distributed training topology (1 process per GPU with `dist.init_process_group`-style init) is untested. May need `MXNET_KVSTORE_USETREE` or related tunings to scale.

11. ~~**Test-source bugs blocking 80+ test invocations**~~ **RESOLVED 2026-05-17** via agent #46 (commit `fa51581cb` numpy test fixes) plus the 21-test unskip in `cedeb2f9b`. Re-verified on cuDNN-9.22 build:
   - `test_deformable_psroipooling` → 1 PASS
   - `test_np_empty_like` → 1 SKIP (separate bug about int8/uint8/bool not supported, documented inline at the skip)
   - `test_convolution_options` was never in this fork (no upstream test by that name; bug referenced has different path)

12. **`test_gpu_memory_profiler_symbolic`** (#18564) — profiler probe for `tensordot:dot_backward` attr_name doesn't find a match. The dot kernel was renamed under oneDNN v3 dispatch. Either update the test or restore the tag name in the kernel.

13. **`test_convolution_multiple_streams`** — `rtol=0.01/atol=0.01` mismatch on cuDNN-9 conv across NaiveEngine / ThreadedEngine / ThreadedEnginePerDevice on Blackwell. Real numerics drift. Either loosen tolerance to 0.05/0.05, force deterministic algorithms (`MXNET_CUDNN_AUTOTUNE_DEFAULT=0`), or leave skipped.

14. **ONNX export/import** — `tests/python/onnx/test_models.py` and `test_operators.py` both error at collect time. Likely the ONNX path was never updated for MXNet 2.0 numpy ops; depends how important ONNX interop is.

15. **Gluon pretrained model loading** — `test_gluon_model_zoo.py` partially fails (pretrained checkpoint download + symbol serialisation). If anyone uses the model zoo, this needs validation.

16. **Custom C++ operators** — `test_custom_op_fork` audit was green but the broader custom-op infrastructure with CUDA 13 / Thrust 3 hasn't been exercised.

---

## PERFORMANCE gaps (works but suboptimal)

17. ~~**cuDNN 9.x sm_120 heuristic gap** (task #34).~~ **RESOLVED 2026-05-17** via commit `f103c5491` — bumped cuDNN 9.14 → 9.22 (locally bundled under `cudnn_local/`, system untouched). Headline impact: depthwise 3×3 256→256 went from 0.16 → 1.14 TFLOPS (**~7×**). Other shapes within noise (e.g. 3×3 28×28 256→256 bs32: 41.07 → 41.52 TFLOPS, same arena). Regression smoke clean: `test_fc_subgraph.py` 387/0/16 unchanged, `test_dnnl.py` 97/0 (including 72/72 `test_adaptive_pooling`).

18. ~~**`cudnnFindAlgorithm` / autotune not ported to v9** (task #35).~~ **RESOLVED 2026-05-17** via commit `e0eb106ea` — added `UseFrontendAutotune()` (env `MXNET_CUDNN_AUTOTUNE_FRONTEND`, default off) and `GetCombinedPlans()` which unions `CUDNN_HEUR_MODE_A + MODE_B` engine configs (deduped by plan string) and feeds the merged candidate list to `FindTopPlans`. On sm_120 / cuDNN 9.22, both modes select the same engine for the canonical 256→256 3×3 bs32 shape (parity with legacy). The combined path exposes a larger candidate set (20–23 plans vs fewer from a single mode) and is valuable for non-standard shapes where Mode A alone misses the fastest kernel. See `cudnn_autotune_v9.md` for details.

19. **`cuBLASLt` not adopted**. Single-precision GEMM goes through legacy cuBLAS; Blackwell's faster algorithms only surface via `cublasLtMatmulAlgoGetHeuristic`. Major hidden FLOPS on any matmul-heavy network.

20. **No fatbin SASS for sm_120**. We only generate compute_120 (PTX). First kernel launch per process pays a JIT compile stall (PTX→SASS) that can be 100s of ms for kernels with template specializations. Fix: add `-gencode arch=compute_120,code=sm_120` (or `code=[compute_120,sm_120]`) to nvcc flags.

21. **Sparse ops post-CCCL-3** — Thrust 3 unification changed default execution policies. `dense_to_csr`, `csr_to_dense`, sparse `topk` performance is uncharacterized. Could be the same; could be 2× worse. Benchmark.

22. **Storage manager defaults**. Test output shows "Pooled (Naive) StorageManager" on every test. The `MXNET_GPU_MEM_POOL_TYPE=Round` variant fragments less on 24GB cards. Worth profiling whether it's a free win.

23. **TF32 default selection**. cuDNN 9 changed when TF32 is enabled. Audit `cudnnSetTensorMathType` call sites; FP32 conv that opts in to TF32 gets ~2× free.

24. **fp16 tensor cores**. With F16C now enabled on CPU side, the GPU fp16 path may be underexercised. Run AMP / FP16 benchmarks against PyTorch on the same model to confirm parity.

---

## TEST COVERAGE gaps

25. **Full `test_operator_gpu.py` sweep never run** on this build. ~28,408 tests total in `tests/python/`; this session validated DNNL subgraphs (815/3 pass), RNN (357/0), a slice of INT8 (~25 tests). Planned post-rebuild.

26. **Distributed training** — kvstore parameter server, NCCL collective, Horovod compatibility — none exercised beyond a 2-GPU smoke test.

27. **`test_gluon_data*.py`** files segfault (`SIGSEGV rc=134`) — `test_gluon_data.py`, `test_contrib_gluon_data_vision.py`, `test_image.py` all had crashes in earlier sessions. Data pipeline (DataLoader, transforms, image augmentation) needs investigation.

28. **Mixed dtype matrices** — fp16/fp32 (AMP), int8/fp32 (quantize), int8/fp16 are separate paths; each needs its own pass.

29. **Out-of-tree operators** — Gluon NLP, Sockeye, AutoGluon, DGL all build on MXNet. Compatibility untested.

---

## BUILD / RELEASE / OPS gaps

30. **Wheel doesn't bundle CUDA/cuDNN/NCCL runtimes**. `mxnet-2.0.0+cu13.bw.20260517-py3-none-linux_x86_64.whl` is 158MB but the user still needs to `apt install libcudnn9 libnccl2 cuda-13` separately. `auditwheel repair` or a manylinux workflow makes it self-contained.

31. **Multi-arch fatbin**. Currently only sm_120. Add sm_80 (Ampere A100), sm_86 (RTX 30 / Ada), sm_89 (RTX 40), sm_90 (H100) to make the wheel useful on more GPUs. Single-arch sm_120 means RTX 30 / 40 users get nothing.

32. **No CI on smolix/mxnet**. Every test in this session was hand-run. Even a minimal GitHub Actions job that builds + runs the DNNL subgraph subset would catch regressions.

33. **GitHub Release** (task #28) — wheel built but not published. Blocked on YubiKey tap for push.

34. **Release notes / changelog**. Nothing documents the v2→v3 changes, the cuDNN port, the quantization fixes, the breaking-change matrix.

35. **README** still reads "Apache MXNet" with stale dependency lists.

36. **Documentation** — no mention of Blackwell, CUDA 13, cuDNN 9 requirements; users will be surprised.

37. **Conda / system package** — only wheel exists; conda-forge MXNet is way behind.

---

## CODE-QUALITY (from bugreport.md)

38. 8 critical bugs (B1+B2 already fixed in `8f6cc19ad`; B3-B8 currently with agent #42).
39. 12 fragile patterns (F1-F12 currently with agent #42).
40. 11 code smells / TODOs (S1-S11 currently with agent #42).

When agent #42 lands its commit, items 38-40 will be resolved or explicitly marked deferred.

---

## STRATEGIC

41. **MXNet upstream archived 2023-11-17**. No upstream fixes are possible. Every new bug found here has to be fixed in `smolix/mxnet`. Long-term either keep up with CUDA / cuDNN cadence ourselves, or treat MXNet as a frozen runtime that only changes when the underlying libraries break it.

42. **No alternative to oneDNN v3 dependence** on the CPU side. Future Intel cadence (v4 etc.) will force more porting.

43. **Python 3.13+ untested**. Current build is 3.11. The `_npi_*` C extensions may not be 3.13-ABI-compatible.

44. **NumPy 2.x ABI**. `setup.py` pins `numpy >= 1.17`; some test failures already trace to numpy 2.x API drift. The `python/mxnet/` C extension probably needs adjustment for numpy 2's stable ABI.

45. **DLPack version**. The 3rdparty/dlpack vendored copy is older; PyTorch/JAX have moved on. Interop may be broken.

---

## Priority recommendation if triaging

**Before a "preview release" wheel ships publicly:**
- #1 adaptive_pool
- #2 quantize_gluon recheck
- #3 quantize_asym v3 fix
- #11 test-source numpy fixes (free 80+ tests)
- #21 full `test_operator_gpu.py` sweep
- #30 wheel bundling
- #31 multi-arch fatbin
- #33 + #34 + #35 publishing + changelog

**Before "1.0 of the fork":**
- All correctness items (#1-6)
- #25 full test sweep green
- #14 ONNX
- #17 cuDNN bump for perf
- #19 cuBLASLt
- #32 CI

**Lower priority (lift later):**
- #21 sparse benchmarking
- #18 autotune frontend (after #17 makes a noticeable diff)
- #44 numpy 2 / #43 python 3.13 (when users complain)

**Out-of-scope here:**
- #7 bf16 (hardware)
- #41 strategic / archived-upstream concerns (these are decisions, not work items)
