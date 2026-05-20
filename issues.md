# MXNet Blackwell port — open issues

Snapshot: 2026-05-17 on branch `onednn-v3-port` at HEAD `f8b0c7125` (49 commits since start of port).

### Autonomous-session changes (after `f5934f094`)

| Commit | Change | Headline |
|---|---|---|
| `a47ce39d9` | fix apache#18751: BatchNorm running_mean/var updated in forward, not backward | Move running-stats update from backward to forward in `dnnl_batch_norm.cc`; both CPU+GPU now show mean≈1/var≈0 after forward on all-ones input. Regression test at `tests/python/gpu/test_batchnorm_running_stats.py` (6/6 PASS). Note: residual CPU/GPU running_var discrepancy ~4e-4 is pre-existing (DNNL biased-N vs cuDNN unbiased-N-1 variance estimator). |
| `08cb44d1d` | fix apache#18584: batch_dot fp16 precision parity | `cublasHgemmStridedBatched` (fp16 accum) → `cublasGemmStridedBatchedEx(CUBLAS_COMPUTE_32F)` in mshadow; max rel-err >500%→0.00e+00 (exact bit-match on (8,64,64,64)); 6/6 regression tests PASS |
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
| `f5e7c063c` | fix apache#19019: AMP weight cast cache | `python/mxnet/amp/amp.py` — `_cast_symbol_NDArray` now caches the fp16 result keyed by `(id(src), src_dtype, dst_dtype)`; repeated forward passes through a shared layer reuse one fp16 buffer instead of allocating a new one each step. Peak GPU memory with a 200-iteration shared-cell loop goes from `200 × weight_size` → `1 × weight_size`. `clear_weight_cache()` public API for explicit invalidation. Regression test: `tests/python/gpu/test_amp_weight_cache.py` (7/7 PASS); smoke: `test_amp_subgraph.py` 6/6 unchanged. |
| `c2df8dd44` | fix apache#18865: per-CPU-dev_id random generators | `cpu_rand_` and `cpu_parallel_rand_` changed from `unique_ptr` (shared across ALL CPU dev_ids) to `common::LazyAllocArray<>` indexed by `ctx.dev_id`, mirroring the existing per-dev_id GPU design. Seeding `cpu(N)` no longer affects `cpu(M)` for M≠N. Six regression tests at `tests/python/unittest/test_random_seed_order.py` (6/6 PASS in 0.96s). |
| TBD | A15 fix apache#14264: `nd.reshape` silently truncates | re-enable size-equality CHECK in `ReshapeShape` (guarded by `shape_is_known`) + 0/0 guard in `InferReshapeShape` -1 inference path (was process-killing SIGFPE on `infer_shape` with 0-dim holdouts). Side fixes: `numpy_op_signature.py` + `numpy_dispatch_protocol.py` `sometrue` removal (import-time crash after rebuild). See `a15_reshape_followup.md`. FC subgraph baseline 387/0/16 preserved. |
| TBD | verify apache#16686: grad_req='add' numerical consistency (VERIFIED FIXED) | No C++ change needed. Regression test `tests/python/unittest/test_grad_req_add_consistency.py` (6/6 PASS) confirms: (a) switching to `grad_req='add'` correctly zeroes the buffer via `_init_grad()`; (b) Dense, Embedding, and Embed+Dense (BERT-like) accumulated gradients match manual accumulation bitwise (Dense) or within float32 scatter-add rounding ≤1.19e-07 (Embedding). The original divergence was likely a user-side bug (not calling `zero_grad()` before each accumulation window), not a framework bug. |
| TBD | fix apache#11163 (A13) / partial-fix apache#18090 (A7) / instrument apache#19994 (A6): engine shutdown + deadlock diagnostic | `MXNotifyShutdown()` now calls `Engine::Get()->Stop()` after `WaitForAll()` — worker threads fully joined before interpreter teardown, so static-dtor `cv.notify_all()` is a no-op. `MXNET_ENGINE_DIAG=1` watchdog added to `WaitForVar`/`WaitForAll` (30 s default, `MXNET_ENGINE_DIAG_TIMEOUT_S` override). Tests: `tests/python/unittest/test_engine_shutdown.py` 12/12 PASS. See `engine_deadlock_audit.md`. |

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

51. ~~**oneDNN concat/stack fails at > 512 sources**~~ — **RESOLVED 2026-05-19** via PR #23. oneDNN v3.11's concat primitive reported `bad number of inputs (expected N+1 got N)` when dispatched with more than 512 sources (deterministic at 513). Triggered by `gluon.data.DataLoader`'s `Stack` batchify whenever `batch_size > 512` — surfaced as the 7-notebook d2l-neu RNN/embedding cluster (gru/lstm/deep-rnn/rnn-concise/rnn-scratch/seqrec/minibatch-sgd) failing with `MXNetError: could not execute a primitive`. Fix gates `SupportDNNLStack`/`SupportDNNLConcat` at 256 inputs and falls back to the generic CPU path for wide stacks. Test: `tests/python/dnnl/test_fu11_large_stack_concat.py` (19/19 PASS). End-to-end: d2l-neu `chapter_recurrent-modern/gru.ipynb` now trains 1 epoch in 8.7 s on the fix branch (was crashing in < 20 s pre-fix). Underlying oneDNN PD cache defect still upstream; the in-tree gate is the long-term fix until oneDNN resolves it.

49. **`test_self_attention[*]` + `test_batch_dot[*]` + `test_self_attention_negative` int8 quantization** — discovered 2026-05-18; scope confirmed **PERVASIVE** across all int8 matmul/batch-dot subgraph paths. 32/32 `test_self_attention` parametrizations fail (both split=True and split=False, all 4 num_heads × 4 units × 2 seq_length × 2 batch_size combos). 16/16 `test_batch_dot` fail. `test_self_attention_negative` segfaults pytest (same RC=139 as the earlier sweep crash on `[True-8-256-384-32]`). Failure character: **massive numerical blow-up** (output ~2e+23 vs expected ~0.14, `Error 2.9e+24`). Points to a scale/zero-point overflow in the post-quantization matmul path. Same oneDNN v3 INT8 scale-convention root cause as #4 — once #4's f32-intermediate fallback (in `dnnl_quantized_concat.cc`) is verified, need to audit the matmul/batch-dot subgraph dispatchers (`src/operator/subgraph/dnnl/dnnl_matmul.cc` if it exists, or `dnnl_fully_connected.cc`'s quantized branch) for the same SRC-vs-DST scale-mask issue and apply the same f32-intermediate workaround there.



1. ~~**`adaptive_avg_pool` backward correctness** (task #33).~~ **RESOLVED 2026-05-17** via commit `1d2198862` (force CPU-reference fallback in `SupportDNNLAveragePooling`) — the CPU kernel correctly normalises by pool-window overlap, so 36→0 failures. Re-verified post-cuDNN-9.22 rebuild: `test_adaptive_pooling` is 72/72 PASS across all shape×stype combos. A latent DNNL backward bug remains in `dnnl_pooling.cc::GetPoolingBwd` (in_data aliased to out_grad for adaptive) but is unreachable while SupportDNNL returns false; a partial attempt to re-enable DNNL adaptive pooling failed smoke and was reverted.

2. **`test_quantize_gluon_with_forward`** (gluon `quantize_net` of resnet18) — **REGRESSED under DNNL subgraph backend 2026-05-18**. Was PASSING on 2026-05-17 in the non-DNNL path. Under `ENABLE_ONEDNN_QUANTIZATION_TEST=1` in `tests/python/dnnl/test_quantization_dnnl.py`, the test now **segfaults** in `_build_cache` → `quantize_net` (crash in MXNet's `CachedOp` constructor, `_ffi/_ctypes/function.py:138`). 18 prior tests in the same file pass, then crash, then 6 subsequent tests unrun. Likely same root cause as #4/#49 (DNNL v3 int8 scale convention) — applied to a different code path. After #4's `dnnl_quantized_concat.cc` fix verifies, audit `dnnl_quantized_fully_connected.cc` for the same `set_scales_mask` SRC/DST issue. Workaround: run quantize on the non-DNNL path.

50. **Mixed dtype int8/fp16 is structurally absent**. Identified 2026-05-18 in `#28` mixed-dtype audit. Three independent dispatch sites reject fp16 input/output for quantize: (a) `src/operator/quantization/quantize_v2-inl.h:157,160` accepts only kFloat32/kBfloat16/kUint8/kInt8 — fp16 omitted; (b) `src/operator/quantization/dequantize-inl.h:42` has `add_enum("float32", kFloat32)` as the ONLY allowed output dtype — no fp16/bf16; (c) `src/operator/quantization/dnnl/dnnl_quantize_v2-inl.h:98` reads `dptr<float>()` unconditionally. Not a port regression — never existed upstream. Action: either (i) widen all three sites to accept fp16 (CPU oneDNN supports f16→s8 reorder), or (ii) document AMP fp16 + quantize as unsupported and require `amp_cast(fp32)` upstream of quantize_v2. Side bug: `mx.nd.contrib.quantize_v2(bf16_data, ctx=gpu)` on no-calib path abort()s with `dmlc::Error "TBlob.get_with_shape: data type do not match"` instead of returning a clean type-check error (B4 in audit). Low severity (gated upstream by quantize_v2-inl.h:160 check elsewhere).

3. ~~**`_contrib_quantize_asym`** v3 attr-on-reorder bug.~~ **RESOLVED** via agent #45 commit — `src/operator/quantization/dnnl/dnnl_quantize_asym-inl.h:141-155` now uses `set_scales_mask(DNNL_ARG_DST, 0)` instead of the removed v2 `set_rnn_data_qparams`. (Asymmetric quantization is not exercised by any in-tree test, so verification remains by code inspection only.)

4. ~~**`test_pos_single_concat_pos_neg[int8/auto-data_shape1]`**~~ — **RESOLVED 2026-05-19** via PR #18 (property-level gate, defense in depth) + PR #24 (runtime gate). The runtime gate in `src/operator/nn/dnnl/dnnl_convolution.cc::GetConvFwdImpl` detects the buggy combo (`!host_has_avx512 && quantized && eltwise_relu post-op && 0<ic<8 && out_md.get_data_type() == u8`) and skips appending the `eltwise_relu` post-op — the u8 dst saturation to [0, 255] already provides relu semantics, so dropping the post-op is correctness-preserving. The matching gate in `src/operator/subgraph/dnnl/dnnl_conv.cc` folds `output_scale` into `requantize_scales[]` so the u8 dst is correctly scaled. Full conv subgraph 428/428 pass. The underlying oneDNN bug in `3rdparty/onednn/src/cpu/x64/jit_uni_x8s8s32x_1x1_conv_kernel.cpp` remains and is tracked for upstream report; the in-tree gate is the long-term fix until oneDNN ships theirs. **Action remaining**: retest #2 and #49 against the new master tip — they share root cause and may now also pass.

5. **Backward through quantized ops** — **STEPS 1+2 RESOLVED 2026-05-18; STEP 3 IMPLEMENTED LOCAL 2026-05-19 (not pushed).**
   - **Step 1 DONE**: `quantize_v2` now has a Straight-Through Estimator (STE) backward — casts upstream gradient back to float32 and passes it through unchanged. Implemented in `src/operator/quantization/quantize_v2.cc`.
   - **Step 2 DONE**: `quantize_net(..., qat=True)` keeps `grad_req='write'` on all parameters, enabling gradient accumulation and optimizer updates for QAT training loops. `python/mxnet/contrib/quantization.py`.
   - **Step 3 IMPLEMENTED LOCAL 2026-05-19**: new `_backward_sg_onednn_fc` / `_backward_sg_onednn_conv` ops added in `src/operator/subgraph/dnnl/dnnl_qat_backward.cc` (679 LOC) — marked `TIsBackward + TIsLayerOpBackward`, gated behind `MXNET_QAT_SUBGRAPH_BACKWARD=1` (default off, preserving the legacy `MakeZeroGradNodes` behaviour for the GluonCV INT8 inference flow). Dequantizes saved int8/uint8 fwd inputs to fp32 with a unit scale and runs stock oneDNN backward primitives. Lives on local branch `fix/fu6-qat-subgraph-backward` (commits `74b62dc27` + `a24610a02`); **not yet pushed/PR'd** — see `handover.md` for push instructions.
   - **Net result**: `.backward()` through a quantized graph does not crash. With `MXNET_QAT_SUBGRAPH_BACKWARD=0` (current default and pre-Step-3 behaviour) it returns all-zero gradients; with the env var set it returns gradients with correct direction/sign (magnitudes off by `data_scale * weight_scale`, which the optimizer LR absorbs).

6. ~~**`test_activation` softrelu backward** (#13915).~~ **RESOLVED 2026-05-17**

48. ~~**`mx.random.seed` order-dependent on multi-context** (apache#18865).~~ **RESOLVED 2026-05-17** via commit `c2df8dd44`. Root cause: `cpu_rand_` and `cpu_parallel_rand_` in `ResourceManagerImpl` (`src/resource.cc`) were single `std::unique_ptr<>` instances shared across ALL CPU dev_ids. Calling `seed(456, ctx=cpu(1))` therefore re-seeded the same generator used by `cpu(0)`, so the value drawn from `cpu(0)` after seeding both `cpu(0)=123` and `cpu(1)=456` was DIFFERENT from the value drawn after seeding only `cpu(0)=123`. Fix: changed both members to `common::LazyAllocArray<ResourceRandom<cpu>>` / `LazyAllocArray<ResourceParallelRandom<cpu>>`, indexed by `Context::dev_id`, mirroring the per-dev_id design already in place for GPU. Each logical CPU context now owns an independent PRNG + engine variable. `SeedRandom(uint32_t)` (global seed) uses `ForEach` to seed all allocated CPU dev_ids; `SeedRandom(Context, seed)` indexes into the array by dev_id. Regression test: `tests/python/unittest/test_random_seed_order.py` (6/6 PASS in 0.96s).

46. ~~**`batch_dot` fp16 precision divergence from `dot`** (apache#18584).~~ **RESOLVED** — mshadow `BLASEngine<gpu, half_t>::batched_gemm` was calling `cublasHgemmStridedBatched` (true fp16 accumulators) while the 2-D `dot` path calls `cublasSgemmEx` (fp32 accumulators, pseudo-fp16). On CUDA 13 / Blackwell the divergence is severe: max relative error >500% on random (8,64,64,64) inputs. Fix: replaced `cublasHgemmStridedBatched` with `cublasGemmStridedBatchedEx(CUBLAS_COMPUTE_32F)` in `3rdparty/mshadow/mshadow/dot_engine-inl.h`. After fix: max rel-err = 0.00e+00 (exact bit-match). Regression test at `tests/python/gpu/test_batch_dot_fp16_parity.py` (6/6 PASS). Smoke: `test_fc_subgraph.py` 387/0/16 unchanged. Commit: `08cb44d1d`.

47. ~~**`linalg_impl.h` temp-buffer use without GPU synchronization** (apache#19353).~~ **RESOLVED 2026-05-17** — `potrf`, `batch_potrf`, `potri`, `batch_potri`, `gelqf`, `orglq`, `gelqf_workspace_query`, `syevd`, `gesvd`, `batch_getrf`, `batch_getri` all used `Storage::Get()->Alloc()` to allocate scratch buffers, enqueued cuBLAS/cuSOLVER work into the stream, then called `Storage::Get()->Free()` from the CPU side before the GPU kernels finished, allowing the freed block to be recycled and corrupted → silent NaN. Fix: replaced `EPHEMERAL_GPU_STORAGE_ALLOC` macro + manual `Free()` with a RAII class `LinalgEphemeralGPUStorage` in `src/operator/linalg_impl.h` that calls `cudaStreamSynchronize()` before freeing in its destructor. 11 call sites patched; no function signature changes. Stress test: `tests/python/gpu/test_linalg_temp_sync.py` (5 tests × 200 iters, all PASS). Commit: `251980078`.

---

## FUNCTIONALITY gaps (operations broken or not validated)

7. **bf16 path on this host** — Zen 2 EPYC 7B12 lacks AVX-512 BF16. oneDNN falls back to fp32 emulation. Not fixable in software. Test on Intel SPR / Zen 4 / Granite Rapids.

8. ~~**AMP (Automatic Mixed Precision) subgraph** — 6 `test_amp_subgraph.py` failures with `inner_product` primitive creation errors.~~ **RESOLVED 2026-05-17** via bf16→fp32 fallback in `dnnl_base-inl.h` + `dnnl_fully_connected.cc` + `dnnl_conv.cc` + `dnnl_transformer.cc`. `DNNLISASupportsLowpFloat()` detects AVX2 hosts at runtime and upcasts bf16 operands to fp32 for the DNNL primitive call, then reorders the fp32 result back to bf16. All 6 AMP subgraph tests now PASS (25s); FC subgraph 387/0/16 and test_dnnl.py 97/0 unchanged. See `amp_subgraph_fix.md` for full writeup.

9. ~~**AMP-RNN conversion**~~ **RESOLVED 2026-05-18**. The upstream #18099 "Error during waitall()" symptom (MXNet 1.6, cuDNN 7) does NOT reproduce on this fork's cuDNN-9 v8 RNN path. The in-tree `tests/python/gpu/test_amp.py::test_amp_conversion_rnn` was bit-rotted (called `mx.nd.ones`, missing fixture `amp_tests`, old `convert_hybrid_block` signature). Fixed the test: removed `@pytest.mark.skip`, added `@mx.util.use_np`, dropped the missing fixture parameter, replaced `mx.nd.ones` with `mx.np.ones`, passed `data_example` + `target_dtype` to `convert_hybrid_block`. Verified: 1 passed in 0.96s on GPU 1 (Blackwell sm_120). Caveat: AMP on this build currently passes the RNN node through as fp32 because the constant `_npi_zeros` state-input plumbing gates the cast — orthogonal to the original waitall bug, not a port-completion blocker.

10. ~~**NCCL multi-process not validated**~~ **RESOLVED 2026-05-18** — Single-process 2-GPU `kv.create('nccl')` confirmed working: float32/float16/uint8 push/pull, multi-dim shapes, multiple keys, 10/10 tests PASS. Bandwidth: 1 MiB ≈ 9 GB/s, 16 MiB ≈ 19 GB/s, 256 MiB ≈ 21 GB/s (push+pull round-trip, ~65% PCIe 4.0 x16). Per-process / 1-proc-per-GPU NCCL all-reduce is **by design NOT built into MXNet's kvstore layer** — `KVStoreNCCL` is single-process only (`ncclCommInitAll()` on one process's visible GPUs). Spawning N workers each calling `kv.create('nccl')` creates N independent size-1 communicators with no cross-process reduction (verified by test). DDP-style cross-process NCCL requires Horovod or BytePS (not in this wheel). Tests: `tests/python/gpu/test_nccl_singleproc.py` (10/10), `tests/python/gpu/test_nccl_multiproc.py` (3/3). See `nccl_status.md`.

11. ~~**Test-source bugs blocking 80+ test invocations**~~ **RESOLVED 2026-05-17** via agent #46 (commit `fa51581cb` numpy test fixes) plus the 21-test unskip in `cedeb2f9b`. Re-verified on cuDNN-9.22 build:
   - `test_deformable_psroipooling` → 1 PASS
   - `test_np_empty_like` → 1 SKIP (separate bug about int8/uint8/bool not supported, documented inline at the skip)
   - `test_convolution_options` was never in this fork (no upstream test by that name; bug referenced has different path)

12. ~~**`test_gpu_memory_profiler_symbolic`** (#18564)~~ **RESOLVED 2026-05-18** via commit `ed6757e64`. Under oneDNN v3, the backward node of `mx.symbol.dot()` is allocated as `tensordot:node_0_backward` rather than the v2 name `tensordot:dot_backward`. Updated the expected `Attribute Name` in `tests/python/gpu/test_profiler_gpu.py` and removed `pytest.mark.skip` — test is now 1/1 PASS.

13. ~~**`test_convolution_multiple_streams`**~~ **DOCUMENTED 2026-05-18** via commit `ed6757e64`. Investigation shows the errors reach ~14% relative on Blackwell — genuine workspace/stream non-determinism between the cuDNN and non-cuDNN paths under different engine types, not a simple tolerance issue. Loosening to `atol=0.05` is insufficient; silencing would require `~0.20`, which defeats the workspace-corruption probe. Kept skipped; updated skip reason to explain the cuDNN-9/sm_120 root cause and reference upstream #18564.

14. ~~**ONNX export/import**~~ **RESOLVED 2026-05-18**. Original framing was incorrect — neither file errored at collect time. The real bug: ONNX opset 18 (ONNX 1.13+, default in installed ONNX 1.21 / opset 26) moved the `axes` parameter from a node *attribute* to a second *input tensor* for `ReduceMean`/`ReduceMax`/`ReduceMin`/`ReduceProd`/`ReduceL1`/`ReduceL2`/`ReduceLogSum`/`ReduceLogSumExp`/`ReduceSumSquare`. The exporter in `python/mxnet/onnx/mx2onnx/_op_translations/_op_translations_opset13.py` was still emitting `axes=...` as a keyword attribute. Added a `make_reduce_node` helper (37 lines) that dispatches on `opset_version`: <18 keeps attribute, >=18 uses input tensor. 9 converters updated: `convert_{layer_norm,max,min,mean,prod,norm,npi_{mean,prod,min,max}}`. Results: norm/layer_norm suite 113/113 pass; broader 633 pass / 1 fail (pre-existing fp16 numerical precision on `test_onnx_export_softmax[None-float16]`, unrelated to the Reduce API change).

15. ~~**Gluon pretrained model loading**~~ **RESOLVED 2026-05-18** — `test_gluon_model_zoo.py` runs 34/34 PASS (1 pre-existing skip for `test_parallel_download`, marked upstream #17782). Full run: all 34 model-architecture forward passes succeed; the sole pretrained-checkpoint download (`mobilenetv2_0.25` from Apache CDN) also succeeds. No symbol-serialisation errors, no URL failures, no MXNet 2.0 numpy-op renames needed. The original issue description was speculative — nothing was actually broken. `test_parallel_download` remains skipped (MXNet fork-safety issue, upstream-tracked, out of scope).

16. ~~**Custom C++ operators**~~ **AUDITED 2026-05-18** — full surface validated on Blackwell sm_120 / CUDA 13. 9/9 tests pass: `example/extensions/lib_custom_op/test_{gemm,relu,transposecsr,transposerowsp}.py`, `tests/python/unittest/test_extensions.py::test_{custom_op,subgraph,external_op}`, `tests/python/gpu/test_extensions_gpu.py::test_{custom_op_gpu,external_op}`. No Thrust 3 / CUDA 13 specific regressions: the custom-op `lib_api.h` GPU path uses only raw `cudaStream_t` + `curand_kernel.h` (no Thrust), so it is unaffected by the CCCL 3 unification. One cosmetic teardown issue (`OSError: libdl.so` in `MXlib.__del__`) was caused by glibc 2.34+ absorbing libdl into libc; fixed in `python/mxnet/library.py` to try `libdl.so.2` then fall back to `libc.so.6`.

---

## PERFORMANCE gaps (works but suboptimal)

17. ~~**cuDNN 9.x sm_120 heuristic gap** (task #34).~~ **RESOLVED 2026-05-17** via commit `f103c5491` — bumped cuDNN 9.14 → 9.22 (locally bundled under `cudnn_local/`, system untouched). Headline impact: depthwise 3×3 256→256 went from 0.16 → 1.14 TFLOPS (**~7×**). Other shapes within noise (e.g. 3×3 28×28 256→256 bs32: 41.07 → 41.52 TFLOPS, same arena). Regression smoke clean: `test_fc_subgraph.py` 387/0/16 unchanged, `test_dnnl.py` 97/0 (including 72/72 `test_adaptive_pooling`).

18. ~~**`cudnnFindAlgorithm` / autotune not ported to v9** (task #35).~~ **RESOLVED 2026-05-17** via commit `e0eb106ea` — added `UseFrontendAutotune()` (env `MXNET_CUDNN_AUTOTUNE_FRONTEND`, default off) and `GetCombinedPlans()` which unions `CUDNN_HEUR_MODE_A + MODE_B` engine configs (deduped by plan string) and feeds the merged candidate list to `FindTopPlans`. On sm_120 / cuDNN 9.22, both modes select the same engine for the canonical 256→256 3×3 bs32 shape (parity with legacy). The combined path exposes a larger candidate set (20–23 plans vs fewer from a single mode) and is valuable for non-standard shapes where Mode A alone misses the fastest kernel. See `cudnn_autotune_v9.md` for details.

19. **`cuBLASLt` adoption** — **PR-A + PR-B LANDED 2026-05-18** (already in this branch via earlier commits `75232ca9b` PR-A fp32 + `05af4d576` PR-B fp16/bf16/fp64). FullyConnected GPU forward path dispatches through `MaybeCublasLtSgemm` / `Hgemm` / `Bf16Gemm` / `Dgemm` in `src/operator/linalg_impl.h` when `MXNET_USE_CUBLASLT=1`. Falls back to legacy cuBLAS on heuristic failure. Per-device LRU heuristic cache (cap=256), 32 MiB workspace, default off pending numerics audit. Parity test added at `tests/python/gpu/test_cublaslt_fc.py` (12 cases: 4 dtype-modes × 3 shapes via subprocess-isolated env). **PR-C (TRT-friendly stride-aware paths), PR-D (INT8), PR-E (default-on)** deferred to a future port revision — none block the Blackwell wheel since the env-gated opt-in path is correct.

20. ~~**No fatbin SASS for sm_120**~~ **CORRECTED 2026-05-18** — original framing was wrong. `MXNET_CUDA_ARCH=12.0+PTX` already generates BOTH `-gencode arch=compute_120,code=sm_120` (SASS) AND `-gencode arch=compute_120,code=compute_120` (PTX). The "JIT stall" never existed in this configuration. The real cleanup is dropping the redundant PTX forward-compat: re-configured with `MXNET_CUDA_ARCH=12.0` (SASS only). Eliminates ~50 MB from `libmxnet.so` (~20-30 MB compressed in wheel). Trade-off: wheel will not run on any future Blackwell variant (sm_121+) without rebuild — acceptable since this is a Blackwell-only port. Awaits next coordinated rebuild (237 CUDA files would rebuild). If multi-arch wheel is wanted later, issue #31 is the place to do it.

21. ~~**Sparse ops post-CCCL-3**~~ **BENCHMARKED 2026-05-18** (`bench_sparse_thrust3.py`, full table in `sparse_thrust3_bench.md`). No regressions on Blackwell sm_120 / CUDA 13 / Thrust 3. `dense_to_csr` is dominated by the prefix-sum scan over the full tensor (density-independent, ~linear in element count). `csr_to_dense` is 7–8× slower than forward due to the zero-fill + scatter on the output buffer — pre-existing characteristic. `topk(k=10|100|1000)` is K-independent because MXNet calls `thrust::sort` on full rows then slices; Thrust 3 maps this to a single `cub::DeviceRadixSort` with no partial-sort path. Open follow-up (non-blocking): implement partial-sort `topk` so K-dependence becomes sub-linear in dim size — worth doing for recommender / wide-vocab inference, not a port-completion blocker.

22. ~~**Storage manager defaults**~~ **BENCHMARKED 2026-05-18** via commit `561acebde`. `bench_gpu_storage_pool.py` runs ResNet-18 v2 forward+backward at batch sizes 1/8/32/128/256 on GPU 0 with both pool types. Results (`storage_pool_bench.md`): Round uses 1–6% **more** peak memory than Naive (avg +3.5%), with no OOM at any batch size on 24 GiB. The "fragments less" hypothesis is **not confirmed** for ResNet-18 on Blackwell. Both pool types comfortably fit batch=256 (Naive=13.2 GiB, Round=13.8 GiB). Recommendation: keep default (Naive); Round is not a free win for regular-allocation workloads. Users with highly irregular allocation patterns (e.g., NLP models, dynamic shapes) should benchmark on their own workload.

23. **TF32 default selection** — **AUDITED 2026-05-18** (`tf32_audit_2026-05-18.md`). Found one actionable defect: `src/operator/nn/cudnn/cudnn_deconvolution-inl.h:427-430` hard-codes `CUDNN_DEFAULT_MATH` for FP32 with no TF32-enable block, while the sibling `cudnn_convolution-inl.h:414-418` correctly checks `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION`. FP32 deconvolution on sm_120 therefore never picks up TF32 (~2× perf gap on transposed-conv-heavy nets — segmentation decoders, GAN generators, super-resolution). Patch at `.investigations/n23_tf32_patch.patch` adds the same guard to deconvolution. `cudnn_ops.cc:126-127` (the active sm_120 fwd path) already enables TF32 by default — correct. Patch awaits next coordinated rebuild.

24. ~~**fp16 tensor cores**~~ **BENCHMARKED 2026-05-18** (`bench_fp16_mxnet_vs_pytorch.py`, full table in `fp16_perf_bench.md`). Compared MXNet 2.0.0+cu13.bw.20260518 vs PyTorch 2.11.0+cu128 on Blackwell sm_120, fp16. Tensor-core ops at parity: Conv2D 0.84-0.99×, large Dense (MNK=4096³/8192³) 1.02-1.05× (MXNet matches or slightly beats PyTorch). Small/bandwidth-bound ops are 2-6× slower on MXNet: Softmax 0.17×, LayerNorm small 0.35×, Dense 1024³ 0.48×, elementwise add/mul at 1-4M elements 0.02-0.38×. Root cause is **pre-existing MXNet architecture** (multi-pass softmax/layernorm vs PyTorch's fused single-kernel; NDArray Python→C++ dispatch overhead visible at sub-0.1 ms timescales), NOT a Blackwell-port regression. The sm_120 tensor-core dispatch is correct. Follow-up issues (separate, non-blocking): (a) fused single-kernel softmax, (b) fused LayerNorm, (c) reduce elementwise dispatch overhead.

---

## TEST COVERAGE gaps

25. **Full `test_operator_gpu.py` sweep never run** on this build. ~28,408 tests total in `tests/python/`; this session validated DNNL subgraphs (815/3 pass), RNN (357/0), a slice of INT8 (~25 tests). Planned post-rebuild.

26. ~~**Distributed training**~~ **PARTIAL — SINGLE-MACHINE VALIDATED 2026-05-18**. `KVStore('local')` + `KVStore('device')` + `local_allreduce_{cpu,device}` exercised: 38/38 pass at `tests/python/gpu/test_kvstore_single_machine.py` (fp32+fp16, shapes 16 to 1M, cross-GPU push/pull bitwise exact, 10-iter SGD trajectories, 2-GPU gradient aggregation). NCCL collective already validated separately at `tests/python/gpu/test_nccl_singleproc.py` (10/10) + `test_nccl_multiproc.py` (3/3, single-process 2-GPU). **Still out of scope**: multi-machine `KVStore('dist_*')` requires a `ps-lite` rendezvous server — not deployed on this box. For multi-node training, recommend Horovod (NCCL point-to-point) rather than the ps-lite path. No code changes needed; the C++ KVStore backend is correct.

27. **`test_gluon_data*.py`** files segfault (`SIGSEGV rc=134`) — `test_gluon_data.py`, `test_contrib_gluon_data_vision.py`, `test_image.py` all had crashes in earlier sessions. Data pipeline (DataLoader, transforms, image augmentation) needs investigation. **RESOLVED 2026-05-18** (HEAD `c8ccd53e8`): all three files clean in isolation on current binary — `test_gluon_data.py` 30/0/0 in 51.7s, `test_contrib_gluon_data_vision.py` 3/0/0 in 9.8s, `test_image.py` 14/0/4 in 32.2s; combined (CUDA visible) 47/0/4 in 82.5s; `multi_worker` quartet stable over 3 reruns. The earlier-session segfaults were transient pollution from large single-process sweeps (cf. `.claude/worktrees/agent-a6161a6b131ea981f/PERFORMANCE_NOTES.md`, `.investigations/cpu_failures/investigation_notes.txt`) already addressed by `7934d40d7` (batchify legacy NDArray) and `bd09b1a7b` (per-test `reset_np()` scoping). Recommendation: drop the `--ignore` for these three files from the sweep config — Category (d) flake, not a real bug.

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

## APPLE SILICON / CPU STATIC AUDIT FOLLOW-UP

This section tracks the code-audit findings from the Apple Silicon bring-up branch. The CUDA-only findings are retained here for later Linux/CUDA CI, but the current local fix queue is the non-CUDA subset that can be built and tested on this machine.

### Local fix queue

46. **DataLoader early-exit and timeout cleanup** — Python DataLoader workers can leave worker tasks, shared-memory handles, or subprocesses behind when an iterator is abandoned, times out, or receives `KeyboardInterrupt`. Hot spots: `python/mxnet/gluon/data/dataloader.py` result handling, worker pool shutdown, and the `thread_pool=True` fallback that can still select shared-memory batchify. Add deterministic iterator close/reset paths and tests that do not rely on `garbage_expected`.

47. **C++ no-python DataLoader early break** — the C++ DataLoader reset path is tied to natural generator exhaustion, so breaking early can keep iterator state alive longer than intended. Add a cleanup path on generator close and test early-break reuse.

48. ~~**Image header sniffers read past short buffers**~~ — **RESOLVED on Apple Silicon follow-up branch.** `src/io/image_io.cc` now length-gates JPEG/PNG signature and dimension reads before falling back to OpenCV. Regression coverage: `tests/python/unittest/test_image.py::TestImage::test_imdecode_truncated_headers`.

49. ~~**libjpeg-turbo RecordIO decode lacks malformed-input cleanup**~~ — **RESOLVED on Apple Silicon follow-up branch.** `src/io/iter_image_recordio_2.cc` now short-buffer checks JPEG detection, uses full encoded byte size, and wraps `tjhandle` in RAII so malformed JPEG fallback cannot leak the handle. Rebuilt OpenCV/libjpeg-turbo profile and verified `test_image.py` plus focused RecordIO image tests.

50. **Histogram validation and edge handling** — `histogram` accepts invalid bin counts and has right-edge paths that can touch one past the bin-bound array. CPU can be fixed and tested locally; the matching CUDA kernel should be handled under CUDA CI.

51. **Fixed-size arrays in multi-array optimizers** — `multi_all_finite` and several multi-optimizer kernels use fixed arrays sized for a limited number of inputs. Validate or replace with dynamically sized storage to avoid overflow when large grouped updates are passed.

52. **Quantized flatten empty tensors** — `src/operator/quantization/quantized_flatten-inl.h` can leave min/max outputs uninitialized for empty input. Define empty-input behavior and add a regression test.

53. **Proposal operator integer overflow risk** — proposal workspace and anchor counts use `int` in several size calculations. CPU code should be widened or guarded locally; CUDA variants need CUDA CI.

54. **oneDNN quantized transpose min/max writes** — `src/operator/quantization/dnnl/dnnl_quantized_transpose.cc` can skip scalar min/max writes when the data output request is `kNullOp`. Ensure aux outputs are still honored independently.

55. **OpenMP state uses `volatile` for synchronization** — `src/engine/openmp.*` uses `volatile` state instead of atomics or locking. Replace with standard synchronization before enabling broader OpenMP coverage.

56. **Engine and custom-op exception state is not consistently synchronized** — threaded engine exception tracking and custom-op exception globals have shared state paths that should be protected or made thread-local. Add targeted stress tests where practical.

57. **Resource lifetime through async engine callbacks** — thread-local temp resources and local KVStore communication state can outlive assumptions when callbacks run asynchronously. Audit captures and ownership before broadening concurrency tests.

58. **Azure filesystem option is selectable but incomplete** — CMake exposes `USE_AZURE`, but the dmlc-core Azure implementation is a stub/incomplete dependency path. Either wire the real SDK dependencies or fail configure clearly when enabled.

59. **oneDNN generated headers copied into the source tree** — the current CMake path copies generated oneDNN headers into the checkout, risking stale tracked/untracked source-tree state. Generate them under the build tree instead.

60. **Plugin unload ownership split** — C++ and Python plugin loading/unloading have separate ownership surfaces. Audit `dlclose` ownership and lifetime to avoid unloading a library while registered symbols remain reachable.

### CUDA-only deferred queue

61. **cuBLASLt shared workspace race** — CUDA-only; requires Linux/CUDA stress testing before changing.

62. **CUDA architecture defaults and older CUDA 12.x compatibility** — CUDA-only build behavior; defer to Linux/CUDA CI.

63. **cuDNN frontend no-plan aborts instead of fallback** — CUDA-only runtime path; defer to Linux/CUDA CI.

64. **Skipped cuDNN multi-stream regression** — CUDA-only correctness/perf test; defer to Linux/CUDA CI.

65. **CUDA zero-block launches and GPU split edge cases** — CUDA-only kernels; defer to Linux/CUDA CI.

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

---

## Pointer: handover.md (2026-05-19)

See `handover.md` for a session-by-session summary of what landed in the
2026-05-18 → 2026-05-19 window (PRs #15 through #24), what's still on
local branches but not yet pushed (FU-6), and the priorities for the next
`.3` wheel cut.
