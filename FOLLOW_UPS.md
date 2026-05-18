# Follow-up tracking — smolix/mxnet

Issues are disabled on smolix/mxnet, so this file tracks open work items.
Each entry has a stable ID, last-updated date, status, and concrete next step.

---

## FU-1 — DNNL v3 INT8 quantize subgraph regression family

**Status**: OPEN. Affects three test files; all share the same suspected oneDNN v3 INT8 scale-convention root cause.

**Failures**:

| Test | File | Symptom |
|---|---|---|
| `test_pos_single_concat_pos_neg[int8|auto-data_shape1]` | `tests/python/dnnl/subgraphs/test_conv_subgraph.py` | Max int8 error 5.0 vs atol=0.12. Channels 3-6 of output (the `relu(conv0(x))` half) zeroed. |
| `test_self_attention[*]` (32/32) + `test_self_attention_negative` (segfault) + `test_batch_dot[*]` (16/16) | `tests/python/dnnl/subgraphs/test_matmul_subgraph.py` | Massive overflow: output ~2e+23 vs expected ~0.14. |
| `test_quantize_gluon_with_forward` under `ENABLE_ONEDNN_QUANTIZATION_TEST=1` | `tests/python/dnnl/test_quantization_dnnl.py` | Was passing 2026-05-17. Now segfaults in `_build_cache → quantize_net → CachedOp`. |

**Attempts that did NOT resolve #4 (concat)**:
1. `set_scales_mask(DNNL_ARG_SRC, 0)` with `out_scale/i_scale` (original)
2. `set_scales_mask(DNNL_ARG_DST, 0)` with `i_scale/out_scale` (mirror of `dnnl_quantize_asym-inl.h`)
3. f32 intermediate (u8→f32 dequant + f32→s8 quant). Math correct; channels still zero.

**Next step**: `DNNL_VERBOSE=2` trace + `DNNLConcatFwd::GetCached` cache-key inspection. If u8→s8 JIT-reorder mixed-dtype is the bug, may need a oneDNN-level workaround or upstream bisect (3.11.3 vs older).

**Code**: `src/operator/quantization/dnnl/dnnl_quantized_concat.cc:106-130`, plus matmul/batch-dot subgraph dispatchers (`src/operator/subgraph/dnnl/*matmul*.cc`, `dnnl_fully_connected.cc` quantized branch).

**Repro**: `CUDA_VISIBLE_DEVICES= MXNET_TEST_SEED=11 pytest "tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_single_concat_pos_neg[int8-data_shape1]"`

Tracks `issues.md` #2, #4, #49.

---

## FU-2 — int8/fp16 mixed dtype path absent

**Status**: OPEN (design decision needed).

**Description**: Three independent dispatch sites reject fp16 input/output for quantize:
- `src/operator/quantization/quantize_v2-inl.h:157,160` accepts only kFloat32/kBfloat16/kUint8/kInt8 — fp16 omitted on purpose.
- `src/operator/quantization/dequantize-inl.h:42` has `add_enum("float32", kFloat32)` as the only allowed output dtype.
- `src/operator/quantization/dnnl/dnnl_quantize_v2-inl.h:98` reads `dptr<float>()` unconditionally.

**Implication**: AMP fp16 inference cannot call `contrib.quant.quantize_net` directly. Workflow needs `amp_cast(fp32)` upstream of every quantize.

**Decision needed**: (a) widen all three sites to accept fp16 (CPU oneDNN supports f16→s8 reorder; CUDA kernel needs MSHADOW_REAL_TYPE_SWITCH_EX branch); OR (b) document AMP fp16+quantize as unsupported and add AMP automatic `amp_cast(fp32)` upstream of quantize_v2 nodes.

**Side bug to fix either way**: `mx.nd.contrib.quantize_v2(bf16_data, ctx=gpu)` no-calib path abort()s with `dmlc::Error "TBlob.get_with_shape: data type do not match"` instead of returning a clean type-check error.

Tracks `issues.md` #50.

---

## FU-3 — CPU inference slow on AVX2-only hosts (B8 / apache#19218)

**Status**: OPEN, workaround documented.

**Symptom**: oneDNN v3 picks `brg_conv_fwd:avx2` which pads IC<16 to 16 (wastes 81% of AVX2 vector work on padding zeros) AND scales negatively with thread count at bs=1 on EPYC 8-NUMA-domain hosts (10× slower at 64 threads vs 1).

**Workaround**: `OMP_NUM_THREADS=1` for bs=1 inference. 10× speedup.

**Proposed fix**: in `src/operator/nn/dnnl/dnnl_convolution.cc`, gate `brg_conv` selection away from `IC<16 && bs=1 && AVX2-only`. Allow older `jit:avx2` path to win there.

**Effort**: M (1 day). Not a Blackwell-port blocker (CPU inference is side use case).

Tracks `issues.md` apache section B8 / apache#19218.

---

## FU-4 — d2l Cat 1 fork-time oneDNN cache (apache fork-safe DataLoader)

**Status**: PATCH APPLIED, rebuild in flight.

**Description**: 10 d2l-neu notebooks fail with bare `MXNetError: could not execute a primitive` when `gluon.data.DataLoader(num_workers>0)`. Root caused: `pthread_atfork` handlers in `src/initialize.cc:191-215` Stop/Start the Engine + clamp OpenMP, but do NOT clear oneDNN process-global state (`CpuEngine` singleton, `DNNLStream` thread-local, primitive LRU cache). Worker children inherit corrupt parent oneDNN state.

**Patch**: `.investigations/d2l_cat1_atfork.patch` (~25 lines, 3 files). Adds `g_dnnl_forked_child` atomic + `DNNLAfterForkChild()` that clears DNNL LRU cache, and makes `DNNLEnvSet()` return false in the child (i.e. disable DNNL like `MXNET_ONEDNN_ENABLED=0`).

**Next step**: rebuild completes (~5 more min as of 2026-05-18 17:30), copy libmxnet, verify `rnn-scratch.ipynb` passes with `num_workers=4`. Then commit + push to PR #15.

**Repro**: `.investigations/d2l_cat1_repro.py`, also the upstream `rnn-scratch.ipynb` itself.

Tracks task #85.

---

## FU-5 — cuBLASLt PR-C/D/E (apache#19, partial)

**Status**: PR-A + PR-B landed (`75232ca9b`, `05af4d576`). PR-C/D/E deferred.

**PR-C**: stride-aware (non-contiguous LD) cuBLASLt paths
**PR-D**: INT8 cuBLASLt adoption
**PR-E**: default-on (`MXNET_USE_CUBLASLT=1` default)

**Scope doc**: `/workspace/mxnet/cublaslt_scope.md`.

**Blocker**: PR-E needs a full numerics audit comparing legacy SGEMM vs cuBLASLt heuristic-selected algos for bit-exact divergence in user-facing tests. Not a port-completion blocker (env-gated opt-in is correct).

Tracks `issues.md` #19.

---

## FU-6 — QAT backward through fused subgraph ops (`_sg_onednn_*`)

**Status**: BLOCKED at architecture level.

**Description**: QAT STE (commit `df8f0378d`) gives quantize_v2 a working straight-through gradient. But the fused subgraph ops `_sg_onednn_fully_connected` and `_sg_onednn_conv` still have `FGradient = MakeZeroGradNodes`. An attempt to add a dot-product-based data backward caused segfaults in the NNVM/CachedOp backward executor — these fused ops interact with the static graph executor in a way that does not support custom backward nodes referencing op inputs.

**Net result today**: `.backward()` through a quantized graph does NOT crash; it returns all-zero gradients. The STE is in place and will become effective once `_sg_onednn_*` get proper backward support. Tests: 13 PASS, 4 XFAIL.

**Effort**: L (architectural — needs CachedOp / subgraph executor extension).

Tracks `issues.md` #5 Step 3.

---

## FU-7 — Multi-arch fatbin (apache#31)

**Status**: DEFERRED.

The current wheel is single-arch sm_120. To make the wheel useful on Ampere/Ada/Hopper:

```cmake
MXNET_CUDA_ARCH=8.0;8.6;8.9;9.0;12.0+PTX
```

Trade-offs: +250-400 MB wheel size, +30 min build time. Out of scope for the Blackwell-target wheel; would justify a separate multi-arch release track.

Tracks `issues.md` #31.

---

## FU-8 — Engine deadlock family (A6, A7 — partial)

**Status**: INSTRUMENTED only. A13 was fixed; A6 and A7 are watchdog-instrumented (`MXNET_ENGINE_DIAG=1`, 30 s default timeout) but the actual missing-notify-edge bug needs a minimised ARM (`aarch64-linux-gnu`) reproducer to fix.

**Watchdog active in**: `WaitForVar`, `WaitForAll`. On timeout, logs var pointer, `pending_ops`, `shutdown_phase`, `kill` flags, and a hint to use `MXNET_ENGINE_TYPE=NaiveEngine`.

Tracks `apache_issues_review.md` A6/A7, `engine_deadlock_audit.md`.

---

## FU-9 — d2l book bugs (not wheel)

The following d2l-neu mxnet-errors.md failures are book-side, NOT wheel bugs. Should be reported to d2l-neu maintainers:

1. **Cat 3 sentiment-analysis-rnn**: book uses `init.Xavier()` on flat 1D LSTM weight; should be `init.Normal(0.01)`.
2. **Cat 4 ssd IndexError 1860**: book's `assign_anchor_to_bbox` mishandles `num_gt_boxes=1` modulo (mxnet 2.0's nonzero semantics differ from 1.x).
3. **Cat 5 neumf**: book's `evaluate_ranking()` builds 943 `gluon.data.DataLoader` per eval call × every-epoch. Should use `eval_step=5` or single-DataLoader rewrite.

See `.investigations/d2l_findings_summary.md` for full triage.

---

## FU-10 — Future GitHub Actions sweep coverage

**Status**: Fast PR job at `.github/workflows/test-cpu.yml` (~20 min, ~4 test files). Nightly at `test-cpu-nightly.yml` (90 min, broader).

**Gap**: No self-hosted GPU runner. The GPU sweep (test_operator_gpu.py, test_gluon_gpu.py, etc.) is currently hand-run. Once a GPU runner is available, add a third workflow that runs the GPU subset on PRs touching `src/operator/nn/cudnn/*` or `src/common/cuda/*`.

Tracks `issues.md` #32 (closed in PR #15) — this is the natural follow-up.
