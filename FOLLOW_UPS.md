# Follow-up tracking — smolix/mxnet

Issues are disabled on smolix/mxnet, so this file tracks open work items.
Each entry has a stable ID, last-updated date, status, and concrete next step.

---

## FU-1 — oneDNN AVX2 int8 1x1 conv + eltwise_relu + ic<simd_w bug

**Status**: OPEN, root cause localised (upstream oneDNN bug). Workaround documented.

**Root cause** (found by DNNL_VERBOSE=2 trace + `MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU=1` experiment on 2026-05-18): the channel-zero symptom is NOT in qconcat. It originates upstream in oneDNN's **`jit_uni_int8_1x1:avx2` quantized 1x1 conv** kernel when fused with `eltwise_relu` post-op AND `ic < simd_w` (AVX2 simd_w=8). The failing shape has `ic=3`; passing shapes have `ic=4` (mb=64) and `ic=16` (mb=1). The qconcat reorders faithfully copy already-wrong values.

**Attempts in qconcat that did NOT fix it (all correct, just attacking the wrong primitive)**:
1. `set_scales_mask(DNNL_ARG_SRC, 0)` with `out_scale/i_scale` (original)
2. `set_scales_mask(DNNL_ARG_DST, 0)` with `i_scale/out_scale` (mirror of `dnnl_quantize_asym-inl.h`)
3. u8→f32→s8 intermediate decomposition. Math validated; channels still zero.

**Failures attributable to this root cause**:

| Test | File | Symptom |
|---|---|---|
| `test_pos_single_concat_pos_neg[int8|auto-data_shape1]` | `tests/python/dnnl/subgraphs/test_conv_subgraph.py` | Max int8 error 5.0 vs atol=0.12. Channels 3-6 of output zeroed. |
| `test_self_attention[*]` (32/32) + `test_self_attention_negative` (segfault) + `test_batch_dot[*]` (16/16) | `tests/python/dnnl/subgraphs/test_matmul_subgraph.py` | Massive overflow: output ~2e+23 vs expected ~0.14. Likely same kernel-tail bug applied to int8 matmul/batch-dot subgraphs. |
| `test_quantize_gluon_with_forward` under `ENABLE_ONEDNN_QUANTIZATION_TEST=1` | `tests/python/dnnl/test_quantization_dnnl.py` | Was passing 2026-05-17. Now segfaults in `_build_cache → quantize_net → CachedOp`. Same family. |

**Workaround**: `MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU=1`. Passes 6/6 across all data_shapes + seeds.

**Upstream code**: `3rdparty/onednn/src/cpu/x64/jit_uni_x8s8s32x_1x1_conv_kernel.cpp`. `jcp.ic = rnd_up(ic, simd_w)` pads to 8, but the tail-handling branch in `fma_block` mis-orders with s8s8 compensation (`vmm_shift`) when an eltwise_relu injector follows.

**Proposed in-tree workaround patch**: gate `jit_uni_int8_1x1:avx2` at the subgraph-quantize level in `src/operator/subgraph/dnnl/dnnl_conv_property.h`, rejecting conv+relu fusion on AVX2 hosts when `ic < 8`. Full investigation: `.investigations/qconcat_fix_v2.patch`.

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

**Status**: PATH B LANDED IN WHEEL `.2` (PR #18) — engine reset + `dnnl_mem_` invalidation in the atfork child. Verified by fashion_mnist + `num_workers=4` and via the regression tests in PR #20 (`tests/python/gpu/test_fu4_fork_safe_dnnl.py`). BUT the post-`.2` d2l sweep (2026-05-18 21:28) shows **7 RNN/embedding notebooks still fail** with the same surface symptom — see FU-11. Re-investigation 2026-05-18 22:18 shows that cluster is **NOT a fork issue** (`num_workers=0` in the failing path), so FU-4 itself is considered resolved for its intended scope.

**Description (original)**: 10 d2l-neu notebooks failed with bare `MXNetError: could not execute a primitive` when `gluon.data.DataLoader(num_workers>0)`. Root caused: `pthread_atfork` handlers in `src/initialize.cc:191-215` Stop/Start the Engine + clamp OpenMP, but did NOT clear oneDNN process-global state.

**Path A (Insufficient)**: `g_dnnl_forked_child` atomic + `DNNLEnvSet()` kill-switch (`.investigations/d2l_cat1_atfork.patch`, commit `5d99841c4`). Only checked at operator-level dispatchers; not at `NDArray::GetDNNLData()`. Superseded by Path B.

**Path B (Landed in PR #18)**: `DNNLAfterForkChild()` now (1) drops the oneDNN primitive cache, (2) re-placement-news the `CpuEngine` + `DNNLStream` singletons, (3) sets `g_dnnl_forked_child` so `NDArray::GetDNNLData()` drops a stale `ptr_->dnnl_mem_`. Touched: `src/initialize.cc`, `src/operator/nn/dnnl/dnnl_base-inl.h`, `src/operator/nn/dnnl/dnnl_base.cc`, `src/ndarray/ndarray.cc`. Regression tests in `tests/python/gpu/test_fu4_fork_safe_dnnl.py` (PR #20).

**Effective workaround (still valid)**: `MXNET_ONEDNN_ENABLED=0` env var — makes the broader d2l symptom (including FU-11) go away by avoiding oneDNN entirely.

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

## FU-6 — QAT backward through fused subgraph ops (`_sg_onednn_*`) [SCOPED+SCAFFOLDED]

**Status**: Scoped + scaffolded (104 LOC). Op body implementation deferred.

**Architecture investigation 2026-05-18** (`.investigations/fu6_qat_backward_scope.md`): Approach (a) is sound — new `_backward_sg_onednn_fc` / `_backward_sg_onednn_conv` ops marked `TIsBackward + TIsLayerOpBackward`, gated by `MXNET_QAT_SUBGRAPH_BACKWARD=1` while landing. The prior "segfault" was from feeding int8/quantized inputs into FP32-only generic ops via symbolic graph rewrite, NOT a CachedOp/NNVM limitation. State sharing via `cached_op.cc:628` (the `TIsLayerOpBackward` path) is the right mechanism.

**Scaffolding partial** (`.investigations/fu6_partial_scaffolding.patch`, 104 lines): headers + dispatcher hook on the forward ops. Still to write: `src/operator/subgraph/dnnl/dnnl_qat_backward.cc` (~600-700 LOC) containing the `SgDNNLFCFGradient`/`SgDNNLConvFGradient` lambdas, op registrations, and `FStatefulComputeEx<cpu>` wrappers around oneDNN `inner_product_backward_data/weights` and `convolution_backward_*`.

**Net result today**: `.backward()` through a quantized graph does NOT crash; it returns all-zero gradients. The STE is in place and will become effective once `_sg_onednn_*` get proper backward support. Tests: 13 PASS, 4 XFAIL.

**Effort to complete**: M (~1 focused day on the op bodies).

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

---

## FU-11 — d2l RNN/embedding cluster: `np.stack` op fails after Trainer setup (wheel `.2`)

**Status**: OPEN, **NOT a fork issue** — re-investigation 2026-05-18 22:18 found `num_workers=0` in `d2l.DataModule.get_tensorloader` (the path used by `TimeMachine`). FU-4 Path B is unrelated to this cluster.

**Source**: `/workspace/d2l-neu/mxnet-errors.md` (sweep against wheel `2.0.0+cu13.bw.20260518.2`, run 2026-05-18 21:28).

**7 failing notebooks** (all under `_notebooks/mxnet/`):
- `chapter_optimization/minibatch-sgd.ipynb` — multi-GPU `train_ch11`
- `chapter_recommender-systems/seqrec.ipynb` — `train_ranking` w/ NeuMF-style CTR DataLoader
- `chapter_recurrent-modern/deep-rnn.ipynb` — `Trainer(num_gpus=1).fit(GRU + 2-layer)`
- `chapter_recurrent-modern/gru.ipynb` — `Trainer.fit(RNNLMScratch(gru,...))`
- `chapter_recurrent-modern/lstm.ipynb` — same shape, LSTM
- `chapter_recurrent-neural-networks/rnn-concise.ipynb` — `Trainer.fit(RNN)` with fused `rnn.RNN`
- `chapter_recurrent-neural-networks/rnn-scratch.ipynb` — `Trainer.fit(RNNLMScratch w/ manual GRU)`

**Symptom**: bare `MXNetError: could not execute a primitive` from `_LIB.MXNetFuncCall`.

**Failure point (NaiveEngine localised, 2026-05-18 22:17)**:
```
fit_epoch → for batch in train_dataloader → DataLoader.__iter__ → same_process_iter
  → self._batchify_fn(...)
  → gluon.data.batchify.Stack.__call__ at batchify.py:95
  → mxnet.numpy.stack at multiarray.py:7536
  → _api_internal.stack(*arrays, axis, out)  ← FAILS HERE
```

**Key isolation results (this branch, wheel `.2`)**:
- `data.train_dataloader()` then iterating 3 batches DIRECTLY (no model, no trainer) → **PASSES** with NaiveEngine.
- Same DataLoader after constructing `RNNLMScratch(gru, ...)` + `d2l.Trainer(...)` + `trainer.fit(model, data)` → **FAILS** at first batch's `Stack` batchify (above).
- `MXNET_ONEDNN_ENABLED=0` → notebook trains successfully (timed out at 60 s mid-training, not a crash). Definitive workaround.

**Diagnosis (current best guess)**: model/optimizer initialization on GPU mutates some CPU oneDNN process-global state (primitive cache key counter? scratchpad? layout-engine singleton?) which then makes a subsequent CPU `np.stack` op (a tiny int32 stack of `(32,)` rows) fail. The stack op normally has nothing to do with the model, but it shares the CPU engine + DNNL primitive cache that the model's CPU-side init touched.

**Suggested next diagnostic**:
1. Bisect the trainer setup: which of `gru = GRU(...)`, `gru.initialize()`, `model = RNNLMScratch(...)`, `Trainer(...)`, `trainer.fit(...)`'s pre-loop setup actually moves the failure forward.
2. With the failure repro from (1), strace/perf the C++ side around `_api_internal.stack` to see whether DNNL is consulted on a tiny `(32, 32) int32` stack (it shouldn't be, but `MXNetStorageInferDNNL` may be over-eager).
3. Add a temporary `LOG(INFO)` at `MXNetStorageInferDNNL` and at the failing primitive's `execute` to capture which oneDNN primitive aborts.

**Repro**: `.investigations/fu11_d2l_rnn_stack_repro.py` (control path); full failure path requires `jupyter nbconvert` on the notebook (see the file's docstring).

**Workaround**: `MXNET_ONEDNN_ENABLED=0` (verified definitive).

Tracks d2l-neu `mxnet-errors.md` §A.1 (post-`.2` re-run).

---

## FU-12 — `transformer.ipynb` DeadKernelError under concurrent GPU load

**Status**: OPEN, root-cause unknown. Minimal repro from `mxnet-errors.md` does NOT reproduce in isolation — needs concurrent GPU pressure.

**Source**: `/workspace/d2l-neu/mxnet-errors.md` §A.2.

**Single failing notebook**: `_notebooks/mxnet/chapter_attention-mechanisms-and-transformers/transformer.ipynb`. Kernel dies stably at ~7 s × 5 retries. Sister notebooks in the same chapter PASS (multihead-attention, bahdanau-attention, attention-pooling, attention-scoring-functions, queries-keys-values, self-attention-and-positional-encoding).

**The d2l report's "8-line minimal" repro PASSES** in isolation on this machine (both CPU and GPU contexts), confirmed 2026-05-18 22:12. So the bug is environmental: jupyter kernel context, GPU contention with other notebook workers (`GPU_SLOTS=2`), or specific cell-ordering state.

**Hypothesis**: SIGKILL from OOM-killer or CUDA driver under contention. `nbconvert --execute` exit codes don't distinguish died-from-SIGKILL vs died-from-SIGABRT. Need to capture `dmesg` + GPU memory usage at death time.

**Suggested next diagnostic**:
- Re-run with `GPU_SLOTS=1` to isolate from contention.
- Watch `dmesg` + `nvidia-smi -lms 200` during nbconvert.
- Wrap kernel in `strace -e signal` or set `core` rlimit so we can postmortem the C++ stack.

**Workaround**: run notebook in isolation (no concurrent sweep). Not a wheel API bug we can fix from C++ until we have a determinist repro.

Tracks d2l-neu `mxnet-errors.md` §A.2.

---

## FU-13 — `np.nonzero` IndexError in `assign_anchor_to_bbox` context (ssd.ipynb)

**Status**: OPEN, NOT reproducible standalone.

**Source**: `/workspace/d2l-neu/mxnet-errors.md` §A.3.

**Single failing notebook**: `_notebooks/mxnet/chapter_computer-vision/ssd.ipynb`. Surface error from training loop:
```
IndexError: Traceback (most recent call last):
  File "/workspace/mxnet/src/operator/numpy/np_indexing_op.cu", line 370
IndexError: index 3417 is out of bounds for axis 0 with size 1
```

**Site in book**: `d2l.assign_anchor_to_bbox` calls `np.nonzero(max_ious >= iou_threshold)[0]` — fails on first training batch.

**Minimal repro from the d2l report does NOT reproduce** — `np.nonzero` on `(arange(10000) == 3417.0)` returns `array([3417])` correctly (verified 2026-05-18 22:12 on wheel `.2`). The standalone 1-D-bool-mask path works.

**Hypothesis**: bug requires the specific dtype / shape / boolean-mask pattern produced upstream of `assign_anchor_to_bbox`. The index value (`1441 / 1860 / 3417`) changes run-to-run — it's data-dependent. May involve broadcasting of `max_ious >= iou_threshold` over a `(num_anchors, num_gt_boxes)` matrix where `num_gt_boxes == 1`.

**Suggested next diagnostic**: capture the actual arg to `np.nonzero` inside `assign_anchor_to_bbox` (shape, dtype, sum) using a `try`/`except` wrapper, then write a standalone repro with that exact tensor.

**Workaround**: none discovered. d2l book's `assign_anchor_to_bbox` is canonical; we should diagnose the wheel op rather than patch the book.

Tracks d2l-neu `mxnet-errors.md` §A.3.

---

## FU-14 — `sentiment-analysis-rnn.ipynb` OOM during training

**Status**: OPEN, low priority. May not be a leak — could be a real memory-budget issue.

**Source**: `/workspace/d2l-neu/mxnet-errors.md` §A.4.

**Single failing notebook**: `_notebooks/mxnet/chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb`. Fails after ~125 s of training with:
```
File "/workspace/mxnet/src/storage/./pooled_storage_manager.h", line 240
MXNetError: Memory allocation failed out of memory
```

**Model**: `nn.Embedding(vocab=49328, 100) + rnn.LSTM(100, num_layers=2, bidirectional=True) + nn.Dense(2)` on IMDB, `batch_size=64`. 24 GB GPU.

**Hypothesis (per d2l report)**: activation-memory leak vs. real budget issue. Differentiate by running with `GPU_SLOTS=1` (no concurrent worker).

**Suggested next diagnostic**:
- Try `GPU_SLOTS=1` re-run; if it passes, this is contention, not a leak.
- If it still OOMs, instrument `pooled_storage_manager.h:240` to log live/peak buffer footprint.
- Try `MXNET_GPU_MEM_POOL_TYPE=Round` to see if pool fragmentation is at fault.

**Workaround**: `GPU_SLOTS=1` likely sufficient.

Tracks d2l-neu `mxnet-errors.md` §A.4.
