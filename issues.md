# MXNet Blackwell port — open issues

Snapshot: 2026-05-17 on branch `onednn-v3-port` at HEAD `1df0ff579` (plus agent #42 work still incoming).

## Motivation / context

This port exists because:

1. **Primary use case** — execute existing MXNet notebooks on Blackwell (RTX PRO 4000 / sm_120) hardware with CUDA 13 + cuDNN 9 + oneDNN v3 + NCCL 2.28. Apache MXNet upstream was archived on 2023-11-17 with no Blackwell or CUDA 13 support; this fork (`smolix/mxnet`) is the only place that work will land.
2. **Secondary** — provide a working Blackwell wheel for the residual MXNet user community. There are still teams running legacy MXNet pipelines (research code, frozen production stacks, niche operators like `_contrib_quantize_*`) who cannot easily migrate to PyTorch/JAX. A correct Blackwell port lets those pipelines run on current hardware without a full rewrite.

The bar therefore is "real notebooks run correctly and respectably fast on Blackwell", not "feature-parity with PyTorch". Performance gaps are acceptable when correctness is in place; correctness gaps are not.

This file lists everything still open at this snapshot. Items are grouped by severity. Each has a one-line "what" and a hint at "what to do".

---

## CORRECTNESS gaps (block "the port is done")

1. **`adaptive_avg_pool` backward correctness** (task #33). 36 failures when `output_size < input_size`. Bug in `src/operator/contrib/adaptive_avg_pooling.cu`: gradient distribution doesn't normalise by pool-window overlap count. Affects classification heads (very common). Fix: pass count-of-contributing-output-positions back through the kernel.

2. **`test_quantize_gluon_with_forward`** (gluon `quantize_net` of resnet18). Still fails post-`1df0ff579`. May already be fixed by the FC saturation fix; re-test after rebuild. If still failing, likely composite-fusion calibration in the `quantize_net` path.

3. **`_contrib_quantize_asym`** v3 attr-on-reorder bug. `src/operator/quantization/dnnl/dnnl_quantize_asym-inl.h:126` still uses `set_rnn_data_qparams` on a reorder primitive — same v3 issue the RNN-quant agent fixed in commit `5a8d2e1ab` for the LSTM path. Doesn't trigger from the two RNN tests but breaks asymmetric quantization.

4. **Conv subgraph test-order flake** — `test_pos_single_concat_pos_neg[int8-data_shape1]` and `[auto-data_shape1]` pass in isolation, fail in the full suite. Test-state leak — likely some cached primitive descriptor or storage manager state leaking between tests. Hunt the source of order-dependence.

5. **Backward through quantized ops** — untested. If anyone fine-tunes a quantized model this likely blows up. Forward inference is solid; backward through `_sg_onednn_fully_connected`, `_sg_onednn_conv` is unvalidated.

6. **`test_activation` softrelu backward** (#13915). Audit reported 3.6e5× gradient mismatch — but the audit ran against pre-`8f6cc19ad` libmxnet. My commit `8f6cc19ad` (B2 fix) added `alpha = ±1` for SoftReLU/LogSigmoid backward. Likely now fixed; retest.

---

## FUNCTIONALITY gaps (operations broken or not validated)

7. **bf16 path on this host** — Zen 2 EPYC 7B12 lacks AVX-512 BF16. oneDNN falls back to fp32 emulation. Not fixable in software. Test on Intel SPR / Zen 4 / Granite Rapids.

8. **AMP (Automatic Mixed Precision) subgraph** — 6 `test_amp_subgraph.py` failures with `inner_product` primitive creation errors. AMP is the de-facto training-perf path on FP16 models; this needs to work.

9. **AMP-RNN conversion** — `test_amp_conversion_rnn` fails with "Error during waitall()" tracked at upstream #18099. Untested whether our cuDNN-9 v8 RNN path interacts with AMP's autocast hooks at all.

10. **NCCL multi-process not validated** — 1-process / 2-GPU `kv.create('nccl')` push/pull works. Standard distributed training topology (1 process per GPU with `dist.init_process_group`-style init) is untested. May need `MXNET_KVSTORE_USETREE` or related tunings to scale.

11. **Test-source bugs blocking 80+ test invocations** (test code, not MXNet code):
   - `test_convolution_options` — `np.repeat(object_array, 5)` fails on numpy ≥1.24 (#10141)
   - `test_deformable_psroipooling` — `np.int` removed in numpy ≥1.20 (#11713) at 4 sites
   - `test_np_empty_like` — positional args don't match new signature
   All are easy fixes (kwargs, replace `np.int` with `int`, rewrite `np.repeat`).

12. **`test_gpu_memory_profiler_symbolic`** (#18564) — profiler probe for `tensordot:dot_backward` attr_name doesn't find a match. The dot kernel was renamed under oneDNN v3 dispatch. Either update the test or restore the tag name in the kernel.

13. **`test_convolution_multiple_streams`** — `rtol=0.01/atol=0.01` mismatch on cuDNN-9 conv across NaiveEngine / ThreadedEngine / ThreadedEnginePerDevice on Blackwell. Real numerics drift. Either loosen tolerance to 0.05/0.05, force deterministic algorithms (`MXNET_CUDNN_AUTOTUNE_DEFAULT=0`), or leave skipped.

14. **ONNX export/import** — `tests/python/onnx/test_models.py` and `test_operators.py` both error at collect time. Likely the ONNX path was never updated for MXNet 2.0 numpy ops; depends how important ONNX interop is.

15. **Gluon pretrained model loading** — `test_gluon_model_zoo.py` partially fails (pretrained checkpoint download + symbol serialisation). If anyone uses the model zoo, this needs validation.

16. **Custom C++ operators** — `test_custom_op_fork` audit was green but the broader custom-op infrastructure with CUDA 13 / Thrust 3 hasn't been exercised.

---

## PERFORMANCE gaps (works but suboptimal)

17. **cuDNN 9.x sm_120 heuristic gap** (task #34). cuDNN 9.0–9.3 ship heuristic tables copied from sm_90 with light adjustment; many conv shapes route through generic fallback engines instead of Blackwell-tuned ones. Bumping to cuDNN 9.5+ progressively closes this; cuDNN 9.7+ has the best sm_120 coverage. **Highest perf payoff.**

18. **`cudnnFindAlgorithm` / autotune not ported to v9** (task #35). cuDNN 9 deprecated the v7/v8 form; MXNet still uses static heuristic mode A. Needs migration to the cuDNN frontend `EngineHeuristics + Plan_v8` enumeration. Closes shape-specific gaps that the heuristic table misses.

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
