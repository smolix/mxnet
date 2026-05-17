# Changelog

All notable changes to this fork of Apache MXNet are documented in this
file. This fork lives at [`smolix/mxnet`](https://github.com/smolix/mxnet)
and exists to port MXNet 2.0 to Blackwell (sm_120) / CUDA 13 / cuDNN 9 /
oneDNN v3, since upstream Apache MXNet was archived on 2023-11-17.

Version string format: `<upstream>+cu<cuda-major>.bw.<YYYYMMDD>`.

## 2.0.0+cu13.bw.20260517 — 2026-05-17

First Blackwell preview release. Forward inference for fp32 / fp16 / INT8 is
solid; quantized backward and several auxiliary paths remain open (see
[`issues.md`](issues.md)).

### Added

- Blackwell (sm_120) support: CUDA 13.0, cuDNN 9.22.0, NCCL 2.28.3.
- F16C CPU intrinsics for fast fp16 host-side (de)serialization.
- oneDNN v3.11 with the full INT8 path enabled: per-OC weight scales, fused
  conv / FC, fused sum, dequant-to-fp32 output, fused activations.
- GitHub Release wheel artefact for `linux_x86_64` (Python 3.10 - 3.13).
- Top-level `BUILDING.md` documenting the build recipe used for the
  release wheel.
- Top-level `issues.md` listing 45 open items (correctness, functionality,
  performance, test coverage, build/release, code quality).

### Changed

- cuDNN RNN code rewritten against the cuDNN v8 API
  (`cudnnRNNForward`, `cudnnRNNBackwardData_v8`,
  `cudnnRNNBackwardWeights_v8`, `cudnnSetRNNDescriptor_v8`). Legacy
  `cudnnRNNForwardTraining` / `cudnnRNNForwardInference` removed.
- INT8 plumbing: per-output-channel scales now bound on `DNNL_ARG_WEIGHTS`
  (oneDNN v3 conv / inner-product reject a non-zero mask on
  `DNNL_ARG_DST`).
- Reorder paths now route through `DNNL_ARG_SRC`. oneDNN v3 inverts the
  scale direction relative to v2 (in v3 the DST scale divides; the SRC
  scale multiplies).
- FC f32-output dequantization path now applies `DNNL_ARG_SRC` +
  `DNNL_ARG_WEIGHTS` scales together with an f32 bias (oneDNN v3 matmul
  rejects the v2 combination of s32 bias + f32 dst).
- Version string bumped to `2.0.0+cu13.bw.20260517` so downstream installs
  can pin a Blackwell build distinctly from the upstream 2.0.0 release.
- oneDNN submodule bumped from v2.x to v3.11; mechanical API drift swept
  across every primitive in `src/operator/nn/dnnl/`.

### Fixed

- `dd7553781` Port MXNet to CUDA 13 / cuDNN 9 / sm_120 (Blackwell) — root
  build / link / SASS-arch bringup.
- `273d03ec5` oneDNN v3 port: bump submodule + mechanical API drift fixes.
- `960be41c0` oneDNN v3 port: complete batch_normalization fwd + bwd.
- `a4fbf409e` oneDNN v3 port: mechanical fix-up across primitives.
- `d0ff77104` oneDNN v3 port: fix `get_dims()` temporary-pointer UB +
  batch_norm scale/shift descriptor.
- `6a0d94b24` oneDNN v3 port: subgraph fusion + pytest collection hygiene.
- `f170ceef3` oneDNN v3 port: drop `DNNL_ARG_SHIFT` from batch_norm
  backward args (v3 packs scale+shift differently).
- `9f3a77d28` oneDNN v3 port: drop double-free of `dnnl_memory_desc_t` in
  transpose.
- `46ada1129` oneDNN v3 port: wire INT8 runtime scales + fix per-OC mask +
  reorder direction.
- `740165f04` oneDNN v3 port: rescale `sum_scale` by `DST_scale` for fused
  conv / FC sum.
- `09055f014` oneDNN v3 port: cap weight_scales to prevent int32 bias
  overflow in conv.
- `817f5bea1` oneDNN/cuDNN port: rewrite cuDNN RNN path for cuDNN v8 API.
- `a3580d514` oneDNN v3 port: quantized FC dequant-output + `eltwise_soft_relu`
  alpha fix.
- `37489fd90` release: bump version to 2.0.0+cu13.bw.20260517.
- `9d5ce375d` oneDNN v3 port: bind per-input scale args at execute for
  quantized binary add.
- `870e144d3` docs: add `bugreport.md` from 2026-05-17 oneDNN v3 port
  audit.
- `8f6cc19ad` oneDNN v3 port: masked_softmax scale direction + activation
  backward alpha + conv f32 output dequant.
- `5a8d2e1ab` oneDNN v3 port: route quantized RNN weight reorder through
  brgemm RNN implementation.
- `1df0ff579` oneDNN v3 port: invert per-tensor DST scale for quantized FC.
- `7578f3ca9` oneDNN v3 port: fix B3+B5+B6+B7+B8 critical bugs (plus
  F1/F5/F10 follow-ups).
- `769735127` docs: add `issues.md` with remaining port work.

#### Late-day perf + correctness round (autonomous session, 2026-05-17)

- `cedeb2f9b` test: re-enable 21 upstream-disabled tests audited GREEN.
- `7934d40d7` gluon.data: handle legacy `nd.NDArray` samples under
  np-semantics (unblocks `test_gluon_data.py`, 30/30 in isolation).
- `bd09b1a7b` test: scope `test_image.py::reset_np()` to per-test fixture
  (cross-file pytest pollution gone).
- `783cfa133` cudnn ops: enable TF32 by default for FP32 conv on cuDNN 9.
  **Measured 2.87× speedup** on sm_120 (3×3 28×28 256→256 batch 32:
  14.46 → 41.48 TFLOPS). Mirrors PyTorch / TF defaults on cuDNN 9.
- `f103c5491` deps: bump cuDNN 9.14 → 9.22 (locally bundled). Headline
  win: **depthwise 3×3 256→256 went 0.16 → 1.14 TFLOPS (~7×)** —
  exactly the sm_120 fallback case Issues.md #17 flagged. No
  shape regressed.
- `7e4231da5` test: re-enable `test_activation` (issue #13915 flake no
  longer reproduces; 4/4 PASS across 4 different `MXNET_TEST_SEED`).
- `f8b0c7125` setup: tag wheel as binary distribution (was incorrectly
  `py3-none-any`; now `cp311-cp311-linux_x86_64`).
- `83718e389` release: switch to PyTorch-style pip-deps wheel — no
  longer bundles CUDA / cuDNN / NCCL libs under `mxnet/lib/`. Declares
  `nvidia-cudnn-cu13>=9.22` and `nvidia-nccl-cu13>=2.28` as
  `install_requires`; the rest of the CUDA 13 toolkit (cudart /
  cublas / cufft / cusolver / curand / nvrtc) is resolved at runtime
  via `libmxnet.so` `RUNPATH` against the system CUDA 13 install at
  `/usr/local/cuda/`. **Wheel size: 2.22 GB → 454 MB (79% reduction)**,
  now fits under the GitHub Releases per-asset limit.
- Tagged and published as
  [`v2.0.0.cu13.bw.20260517-beta`](https://github.com/smolix/mxnet/releases/tag/v2.0.0.cu13.bw.20260517-beta)
  (prerelease) with the slim wheel as a release asset.

### Known issues

See [`issues.md`](issues.md) for the full list. Notable remaining work
for the next release:

- `test_pos_single_concat_pos_neg[*-data_shape1]` — real int8 quantized
  concat numerical bug (entire output channels are zeroed). Suspect
  oneDNN v3 uint8→int8 reorder semantics; needs `dnnl_verbose=2` trace.
- AMP (`test_amp_subgraph.py`) — 6 `inner_product` primitive creation
  failures.
- bf16 path silently falls back to fp32 on CPUs without AVX-512-BF16
  (Zen 2 / Skylake / Ice Lake). Hardware-bound.
- ONNX export / import does not collect; module path was never updated
  for MXNet 2.0 numpy ops.
- Backward through quantized ops is forward-only validated.
- `cublasLt` not adopted (FP32 GEMM goes through legacy cuBLAS).
  Scoped at ~1130 LOC across 5 PRs (see `cublaslt_scope.md`).
- cuDNN frontend autotune not ported (heuristic mode A only).

#### Resolved between intermediate snapshot and release

- ~~`adaptive_avg_pool` backward gives wrong results~~ — CPU-reference
  fallback path; 72/72 PASS.
- ~~order-dependent conv subgraph test~~ — re-tested and identified as
  a real bug (no longer marked flake; see above).
- ~~Only `sm_120` in fatbin~~ — multi-arch (sm_80, 86, 89, 90, 120 +
  PTX 120 fallback) shipped.
- ~~Wheel does not bundle CUDA/cuDNN/NCCL runtimes~~ — wheel is now
  self-contained (cuDNN 9.22, NCCL 2.28, CUDA 13 runtime libs bundled
  under `mxnet/lib/` with RUNPATH = `$ORIGIN/lib`).
- ~~`_contrib_quantize_asym` v3 attr-on-reorder pattern~~ — fixed via
  agent #45 commit (`set_scales_mask` on the reorder primitive_attr).

---

For Apache MXNet's pre-archive history see [`NEWS.md`](NEWS.md).
