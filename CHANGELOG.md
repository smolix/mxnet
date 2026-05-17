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

- Blackwell (sm_120) support: CUDA 13.0, cuDNN 9.14.0, NCCL 2.28.3.
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

### Known issues

See [`issues.md`](issues.md) for the full list (45 items). Notable
remaining work for the next release:

- `adaptive_avg_pool` backward gives wrong results when
  `output_size < input_size` (36 failures, classification heads affected).
- One conv subgraph test (`test_pos_single_concat_pos_neg`) is order-dependent
  in the full pytest run (passes in isolation).
- AMP (`test_amp_subgraph.py`) — 6 `inner_product` primitive creation
  failures.
- bf16 path silently falls back to fp32 on CPUs without AVX-512-BF16
  (Zen 2 / Skylake / Ice Lake).
- ONNX export / import does not collect.
- `_contrib_quantize_asym` still uses the oneDNN v2 attr-on-reorder
  pattern.
- Backward through quantized ops is forward-only validated.
- Wheel does not bundle CUDA / cuDNN / NCCL runtimes; requires
  `apt install cuda-13 libcudnn9-cuda-13 libnccl2`.
- Only `sm_120` is in the fatbin — Ampere / Ada / Hopper users get nothing
  until a multi-arch build lands.

---

For Apache MXNet's pre-archive history see [`NEWS.md`](NEWS.md).
