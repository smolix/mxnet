# MXNet Code Quality Audit — 2026-05-17

## Audit scope
- Branch: `onednn-v3-port` at `37489fd90`
- Focus: recently-touched oneDNN v3 + cuDNN 9 + INT8 + RNN port files
- Methodology: read-only static review; no compilation, no test runs

Files audited in detail:
- `src/operator/nn/dnnl/dnnl_*.{cc,h}` (act, base, batch_dot, batch_norm,
  conv, deconv, eltwise, fully_connected, layer_norm, masked_softmax,
  pow_mul_scalar, rnn, softmax, softmax_output, sum)
- `src/operator/subgraph/dnnl/dnnl_{conv,fc,transformer}.cc`
- `src/operator/quantization/dnnl/dnnl_{quantize,dequantize,requantize,quantized_batch_norm,quantized_concat,quantized_elemwise_add,quantized_rnn}-inl.h` and `.cc`
- `src/operator/rnn-inl.h`, `src/operator/rnn.cc`
- `src/ndarray/ndarray.cc` (oneDNN-related diff)
- `src/imperative/cuda_graphs.h`

---

## Critical bugs
*(things that can cause crashes, silent corruption, or wrong numerical results)*

### B1: masked_softmax inverts the (mask-1)*INF computation under v3
- **File**: `src/operator/nn/dnnl/dnnl_masked_softmax.cc:166-183`
- **Code**:
  ```cpp
  // 1. B) out = out * inf
  ...
  // v3: set_output_scales(mask, scales) is gone. For reorder we use
  //     set_scales_mask(DNNL_ARG_DST, 0) and feed the scale as a runtime
  //     memory arg DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST.
  dnnl::primitive_attr attr;
  attr.set_scales_mask(DNNL_ARG_DST, 0);
  ...
  float scale_value = mshadow::red::limits::MaxValue<float>();
  ```
- **Issue**: In v3, the scale bound to `DNNL_ARG_DST` of a reorder *divides* the
  destination (i.e., `dst = src / scale`). The author of the FC subgraph and
  every other v3 site in this repo explicitly notes this and uses
  `DNNL_ARG_SRC` when they want a multiplicative factor — see
  `src/operator/subgraph/dnnl/dnnl_fc.cc:289-290` ("v3 reorder: DNNL_ARG_DST
  divides; use DNNL_ARG_SRC to multiply by u8_to_s8_scale"), and the
  quantize / dequantize / requantize / quantized_elemwise_add / quantized
  batch-norm-concat sites which all bind to `DNNL_ARG_SRC`. The intent here is
  the v2 `(mask - 1) * MAX_FLOAT` step that produces `-inf` for masked-out
  positions; with `DNNL_ARG_DST` the result is `(mask - 1) / MAX_FLOAT ≈ 0`
  instead, so the additive mask becomes a no-op and `softmax` no longer zeros
  the masked positions.
- **Impact**: Masked-softmax (used by transformer attention masks) produces
  wrong numerical output; masked positions are *not* suppressed.
- **Suggested fix**: Change `set_scales_mask(DNNL_ARG_DST, 0)` to
  `set_scales_mask(DNNL_ARG_SRC, 0)` and bind the runtime arg under
  `DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC`. Update the inline comment too.

### B2: activation Backward never re-initializes `param_.slope`
- **File**: `src/operator/nn/dnnl/dnnl_act.cc:250-293`
- **Code**:
  ```cpp
  void DNNLActivationBackward(...) {
    ...
    const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
    ...
    DNNLActParam param_;
    param_.alg = GetDNNLActAlgo(param);
    // <-- no slope assignment for SoftReLU / LogSigmoid here
  ```
- **Issue**: `DNNLActParam::slope` defaults to 0.f (see
  `dnnl_act-inl.h:41`). The Forward path (`dnnl_act.cc:158-167`) sets
  `slope = 1.0f` for SoftReLU and `-1.0f` for LogSigmoid because the v3
  `eltwise_soft_relu(alpha)` formula is `log(1 + exp(alpha*x))/alpha` — alpha=0
  is a divide-by-zero. The Backward path forgets to do the same, so for
  SoftReLU/LogSigmoid backward the primitive is constructed with alpha=0 and
  the v3 backward op either segfaults inside the kernel or produces NaN/inf
  gradients.
- **Impact**: NaN gradients (or crash) for SoftReLU / LogSigmoid training.
- **Suggested fix**: Mirror the Forward block — add
  ```cpp
  if (param.act_type == activation::kSoftReLU)      param_.slope = 1.0f;
  else if (param.act_type == activation::kLogSigmoid) param_.slope = -1.0f;
  ```

### B3: `DNNLRnnMemMgr::Alloc` size_t underflow + always-false guard
- **File**: `src/operator/nn/dnnl/dnnl_rnn.cc:177-198`
- **Code**:
  ```cpp
  curr_size -= (md.get_size() + padding);
  if (curr_size < 0) {                          // size_t is unsigned!
    ret.reset(new dnnl::memory(md, cpu_engine));
  } else {
    curr_mem += (md.get_size() + padding);
    ret.reset(new dnnl::memory(md, cpu_engine, reinterpret_cast<void*>(addr)));
  }
  ```
- **Issue**: `curr_size` is `size_t` (declared in `dnnl_rnn-inl.h:129`). The
  decrement underflows silently when the request exceeds the remaining
  workspace, producing a huge value. The subsequent `curr_size < 0` check is
  always false for an unsigned type. Then the code happily increments
  `curr_mem` past the end of the buffer and hands out an in-bounds-looking
  pointer that aliases / overruns subsequent allocations.
- **Impact**: Out-of-bounds write into the RNN workspace; heap corruption when
  the workspace runs out (e.g. unusually large batch / seq-len / state-size
  combinations, or unfused multi-layer paths).
- **Suggested fix**: Compare with `if (md.get_size() + padding > curr_size)`
  **before** the subtraction, or cast the comparison to a signed type
  explicitly.

### B4: subgraph conv leaves `cached_bias_min_/max_` unwired
- **File**: `src/operator/subgraph/dnnl/dnnl_conv.cc:114-116, 287-303`
- **Code**:
  ```cpp
  // TODO: wire from quantization inputs (see dnnl_fc.cc kBiasMin/kBiasMax)
  float cached_bias_min_{0.0f};
  float cached_bias_max_{0.0f};
  ...
  if (has_bias && cached_bias_.dtype() == mshadow::kInt8) {
    const float bias_abs_max = std::max(std::abs(cached_bias_min_),
                                        std::abs(cached_bias_max_));
    if (bias_abs_max > 0) {
      // bias-scale cap …
    }
  }
  ```
- **Issue**: `cached_bias_min_` and `cached_bias_max_` are never assigned from
  the quantization input min/max tensors (the FC subgraph reads them via
  `kBiasMin/kBiasMax`; conv subgraph has no equivalent wiring). They stay at
  the default `0.0f`, so `bias_abs_max == 0` and the entire bias-overflow
  guard (the "Bias-overflow guard" block patched in by the recent port) is
  silently skipped for int8-bias conv. Then `weight_scales_[c]` can produce a
  saturated int32 bias.
- **Impact**: Quantized conv with int8 bias re-introduces the int32-bias
  overflow the recent commit `09055f014` was supposed to fix (same root
  cause as the FC patch).
- **Suggested fix**: Plumb bias min/max into the conv subgraph the same way
  FC does (or, if int8 bias is rare here, drop the guard and assert
  bias_dtype != int8).

### B5: layer_norm backward unconditionally overwrites grad outputs even when `req == kNullOp`
- **File**: `src/operator/nn/dnnl/dnnl_layer_norm.cc:180-219`
- **Code**:
  ```cpp
  ...
  if (req[layernorm::kBwdGammaGrad] == kAddTo) {
    memcpy(diff_scale_mem.get_data_handle(),
           outputs[layernorm::kBwdGammaGrad].data().dptr_, bytes);
    memcpy(diff_shift_mem.get_data_handle(),
           outputs[layernorm::kBwdBetaGrad].data().dptr_, bytes);
  }
  ...
  // unconditional copy back
  memcpy(outputs[layernorm::kBwdGammaGrad].data().dptr_,
         diff_scale_mem.get_data_handle(), bytes);
  memcpy(outputs[layernorm::kBwdBetaGrad].data().dptr_,
         diff_shift_mem.get_data_handle(), bytes);
  ```
- **Issue**: The kAddTo branch correctly seeds `diff_scale/shift_mem` with the
  existing output values so the v3 primitive can accumulate into them. But
  the trailing `memcpy` is unconditional — when `req[BwdGammaGrad]` is
  `kNullOp` (caller doesn't want the gradient), the diff_*_mem buffer is
  garbage and we still stomp the output. The pre-v3 code at least went
  through `CommitOutput` which respects `kNullOp` (now removed by the
  scale/shift split rewrite).
- **Impact**: Garbage written into Gamma/Beta gradient buffers when caller
  expected them untouched; also wrong when `req == kAddTo` if the v3
  primitive's `DIFF_SHIFT` output desc differs from `diff_weights_desc(0)`
  (see also F1 below).
- **Suggested fix**: Wrap the final `memcpy` calls in
  `if (req[BwdGammaGrad] == kWriteTo || kWriteInplace || kAddTo)`; for
  `kAddTo`, replace `memcpy` with a `ParallelAdd`-style accumulate
  (otherwise the seeded values are overwritten, not added).

### B6: RNN destructor freed-pointer state vs guard mismatch
- **File**: `src/operator/rnn-inl.h:685-716` and `:1349-1386`
- **Issue**: The RNN `RNNOp` constructor only initialises `init_space_`,
  `temp_init_space_`, `reserve_cpu_space_size_`, `temp_cpu_space_size_` when
  `ctx_.dev_type == kCPU` (line 673-682). When the user constructs `RNNOp`
  on GPU and the cuDNN init path bails partway, the destructor at lines
  685-716 takes the `kGPU` branch and unconditionally calls
  `cudnnDestroyTensorDescriptor` on the (created-at-construct-time)
  descriptors. That part is fine. But `cudnnDestroyRNNDataDescriptor` at
  lines 710-713 is also unconditional and only safe if the constructor's
  `cudnnCreateRNNDataDescriptor` calls (lines 662-665) succeeded. If they
  threw mid-way (e.g. OOM in cuDNN), the descriptor handles are garbage
  pointers and destroy will crash.
- **Impact**: Crash on constructor-failure path (rare but a real
  double-fault).
- **Suggested fix**: Either initialize the four `*_data_desc_` handles to
  `nullptr` and guard each destroy, or move the
  `cudnnCreateRNNDataDescriptor` calls into a try/catch that destroys what
  was created so far.

### B7: `same_shape` on `desc.get_dims().data()` is undefined behavior
- **File**: `src/operator/nn/dnnl/dnnl_base-inl.h:639-642`
- **Code**:
  ```cpp
  inline bool same_shape(const mxnet::TShape& shape, int dtype, const dnnl::memory::desc& desc) {
    return same_shape(shape, desc.get_dims().data(), desc.get_ndims()) &&
           get_dnnl_type(dtype) == desc.get_data_type();
  }
  ```
- **Issue**: `dnnl::memory::desc::get_dims()` returns a `std::vector<dim>` *by
  value* in v3. `.data()` returns a pointer to a buffer that is destroyed at
  the end of the full expression. The C-style array overload of `same_shape`
  (lines 621-628) reads through this pointer, which is dangling by then. The
  expression *happens* to work today on common stdlib + ABIs because the
  temporary's storage is still untouched by the time the read happens, but
  the language standard makes this UB and any future SSO / inline-buffer
  layout change can break it silently. The author called out this exact
  pitfall in commits `d0ff77104` and `9f3a77d28` for other call sites but
  this one slipped through.
- **Impact**: Latent UB; could randomly mis-compare shapes on the next
  toolchain bump.
- **Suggested fix**: Store the temporary first, e.g.
  ```cpp
  const auto dims = desc.get_dims();
  return same_shape(shape, dims.data(), desc.get_ndims()) && ...;
  ```

### B8: tautological CHECKs in `dnnl_base.cc`
- **File**: `src/operator/nn/dnnl/dnnl_base.cc:45, :98`
- **Code**:
  ```cpp
  CHECK(input_pds[0] == input_pds[0]);   // line 45
  ...
  CHECK_EQ(mem, mem);                    // line 98
  ```
- **Issue**: Both checks are always true. They are leftover debug aids; if
  they were ever meant to compare two *different* operands (`input_pds[0]
  == input_pds[1]`? `mem == orig_mem`?) the original intent has been lost.
- **Impact**: Dead checks — they don't catch the condition they were
  presumably meant to catch; cost is small but they signal stale code.
- **Suggested fix**: Either delete them, or restore the intended
  cross-operand comparison.

---

## Likely-correct-but-fragile
*(undefined-but-currently-benign, lifetime hazards, missing guards)*

### F1: layer_norm backward uses `diff_weights_desc(0)` for both DIFF_SCALE and DIFF_SHIFT
- **File**: `src/operator/nn/dnnl/dnnl_layer_norm.cc:191-193`
- **Code**:
  ```cpp
  auto diff_weights_md = bwd_pd->diff_weights_desc();   // == diff_weights_desc(0)
  auto diff_scale_mem  = dnnl::memory(diff_weights_md, cpu_engine);
  auto diff_shift_mem  = dnnl::memory(diff_weights_md, cpu_engine);
  ```
- **Issue**: For a v3 layer-norm bwd PD with split scale/shift, `diff_weights_desc(0)` is the scale-grad desc and `diff_weights_desc(1)` is the shift-grad desc. They are both 1-D `{C}` so in practice the same `dnnl::memory::desc` works for both today, but if oneDNN ever pads or strides them differently the shift memory is wrong size.
- **Suggested fix**: Use `bwd_pd->diff_weights_desc(0)` and
  `bwd_pd->diff_weights_desc(1)` explicitly — even if equal today, it
  documents intent and is robust against future API changes.

### F2: `GetFCWeightDesc` carries an unused `batch_size` parameter
- **File**: `src/operator/nn/dnnl/dnnl_base-inl.h:378-396`, callers at
  `src/operator/nn/dnnl/dnnl_fully_connected.cc:43-44, 111, 126`
- **Issue**: `batch_size` is declared but never read inside
  `GetFCWeightDesc`. Every caller passes `data.shape()[0]`, which is the
  batch size, but the function silently ignores it. Either the parameter
  should be removed, or — if it's supposed to affect the layout — the
  intended logic is missing.
- **Suggested fix**: If unused, drop the parameter. If a v3 weight layout
  change was supposed to depend on batch size, document and implement it.

### F3: `cuda_graphs.h` Update assumes CUDA 12 API form unconditionally
- **File**: `src/imperative/cuda_graphs.h:202-214`
- **Issue**: The comment says "the CUDA runtime ships a backwards-compat
  inline wrapper for the old form" — true for headers, but **only** for
  CUDA 12.0+. On 11.x toolkits this file no longer compiles. The cuDNN-9
  / CUDA-13 port intends to drop 11.x support, but there is no
  `CUDART_VERSION` guard anywhere in this header. A user who keeps 11.x in
  their build setup will get a confusing error.
- **Suggested fix**: Add a `#if CUDART_VERSION >= 12000` guard around the new
  shape (or just a `STATIC_ASSERT_CUDA_VERSION_GE(12000)` at the top of the
  header).

### F4: Many uninitialized `cached_*_min_/max_` and `weight_ver_`/`bias_ver_` in subgraph FC/transformer
- **Files**:
  - `src/operator/subgraph/dnnl/dnnl_fc.cc:96-107`
  - `src/operator/subgraph/dnnl/dnnl_transformer.cc:203-208`
- **Code** (FC):
  ```cpp
  float cached_data_min_;
  float cached_data_max_;
  float cached_weight_min_;
  float cached_weight_max_;
  float cached_sum_min_;
  float cached_sum_max_;
  float cached_bias_min_;
  float cached_bias_max_;
  size_t weight_ver_;
  size_t bias_ver_;
  float cached_output_min_;
  float cached_output_max_;
  ```
- **Issue**: None of these have default-member-initializers. They're set on
  the "first call" path that runs `Forward(...)`, but `cached_output_min_/
  max_` are *read* unconditionally at line 248-249 even when `Forward` was
  never called (e.g. if `dnnl_param.quantized` flips between init and run
  via env-var, or in cached-op re-use). `weight_ver_`/`bias_ver_` are read
  at lines 174, 181, 247 with no first-init guard.
- **Impact**: Garbage `output_min_/max_` written to op outputs in edge
  cases; spurious "not initialized" triggers if version-comparison happens
  before the first real init.
- **Suggested fix**: Add `{0.0f}` / `{0}` default initializers to every
  member; mirrors what `dnnl_conv.cc` partially does for
  `cached_bias_min_/max_`.

### F5: Storage::Handle.dptr read before any allocation
- **File**: `src/operator/rnn-inl.h:705-708`
- **Code**:
  ```cpp
  if (init_cudnn_) {
    init_cudnn_ = false;
    Storage::Get()->Free(reserve_space_);
    if (dev_seq_lengths_.dptr != nullptr) {
      Storage::Get()->Free(dev_seq_lengths_);
  ```
- **Issue**: `reserve_space_` is `Storage::Handle{}`-default-initialized
  (good — `dptr` is nullptr) but `Storage::Get()->Free(handle)` on a
  zero-sized nullptr handle is *almost* certainly a no-op, *but* the API
  contract isn't documented; if any storage backend asserts on `dptr !=
  nullptr` this crashes. The `dev_seq_lengths_` path correctly guards.
- **Suggested fix**: Apply the same `dptr != nullptr` guard to
  `reserve_space_`.

### F6: `Storage::Handle` move semantics across RNN destructor
- **File**: `src/operator/rnn-inl.h:1313`
- **Issue**: `dev_seq_lengths_.dptr = nullptr;` is written inside the cuDNN
  init block to declare "no allocation yet" — but the rest of the handle
  (size, ctx, profiler info) remains default. If `EnsureDevSeqLengthsBuffer`
  is later called with `s` on a different device, the stale `ctx` field is
  used as the comparison key (none of the logic looks at it, but it's a
  fragility).

### F7: Inefficient `get_dims()` calls inside hot loops
- **File**: `src/ndarray/ndarray.cc:681-683, 707-709, 735-737`
- **Code**:
  ```cpp
  for (int i = 0; i < new_desc.get_ndims(); i++)
    required_shape[i] = new_desc.get_dims()[i];
  ```
- **Issue**: Each `get_dims()` allocates and returns a fresh `std::vector`.
  For an 8-D tensor that's 8 heap allocations per call. Not UB but
  needlessly expensive on the hot reshape path.
- **Suggested fix**: Hoist `const auto dims = new_desc.get_dims();` before
  the loop and index `dims[i]`.

### F8: `eltwise_soft_relu` alpha=0 still reachable from generic eltwise path
- **File**: `src/operator/nn/dnnl/dnnl_eltwise.cc:58-67`
- **Issue**: `DNNLEltwiseFwd` always passes `alpha = 0.f, beta = 0.f`. The
  current callers (templated on `DNNLAlgorithm<OP>::value` —
  `elemwise_unary_op.h:472-502`) only instantiate it for tanh, exp,
  square, sqrt, plus, minus, mul, div, none of which need a nonzero
  alpha. If anyone ever adds `soft_relu` or `log_sigmoid` to that switch,
  alpha=0 will silently divide by zero.
- **Suggested fix**: Add `CHECK_NE(algorithm, dnnl::algorithm::eltwise_soft_relu)` or
  thread an explicit `alpha`/`beta` parameter through `DNNLEltwiseFwd`.

### F9: `req.size() < deconv::kBias && req[deconv::kBias]` looks like an off-by-one inversion
- **File**: `src/operator/nn/dnnl/dnnl_deconvolution-inl.h:232`
- **Code**:
  ```cpp
  if (req[deconv::kWeight] || (req.size() < deconv::kBias && req[deconv::kBias])) {
  ```
- **Issue**: The compound `req.size() < deconv::kBias && req[deconv::kBias]`
  reads "size is too small AND access element at that out-of-range index".
  If `req.size() < kBias`, the second clause is OOB read. Almost certainly
  the author meant `req.size() > deconv::kBias`. This is pre-existing
  code (git blame goes back well before the port) — flagging because the
  v3 port surrounds it but didn't touch it.
- **Suggested fix**: Verify intent; if guarding "bias gradient requested",
  use `req.size() > deconv::kBias && req[deconv::kBias]`.

### F10: SgDNNLConvOperator reads `cached_data_min_/max_/sum_min/max` uninitialized in version-check
- **File**: `src/operator/subgraph/dnnl/dnnl_conv.cc:117-124, 211-218`
- **Issue**: Same class of issue as F4: `cached_data_min_`,
  `cached_data_max_`, `cached_sum_min_`, `cached_sum_max_`,
  `cached_output_min_/max_`, `weight_ver_`, `bias_ver_` are not
  default-initialized. The first-pass branch at line 211 reads them before
  they've been set if the first call has `initialized_ == true` (which
  shouldn't happen for a freshly constructed op, but could after
  serialization/cached-op-reuse).
- **Suggested fix**: Default-initialize all of them.

### F11: `SgDNNLSelfAttQKOp::Forward` reads `min_output_/max_output_` even outside the quantized path
- **File**: `src/operator/subgraph/dnnl/dnnl_transformer.cc:203-208, 380-385`
- **Issue**: `min_output_/max_output_` are only assigned inside the
  `param_.quantized` branch of `Initialize` (lines 285-311). In `Forward`
  the read at lines 383-384 is guarded by `param_.quantized &&
  !enabled_float_output.has_value()`, so currently safe — but the members
  are still uninitialized in the non-quantized path. A future use that
  drops the guard would expose garbage.
- **Suggested fix**: Default-initialize `min_output_{0.f}` and
  `max_output_{0.f}`.

### F12: `set_rnn_weights_qparams` mask hex magic
- **File**: `src/operator/quantization/dnnl/dnnl_quantized_rnn.cc:224`
- **Code**:
  ```cpp
  rnn_attr_->set_rnn_weights_qparams(0 + (1 << 3) + (1 << 4), ...);
  ```
- **Issue**: Magic number `0 + (1 << 3) + (1 << 4)` — the bits encode "per-OC
  on dims 3 and 4" (gates and output channel). Acceptable as-is but the
  numerics aren't self-documenting; if oneDNN ever renumbers the axes the
  mask is silently wrong.
- **Suggested fix**: Use a named constant or compute the mask from the
  weight tensor's `ldigo` layout.

---

## Code smells / maintainability

### S1: Stale "v2.6" oneDNN doc URLs in comments
Every `bool SupportDNNL...` function still references `oneDNN/v2.6/dev_guide_...`
in its docstring even though the codebase now targets v3. Locations:
- `dnnl_act.cc:51, 61, 66`
- `dnnl_batch_dot.cc:39`
- `dnnl_convolution.cc:40`
- `dnnl_deconvolution.cc:31`
- `dnnl_eltwise.cc:31`
- `dnnl_layer_norm.cc:32`
- `dnnl_masked_softmax.cc:31`
- `dnnl_rnn-inl.h:565`
- `dnnl_softmax.cc:32`
- `dnnl_softmax_output.cc:95`
- `dnnl_base-inl.h` (`SupportDNNL` helpers don't reference docs but ought to)

Pure documentation drift — should bulk-update to `v3.x/` once the port
stabilizes.

### S2: Twelve TODO/FIXME/XXX/HACK markers across the recently-touched DNNL code
- `dnnl_act.cc:259` — `XXX: for y = relu(x), y is passed as "in_data" to Backward()`
- `dnnl_convolution.cc:375, 411, 498` — three TODO(zhennan/zhengda) about
  caching and kvstore reordering
- `dnnl_base-inl.h:61` — `TODO(PawelGlomski-Intel): add bfloat16 for quantized ops`
- `dnnl_base-inl.h:222` — `TODO(alex): MXNET-1075 cache size`
- `dnnl_base.cc:470` — `TODO(zhengda) we should use temp space to save the converted data.`
- `dnnl_lrn-inl.h:40, 257` — two LRN TODOs (lrn_within_channel core dump,
  nchw8c in_grad bug)
- `dnnl_deconvolution.cc:130` — kvstore TODO
- `dnnl_common.h:70` — `TODO(zhennan): dnnl has bug to handle INT_MAX in bias`
- `dnnl_conv_property.h:135` — `TODO(zhennan): doesn't support int8 conv+sum+relu6`
- `dnnl_fc-inl.h:63` — `TODO(ciyong): some alg doesn't support int8 so far.`
- `dnnl_fc.cc:255, 437, 974` — three TODOs around fallback inplace, INT_MAX
  bias, GluonCV INT8 temp solution
- `dnnl_conv.cc:114, 168, 203, 838` — four TODOs, two of which are the
  ones called out in B4 above
- `rnn-inl.h:794, 1157` — two TODOs in cuDNN RNN paths

### S3: `std::vector<float> output_scales = {0.0f}` ambiguity
- **File**: `src/operator/nn/dnnl/dnnl_fully_connected-inl.h:91-97`
- **Issue**: `output_scales = {0.0f}` is "a single-element vector
  containing zero" — but everywhere it's checked (`output_scales.size()`)
  the code treats `size == 0` as "no scales" and `size >= 1` as "valid
  scale present". So the default is *technically* a valid scale of 0 that
  would multiply outputs to zero. Caller is expected to overwrite before
  use; if anything ever passes an init-only path without overwriting, the
  output is silenced.
- **Suggested fix**: Initialize as `std::vector<float> output_scales = {}`
  (empty) and let size==0 be the sentinel. Or document the semantics in a
  comment.

### S4: Magic factor `u8_to_s8_scale = 0.5` used in two places
- **Files**: `src/operator/subgraph/dnnl/dnnl_fc.cc:70`,
  `src/operator/quantization/dnnl/dnnl_quantized_elemwise_add.cc:210`
- **Issue**: The "rescale uint8 → int8" factor of 0.5 is hardcoded as a
  named-but-still-magic constant in two distinct sites. If quantization
  range definitions ever change (e.g. asymmetric u8), both will need
  manual updates and the value won't survive scrutiny.

### S5: `cached_output_min_/max_` unconditionally written even on `kNullOp` output
- **File**: `src/operator/subgraph/dnnl/dnnl_fc.cc:244-250`,
  `src/operator/subgraph/dnnl/dnnl_conv.cc:445-448`
- **Issue**: When `req[1]/req[2]` for the min/max outputs is `kNullOp`,
  these still write into `outputs[1]/outputs[2].data().dptr<float>()[0]`.
  The downstream might not have allocated memory. Minor and likely never
  hit in practice since the outputs are explicitly declared.

### S6: Debug-fprintf left in production code paths
- **File**: `src/operator/subgraph/dnnl/dnnl_fc.cc:510-517`,
  `src/operator/subgraph/dnnl/dnnl_conv.cc:278-286`
- **Issue**: Two `if (dmlc::GetEnv("MX_FC_DBG", 0)) std::fprintf(stderr, ...)`
  blocks (and one `MX_CONV_DBG` equivalent). These are debug aids checked
  in alongside the recent commit — useful but should be removed (or
  routed through `LOG(DEBUG)`) before this lands on master.

### S7: `output_scale = data_scale_ * weight_scales_[0]` after `weight_scales_.resize(1)` can read uninitialized
- **File**: `src/operator/subgraph/dnnl/dnnl_conv.cc:354-355`
- **Issue**: `weight_scales_.resize(1)` may shrink-OK, but `weight_scales_[0]`
  was previously populated by `GetWeightScales` so this is fine — *if*
  weight_channelwise_scale was false, in which case
  `GetWeightScales` returned a single-element vector. If
  `weight_channelwise_scale` was true and we hit this branch, `[0]` is
  defined but represents only one channel's scale, used as if it were the
  whole-tensor scale.
- **Suggested fix**: Add a `CHECK_EQ(weight_scales_.size(), 1)` before the
  resize, or document why dropping per-channel info is OK here.

### S8: `MaxValue<int32_t>() / 2` hardcoded bias safety constant
- **File**: `src/operator/subgraph/dnnl/dnnl_fc.cc:440`,
  `src/operator/subgraph/dnnl/dnnl_conv.cc:294`
- **Issue**: `MaxValue<int32_t>() / 2` and `int32_t::max() / 2.0f` are the
  bias-overflow cap. The "/2" margin is folklore-knowledge; the original
  oneDNN bug it works around isn't cited in either site. The comment in
  `dnnl_common.h:70` references this as a long-standing oneDNN limitation
  but it's worth a single source-of-truth named constant
  (`kInt32BiasSafetyCap` or similar) shared by both sites.

### S9: `out_scale = data_scale_ * weight_scales_[0]; full_conv_param.requantize_scales.resize(0)` followed by use
- **File**: `src/operator/subgraph/dnnl/dnnl_conv.cc:354-356`
- **Issue**: We compute `output_scale` then immediately empty
  `requantize_scales` — and at the with_sum branch below at line 365 we
  check `if (full_conv_param.requantize_scales.size() == 1)` which is
  false (size is 0), so `sum_scale` is not divided. That's correct because
  there's no DST-scale active. The control flow works but is brittle —
  one extra refactor and the semantic link between "no requantize" and
  "no DST scale division" disappears.

### S10: `tz_volume` static lambda inside `SetDims` doesn't need to be static
- **File**: `src/operator/nn/dnnl/dnnl_rnn.cc:79-84`
- **Issue**: `static auto tz_volume = [](...)` creates a thread-safe
  one-time-initialised static; pure overhead for a pure function with no
  state. Just declare `auto tz_volume = [](...) ` (no `static`).

### S11: Many oneDNN port comments include "(WIP)" or "TODO" markers in commit messages but not code
The recent commit history mentions `(WIP)` for many port commits but the
code mostly has fully-functional rewrites. The `TODO` in
`src/operator/subgraph/dnnl/dnnl_conv.cc:114` is the one outstanding live
TODO that materially impacts correctness (see B4).

---

## Summary
- **8 critical bugs (B1–B8)**: B1 (masked-softmax MAX_FLOAT inversion),
  B2 (act backward slope), B3 (RNN MemMgr underflow), B4 (conv
  cached_bias_min/max not wired), B5 (layer-norm bwd unconditional copy),
  B6 (RNN destructor partial-init), B7 (same_shape get_dims() UB), B8
  (tautological CHECKs).
- **12 fragile patterns (F1–F12)**: layout-assumption fragility, missing
  default-initializers, inefficient temporaries, CUDA-12-only API,
  off-by-one near deconv bias.
- **11 code smells (S1–S11)**: stale v2.6 doc URLs, 20+ TODOs, magic
  numbers (0.5, INT32_MAX/2, mask=0x18), debug fprintf left in,
  `std::vector<float> = {0.0f}` ambiguity.

**Highest-priority items to fix first:**
1. **B1** — `dnnl_masked_softmax.cc:173` — wrong scale direction; masked
   attention is numerically broken on every quantized + temperature path
   that uses it.
2. **B2** — `dnnl_act.cc:264` — SoftReLU/LogSigmoid backward divides by
   zero; training NaNs.
3. **B3** — `dnnl_rnn.cc:189-190` — `size_t` underflow + dead `< 0`
   check → heap corruption when RNN workspace overflows.

Honorable mention: **B4** (silently regresses the v3 port's
just-landed conv int8-bias overflow fix because the cached min/max are
TODO-not-wired).
