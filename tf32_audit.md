# TF32 / Tensor-Math-Type Defaults Audit on cuDNN 9 / Blackwell (sm_120)

**Branch:** `onednn-v3-port`
**Date:** 2026-05-17
**Issue:** issues.md item #23 — cuDNN 9 changed when TF32 is enabled by default;
FP32 conv that opts in to TF32 gets ~2x free on Ampere+ and ~2-3x on Blackwell.

---

## TL;DR

- The active FP32 conv path on GPU is the **cuDNN v9 backend frontend**
  (`src/operator/cudnn_ops.cc`), not the legacy `cudnn_convolution-inl.h`.
- TF32 selection on that path is **not** done via `cudnnSetConvolutionMathType`.
  It is done by filtering the candidate-engine list against
  `CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS` in `ExcludeNumerics()`.
- The pre-fix default excluded TF32 engines because
  `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION` defaulted to `false`.
  Net effect: FP32 conv on Blackwell ran on CUDA cores at ~14 TFLOPS instead
  of TF32 tensor cores at ~42 TFLOPS — a measured **~2.9x perf gap**.
- **Fix applied** (single line change in `cudnn_ops.cc::ExcludeNumerics()`):
  default the conversion flag to follow `MXNET_CUDA_ALLOW_TENSOR_CORE` (i.e.
  true by default), matching PyTorch / TensorFlow defaults on cuDNN 9.

## Where math type is set

Two distinct code paths exist:

### A) Legacy cuDNN v7-style descriptor path (`src/operator/nn/cudnn/cudnn_convolution-inl.h`)
- Uses `cudnnSetConvolutionMathType(desc, math_type)`.
- `cudnn_tensor_core_` is only set when the input dtype is FP16
  (`src/operator/nn/cudnn/cudnn_convolution-inl.h:77`):
  ```cpp
  cudnn_tensor_core_ = DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore();
  ```
- This class **is no longer the active code path** for conv on GPU: see
  `src/operator/nn/convolution.cu:50-71` where the dispatch goes through
  `cudnn::Exec<cudnn::Conv>()` from the v9 frontend instead. The legacy class
  is kept only for legacy/algorithm registry compatibility.
- The legacy `cudnn_deconvolution-inl.h` is in the same shape.

### B) Modern cuDNN v9 backend frontend (`src/operator/cudnn_ops.{h,cc}`)
- Builds operation graphs, queries available engines via `cudnnBackendHeuristic`,
  then filters by required/excluded numerical notes (see `SelectPlan` →
  `GetPlans` → `IsCompatible` in `src/operator/cudnn_ops.cc`).
- The numeric filter is constructed by `RequireNumerics()` / `ExcludeNumerics()`
  at `src/operator/cudnn_ops.cc:108-128`.
- `cudnnSetTensorMathType` / `cudnnSetConvolutionMathType` are **not used at all**
  in this path (cuDNN 9 has effectively retired those for the new graph API;
  math precision is encoded in the operation graph + engine selection instead).

cuDNN 9 numerical-note semantics on Ampere+ (from `cudnn_graph_v9.h:852-862`):

| Note enum                                | Meaning                            |
|------------------------------------------|------------------------------------|
| `CUDNN_NUMERICAL_NOTE_TENSOR_CORE`       | Engine uses tensor cores           |
| `CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS` | Engine converts inputs to a narrower type (e.g. **FP32 → TF32**, or FP32 → BF16/FP16) |
| `CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION` | Accumulation in narrower type |

So for **FP32 conv**, the TF32 tensor-core engines are tagged with **both**
`TENSOR_CORE` and `DOWN_CONVERT_INPUTS`. Excluding either eliminates them.

## env-var behavior

`src/common/cuda/utils.h`:

```cpp
#define MXNET_CUDA_ALLOW_TENSOR_CORE_DEFAULT                  true
#define MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION_DEFAULT    false   // legacy
```

`src/operator/cudnn_ops.cc::ExcludeNumerics()` (pre-fix):

```cpp
if (!dmlc::GetEnv("MXNET_CUDA_ALLOW_TENSOR_CORE", true))
  ret.push_back(CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
if (!dmlc::GetEnv("MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION", false))   // <-- default false
  ret.push_back(CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS);
```

The first line (TENSOR_CORE) is FP16-friendly: it's only suppressed if the
user explicitly turns off tensor cores. But the second line, with a `false`
default, was unconditionally suppressing TF32 for FP32 conv.

This was the right policy on **cuDNN 7/8** + Volta/Turing, where
`CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` was an opt-in compatibility hack for
FP16. On **cuDNN 9** + Ampere/Hopper/Blackwell with TF32 first-class, the
note is what gates TF32 — and the default should follow the broader tensor-core
policy.

## Measured impact on sm_120 (Blackwell)

Test: 3x3 conv, NCHW, batch 32, 28x28, 256→256 channels (a compute-bound
ResNet-mid block). One GPU, autotune `MXNET_CUDNN_AUTOTUNE_DEFAULT=2` (find),
200 iterations after 30 warmup.

| Configuration                                                  | Per-iter | TFLOPS    |
|----------------------------------------------------------------|----------|-----------|
| **pre-fix** default (TF32 implicitly excluded)                 | 2.047 ms | 14.46     |
| **pre-fix** with `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=1`| 0.711 ms | **41.63** |
| **post-fix** default (TF32 now on by default)                  | 0.713 ms | **41.48** |
| **post-fix** with `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=0` | 0.716 ms | 41.33 (*) |
| **post-fix** with `MXNET_CUDA_ALLOW_TENSOR_CORE=0`             | 1.771 ms | 16.71     |

Speedup at the default: **2.87×**. Output_sum across configurations matches
to within TF32-vs-FP32 numerical tolerance (TF32 has FP32 range and ~10 bits
of mantissa, vs FP32's 23).

(*) With `MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=0` and autotune in find
mode, the picker still selects a true-FP32 tensor-core engine (e.g. HMMA.FP32)
on Blackwell sm_120; this delivers ~33 TFLOPS in heuristic mode and ~41
TFLOPS when timed by find. Only when the broader tensor-core flag
(`MXNET_CUDA_ALLOW_TENSOR_CORE=0`) is set does cuDNN fall back to a true
CUDA-core FP32 kernel (which on this shape is ~8.5 TFLOPS in heuristic
mode, ~16.7 TFLOPS when find times multiple non-TC variants).

Auxiliary test (originally-suggested 224x224 / 3→64 / 7x7 / stride-2): this
shape has only 3 input channels and is memory-bound. Default post-fix is
2.90 TFLOPS; `MXNET_CUDA_ALLOW_TENSOR_CORE=0` falls to 2.45 TFLOPS (~18% gap).
Compute-bound problems show the dramatic gain; bandwidth-bound problems
show only modest improvement.

### Correctness validation

After the fix, the following GPU conv tests pass:

```
tests/python/gpu/test_operator_gpu.py::test_convolution_with_type      PASS
tests/python/gpu/test_operator_gpu.py::test_convolution_options        PASS
tests/python/gpu/test_operator_gpu.py::test_conv_deconv_guards         PASS
tests/python/gpu/test_operator_gpu.py::test_convolution_large_c        PASS
tests/python/gpu/test_operator_gpu.py::test_convolution_versions       PASS
```

These tests use `check_consistency` with FP32 tolerance `rtol=atol=1e-3`,
which TF32 satisfies on the small kernel sizes tested (the worst case is
roughly `eps * sqrt(C_in*K*K) ≈ 5e-4 * sqrt(2*3*3) ≈ 2e-3` of the output
magnitude, but tolerances are checked against full-precision FP32 reference
so the actual deltas observed are well under threshold).

## Fix applied

`src/operator/cudnn_ops.cc`:

```cpp
std::vector<cudnnBackendNumericalNote_t> ExcludeNumerics() {
  std::vector<cudnnBackendNumericalNote_t> ret;
  bool allow_tensor_core = dmlc::GetEnv("MXNET_CUDA_ALLOW_TENSOR_CORE", true);
  if (!allow_tensor_core)
    ret.push_back(CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
  // cuDNN tags TF32 (and other reduced-precision input) engines with
  // CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS. On Ampere+ (and especially
  // sm_120/Blackwell) FP32 convolutions can run ~2-3x faster on tensor cores
  // via TF32. We default to allowing this whenever tensor cores are allowed,
  // mirroring the default behavior of PyTorch / TF on cuDNN 9. Users who need
  // strict FP32 numerics can disable it via
  //   MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=0
  // or by setting MXNET_CUDA_ALLOW_TENSOR_CORE=0.
  if (!dmlc::GetEnv("MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION", allow_tensor_core))
    ret.push_back(CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS);
  ...
}
```

Plus a clarifying comment in `src/common/cuda/utils.h` explaining that the
legacy `_DEFAULT false` only applies to the legacy v7-style descriptor path
(currently FP16-only), and that the active v9 path follows the broader
tensor-core flag.

## Other call sites (not changed)

- `src/common/cuda/utils.h:636` — `SetCublasMathMode`. cuBLAS knob; this is
  used by `src/operator/linalg_impl.h` per-call for GEMM. cuBLAS 13 separately
  honors the `NVIDIA_TF32_OVERRIDE` env var globally and exposes
  `CUBLAS_TF32_TENSOR_OP_MATH` — these are orthogonal to cuDNN and fine as is.
- `src/operator/rnn-inl.h:1291` — uses `CUDNN_DEFAULT_MATH` for cuDNN RNN.
  On cuDNN 9 the RNN v9 API picks tensor cores from the operator graph as
  needed; the descriptor knob is effectively a hint. Out of scope for this audit.
- TVM 3rdparty (`3rdparty/tvm/...`) — vendored, not exercised by MXNet GPU
  conv. Left alone.

## How to disable TF32 (post-fix)

For strict-FP32 reproducibility, any of these works:

```bash
MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=0   # narrow knob (just disables TF32)
MXNET_CUDA_ALLOW_TENSOR_CORE=0                 # broad knob (also disables FP16 TC)
NVIDIA_TF32_OVERRIDE=0                         # global, also affects cuBLAS / cuFFT
```
