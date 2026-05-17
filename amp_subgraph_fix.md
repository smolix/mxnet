# AMP Subgraph Fix — bf16→fp32 fallback for non-AVX512-BF16 CPUs

**Issue:** issues.md #8 — 6 `test_amp_subgraph.py` failures with `inner_product` / matmul
primitive-creation errors on AVX2 hosts (this Zen 2 EPYC 7B12 box lacks AVX-512 BF16).

## Before

All 6 AMP subgraph tests failed with oneDNN v3 "could not create a primitive descriptor" errors
when the AMP pass promoted weights and activations to bf16 and the fused ops (FC, Conv, QK/ValAtt
matmul) attempted to create inner_product / convolution / matmul primitives in bf16. oneDNN v3
removed the AVX2 bf16 software emulation that v2 provided.

| Test | Before |
|---|---|
| `test_amp_fc` | FAIL |
| `test_amp_conv` | FAIL |
| `test_amp_transformers` | FAIL |
| `test_amp_concat` | FAIL |
| `test_amp_fuse_with_branch` | FAIL |
| `test_amp_excluding_after_graph_pass` | FAIL |

## After

All 6 pass in 25s.

| Test | After |
|---|---|
| `test_amp_fc` | PASS |
| `test_amp_conv` | PASS |
| `test_amp_transformers` | PASS |
| `test_amp_concat` | PASS |
| `test_amp_fuse_with_branch` | PASS |
| `test_amp_excluding_after_graph_pass` | PASS |

Smoke regressions: FC subgraph 387/0/16 (unchanged), test_dnnl.py 97/0 (unchanged).

## Strategy

Four files were modified:

### `src/operator/nn/dnnl/dnnl_base-inl.h` (+45 LOC)

New helper `DNNLISASupportsLowpFloat(int dtype)`:

- Calls `dnnl::get_effective_cpu_isa()` at runtime.
- Returns `true` for bf16 only on ISAs with native bf16 kernels:
  `avx2_vnni_2`, `avx512_core_bf16`, `avx10_1_512`, `avx10_1_512_amx`,
  `avx10_1_512_amx_fp16`, `avx10_2_512`, `avx10_2_512_amx_2`.
- For fp16 a stricter subset is used (no `avx512_core_bf16`).
- Returns `true` unconditionally for all other dtypes (fp32, int8, etc.).

Note: oneDNN's cpu_isa enum is **not monotonic** — `avx2_vnni_2` (0x1f) has bf16 but sorts below
`avx512_core` (0x27) which does not, so an `isa >= threshold` comparison is wrong.
The explicit switch-case is the correct approach.

### `src/operator/nn/dnnl/dnnl_fully_connected.cc` (+58 LOC)

At the top of `DNNLFCForwardImpl`: if any input/output is bf16 AND
`DNNLISASupportsLowpFloat(kBfloat16)` returns false, promote all bf16 NDArrays to fp32 via
`nd.Reorder2DefaultFloatFormat()`, allocate fp32 output buffers, call recursively, then reorder
fp32 results back into the caller's bf16 output NDArrays via `ReorderTo`.

### `src/operator/subgraph/dnnl/dnnl_conv.cc` (+55 LOC)

Same pattern at the top of `SgDNNLConvOperator::Forward`.

### `src/operator/subgraph/dnnl/dnnl_transformer.cc` (+97 LOC)

Same pattern applied twice: once at the top of `SgDNNLSelfAttQKForward` (QK matmul) and once
at the top of `DNNLSelfAttValAttForward` (value-attention matmul).

## Important implementation notes

- **No `mshadow::DataType<T>::kFlag` ODR hazard.** All output NDArray allocations use
  `static_cast<int>(mshadow::kFloat32)` instead of `mshadow::DataType<float>::kFlag`.
  The latter is a non-inline `static const int` that would be odr-used by `emplace_back`'s
  perfect-forward reference, producing an undefined-symbol link error. The `static_cast`
  sidesteps this entirely.

- **`DNNLStream::Get()->Submit()` before reorder-back.** The recursive call may queue async
  reorders inside oneDNN's stream; we flush before reading the fp32 output into the bf16 dst.

- **ISA check is one-shot per call, not cached.** `dnnl::get_effective_cpu_isa()` is cheap
  (returns a cached integer). The overhead on the fast-path (ISA supports bf16, check returns
  false immediately because `any_bf16 == false`) is negligible.

## Caveats

- The fallback adds one extra bf16→fp32 reorder on input and one fp32→bf16 reorder on output
  per forward pass. On this AVX2 host this is acceptable since we're already doing fp32 compute.
  On a bf16-capable host (avx512_core_bf16 / amx) the entire fallback is bypassed.

- Backward pass through AMP subgraph ops is **not validated** (issues.md #5). The fallback
  only covers forward.

- Only bf16 is handled. fp16 AMP subgraph (if it existed) would need a symmetric fix; the
  `DNNLISASupportsLowpFloat` helper already handles fp16 ISA detection correctly.
