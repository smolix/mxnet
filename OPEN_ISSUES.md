# Open issues

Known limitations and outstanding work in the **`smolix/mxnet`** fork. This is the
scannable index; deep context (root-cause analyses, reproducers, deferred-fix
rationale) for each item is in [`OPEN_ISSUES_DETAILS.md`](OPEN_ISSUES_DETAILS.md).
What has already been fixed is in [`FIXED.md`](FIXED.md).

Severity: **High** = can produce wrong results or block a common workflow ·
**Med** = perf / robustness / niche correctness · **Low** = cosmetic / informational.

## Start here — the two things most likely to bite you

> The old **CUDA 13.0 / R580** driver limitation (the `nvidia-cublas>=13.5` pin needs
> driver **R590+**) is now an accepted, documented platform constraint rather than open
> work — see [`FIXED.md`](FIXED.md) §1 (was OI-19). Upgrade the driver to R590+, or pin
> an older wheel for R580.

1. **Apple Silicon oneDNN INT8/fusion is gated off** — quantization + subgraph fusion
   fall back to native kernels on arm64; the `tests/python/dnnl` lane does not apply. ([OI-17](OPEN_ISSUES_DETAILS.md#oi-17))
2. **Composite-fusion QAT backward is broken** — simple INT8 FC and Conv(+ReLU)
   backward are validated and work, but a composite (Conv→…→Dense with a quantized conv
   output) crashes in `backward()` on a fused-subgraph type-inference conflict. ([OI-8](OPEN_ISSUES_DETAILS.md#oi-8))

## Correctness / numeric

| ID | Sev | Summary | Workaround |
|----|-----|---------|------------|
| [OI-4](OPEN_ISSUES_DETAILS.md#oi-4) | Low | NumPy *view-aliasing* gaps: positive-stepped slicing (`a[::2]`) and axis-moving (`moveaxis`/`rollaxis`) return copies, not views — PyTorch returns views (flip/rot90 already match PyTorch, which copies) | results are correct copies; only in-place aliasing differs |

> OI-1 (integer `dot`/`matmul`/`tensordot`), OI-2 (`np.mean(int)` dtype), and OI-3
> (fp16 `var`/`std` overflow) are **resolved** — see [`FIXED.md`](FIXED.md) §7.

## oneDNN INT8 quantization (CPU / x86)

| ID | Sev | Summary |
|----|-----|---------|
| [OI-5](OPEN_ISSUES_DETAILS.md#oi-5) | Med | Asymmetric quantize loses the sub-integer `shift` (oneDNN zero-points are integer) |
| [OI-6](OPEN_ISSUES_DETAILS.md#oi-6) | Med | Quantized concat / batch_norm affine fallback layout + u8→s8 roundtrip |
| [OI-7](OPEN_ISSUES_DETAILS.md#oi-7) | Med | uint8 requantize uses a CPU fallback; needs an asymmetric reorder |
| [OI-8](OPEN_ISSUES_DETAILS.md#oi-8) | Med | Composite-fusion QAT backward type-inference gap (simple FC/Conv INT8 backward validated & working) |

## Performance / refactor (deferred — perf only, results correct)

| ID | Sev | Summary |
|----|-----|---------|
| [OI-9](OPEN_ISSUES_DETAILS.md#oi-9) | Med | RNN re-issues cuDNN descriptors every forward (`use_sequence_length` path) |
| [OI-10](OPEN_ISSUES_DETAILS.md#oi-10) | Low | Proposal contrib ops do per-call `cudaMalloc`/`cudaFree` |
| [OI-11](OPEN_ISSUES_DETAILS.md#oi-11) | Med | `SetTBlob()` mutates via `const_cast` (latent thread-safety) |
| [OI-12](OPEN_ISSUES_DETAILS.md#oi-12) | Med | Eager per-op heap NDArray allocation (reverted; needs redesign) |
| [OI-13](OPEN_ISSUES_DETAILS.md#oi-13) | Low | GPU axis reductions ~35–40% BW headroom; CPU float64 mean single-threaded |
| [OI-14](OPEN_ISSUES_DETAILS.md#oi-14) | Med | Batch-size-1 inference slow on AVX2-only CPUs (IC=3 brgemm padding) |
| [OI-15](OPEN_ISSUES_DETAILS.md#oi-15) | Low | cuBLASLt follow-ups deferred (mshadow dot_engine, INT8, default-on flip) |
| [OI-16](OPEN_ISSUES_DETAILS.md#oi-16) | Low | CUDA Graphs: host-generator `kRandom` ops excluded from capture (need a device-resident offset) |

## Platform

| ID | Sev | Summary |
|----|-----|---------|
| [OI-17](OPEN_ISSUES_DETAILS.md#oi-17) | High | arm64: oneDNN INT8 + subgraph fusion gated off (Xbyak_aarch64 JIT unreliable) |

> OI-19 (cuBLAS≥13.5 / driver R590+) and OI-18 (bf16 emulated in fp32 on CPUs without
> AVX-512-BF16 — inherent to oneDNN v3 / the ISA, not a defect) are **accepted
> constraints**; OI-20 (cuDNN minor-version warning) is **fixed** — see
> [`FIXED.md`](FIXED.md) §1 (OI-19/20) and §11 (OI-18).

## Engine / concurrency

| ID | Sev | Summary |
|----|-----|---------|
| [OI-21](OPEN_ISSUES_DETAILS.md#oi-21) | Med | Rare long-running inference hang (A6) — instrumented, needs an aarch64 repro |
| [OI-22](OPEN_ISSUES_DETAILS.md#oi-22) | Low | `LazyAllocArray::Get()` lock-free read is a benign data race (needs C++20 atomics) |

> OI-23 (CUB global-reduce input aliasing) is closed **won't-fix** — the fast path is
> correct on fp16/fp32/fp64 despite the overlap, and a guard added to "fix" it regressed
> fp16. See [`FIXED.md`](FIXED.md) §11.

## Ecosystem / packaging / CI

| ID | Sev | Summary |
|----|-----|---------|
| [OI-24](OPEN_ISSUES_DETAILS.md#oi-24) | Med | Wheel publication is manual; no conda/system packaging or release automation |
| [OI-25](OPEN_ISSUES_DETAILS.md#oi-25) | Med | No CUDA build-matrix CI (Ada/Hopper/Blackwell + CUDA 12.x) |
| [OI-26](OPEN_ISSUES_DETAILS.md#oi-26) | Low | Downstreams unverified (GluonNLP/Sockeye/AutoGluon, ps-lite, Py3.13+, NumPy 2.x [op shape/axis-param rendering fixed], DLPack) |

> OI-27 (ONNX shipped in wheels) is **resolved** — the wheel now bundles the
> `mxnet.onnx` packages; `pip install "mxnet[onnx]"` pulls onnx. See [`FIXED.md`](FIXED.md) §2.

> Closed, no action planned: **D2L book compatibility** (all items resolved — `train_ch13`
> multi-GPU DeadKernel and the two book-side convergence gaps, was OI-28/29, `FIXED.md` §10);
> **OI-30** vendored dmlc/oneDNN build warnings (won't-patch by policy — they clear on a
> future submodule bump, `FIXED.md` §11).
