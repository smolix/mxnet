# Open issues

Known limitations and outstanding work in the **`smolix/mxnet`** fork. This is the
scannable index; deep context (root-cause analyses, reproducers, deferred-fix
rationale) for each item is in [`OPEN_ISSUES_DETAILS.md`](OPEN_ISSUES_DETAILS.md).
What has already been fixed is in [`FIXED.md`](FIXED.md).

Severity: **High** = can produce wrong results or block a common workflow ·
**Med** = perf / robustness / niche correctness · **Low** = cosmetic / informational.

Issue IDs (`OI-N`) are **stable** — once an item is resolved or closed it moves to
[`FIXED.md`](FIXED.md) and is dropped from this list, so the `OI-N` sequence here is
intentionally gappy and is **not** a count. **18 items are open** (OI-4–OI-17, OI-21,
OI-24–OI-26); any other ID in the OI-1…OI-30 range is fixed, an accepted constraint, or
won't-fix — look it up in [`FIXED.md`](FIXED.md).

## Start here — the two things most likely to bite you

1. **Apple Silicon oneDNN INT8/fusion is gated off** — quantization + subgraph fusion
   fall back to native kernels on arm64; the `tests/python/dnnl` lane does not apply. ([OI-17](OPEN_ISSUES_DETAILS.md#oi-17))
2. **Composite-fusion QAT backward is broken** — simple INT8 FC and Conv(+ReLU)
   backward are validated and work, but a composite (Conv→…→Dense with a quantized conv
   output) crashes in `backward()` on a fused-subgraph type-inference conflict. ([OI-8](OPEN_ISSUES_DETAILS.md#oi-8))

## Correctness / numeric

| ID | Sev | Summary | Workaround |
|----|-----|---------|------------|
| [OI-4](OPEN_ISSUES_DETAILS.md#oi-4) | Low | NumPy *view-aliasing* gaps: positive-stepped slicing (`a[::2]`) and axis-moving (`moveaxis`/`rollaxis`) return copies, not views — PyTorch returns views (flip/rot90 already match PyTorch, which copies) | results are correct copies; only in-place aliasing differs |

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
| [OI-9](OPEN_ISSUES_DETAILS.md#oi-9) | Med | RNN re-issues cuDNN descriptors / temp-size / clip every forward (no cross-call caching) |
| [OI-10](OPEN_ISSUES_DETAILS.md#oi-10) | Low | Proposal contrib ops do per-call `cudaMalloc`/`cudaFree` |
| [OI-11](OPEN_ISSUES_DETAILS.md#oi-11) | Med | `SetTBlob()` mutates via `const_cast` (latent thread-safety) |
| [OI-12](OPEN_ISSUES_DETAILS.md#oi-12) | Med | Eager per-op heap NDArray allocation (reverted; needs redesign) |
| [OI-13](OPEN_ISSUES_DETAILS.md#oi-13) | Low | GPU axis reductions ~35–40% BW headroom; CPU float64 mean single-threaded |
| [OI-14](OPEN_ISSUES_DETAILS.md#oi-14) | Med | Batch-size-1 inference on AVX2-only CPUs: brgemm IC-padding cliff (dispatch-gate mitigation shipped; upstream kernel fix deferred) |
| [OI-15](OPEN_ISSUES_DETAILS.md#oi-15) | Low | cuBLASLt follow-ups deferred (mshadow dot_engine, INT8, default-on flip) |
| [OI-16](OPEN_ISSUES_DETAILS.md#oi-16) | Low | CUDA Graphs: host-generator `kRandom` ops excluded from capture (need a device-resident offset) |

## Platform

| ID | Sev | Summary |
|----|-----|---------|
| [OI-17](OPEN_ISSUES_DETAILS.md#oi-17) | High | arm64: oneDNN INT8 + subgraph fusion gated off (Xbyak_aarch64 JIT unreliable) |

## Engine / concurrency

| ID | Sev | Summary |
|----|-----|---------|
| [OI-21](OPEN_ISSUES_DETAILS.md#oi-21) | Med | Rare long-running inference hang (A6) — instrumented, needs an aarch64 repro |

## Ecosystem / packaging / CI

| ID | Sev | Summary |
|----|-----|---------|
| [OI-24](OPEN_ISSUES_DETAILS.md#oi-24) | Med | CUDA build→test→tag→release now scripted (`tools/release_cuda_wheel.sh`), but still host-run (no CI runner); no conda/system packaging |
| [OI-25](OPEN_ISSUES_DETAILS.md#oi-25) | Med | No CUDA build-matrix CI (Ada/Hopper/Blackwell + CUDA 12.x) |
| [OI-26](OPEN_ISSUES_DETAILS.md#oi-26) | Low | Downstreams unverified (GluonNLP/Sockeye/AutoGluon, ps-lite, Py3.13+, NumPy 2.x [op shape/axis-param rendering fixed], DLPack) |
