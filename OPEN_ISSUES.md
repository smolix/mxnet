# Open issues

Known limitations and outstanding work in the **`smolix/mxnet`** fork. This is the
scannable index; deep context (root-cause analyses, reproducers, deferred-fix
rationale) for each item is in [`OPEN_ISSUES_DETAILS.md`](OPEN_ISSUES_DETAILS.md).
What has already been fixed is in [`FIXED.md`](FIXED.md).

Severity: **High** = can produce wrong results or block a common workflow ·
**Med** = perf / robustness / niche correctness · **Low** = cosmetic / informational.

## Start here — the five things most likely to bite you

1. **No CUDA 13.0 / R580 driver support** — the wheel pins `nvidia-cublas>=13.5`,
   which needs driver **R590+**. On R580 large GEMMs fail with
   `CUBLAS_STATUS_NOT_INITIALIZED`. Upgrade the driver. ([OI-19](OPEN_ISSUES_DETAILS.md#oi-19))
2. **ONNX is not in the published wheels** — fixed in source (PR #38) but wheels are
   built ONNX-free; you need a source build to use it. ([OI-27](OPEN_ISSUES_DETAILS.md#oi-27))
3. **Apple Silicon oneDNN INT8/fusion is gated off** — quantization + subgraph fusion
   fall back to native kernels on arm64; the `tests/python/dnnl` lane does not apply. ([OI-17](OPEN_ISSUES_DETAILS.md#oi-17))
4. **bf16 on CPUs without AVX-512-BF16 is emulated in fp32** — numerically correct,
   no speedup. ([OI-18](OPEN_ISSUES_DETAILS.md#oi-18))
5. **Backward through quantized ops is unvalidated** — forward INT8 inference is
   solid; training through `_sg_onednn_*` is not verified. ([OI-8](OPEN_ISSUES_DETAILS.md#oi-8))

## Correctness / numeric

| ID | Sev | Summary | Workaround |
|----|-----|---------|------------|
| [OI-1](OPEN_ISSUES_DETAILS.md#oi-1) | Med | `matmul`/`dot`/`tensordot` reject integer inputs (NumPy/PyTorch compute them) | cast to float |
| [OI-2](OPEN_ISSUES_DETAILS.md#oi-2) | Med | `np.mean(int)` doesn't round-to-int like NumPy/PyTorch | cast result |
| [OI-3](OPEN_ISSUES_DETAILS.md#oi-3) | Med | `np.var`/`np.std(fp16)` *moments-with-custom-workspace* path can still overflow (simple path fixed) | use fp32 |
| [OI-4](OPEN_ISSUES_DETAILS.md#oi-4) | Med | 6–10 NumPy view/stride xfails (stepped slicing, negative-stride flip/rot/squeeze) | materialize a copy |

## oneDNN INT8 quantization (CPU / x86)

| ID | Sev | Summary |
|----|-----|---------|
| [OI-5](OPEN_ISSUES_DETAILS.md#oi-5) | Med | Asymmetric quantize loses the sub-integer `shift` (oneDNN zero-points are integer) |
| [OI-6](OPEN_ISSUES_DETAILS.md#oi-6) | Med | Quantized concat / batch_norm affine fallback layout + u8→s8 roundtrip |
| [OI-7](OPEN_ISSUES_DETAILS.md#oi-7) | Med | uint8 requantize uses a CPU fallback; needs an asymmetric reorder |
| [OI-8](OPEN_ISSUES_DETAILS.md#oi-8) | High | Backward through quantized ops (`_sg_onednn_*`) unvalidated |

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
| [OI-16](OPEN_ISSUES_DETAILS.md#oi-16) | Low | CUDA Graphs: host-generator `kRandom` ops excluded; `tensordot`/`np.dot` reroute pending |

## Platform

| ID | Sev | Summary |
|----|-----|---------|
| [OI-17](OPEN_ISSUES_DETAILS.md#oi-17) | High | arm64: oneDNN INT8 + subgraph fusion gated off (Xbyak_aarch64 JIT unreliable) |
| [OI-18](OPEN_ISSUES_DETAILS.md#oi-18) | Med | bf16 emulated in fp32 on non-AVX-512-BF16 CPUs |
| [OI-19](OPEN_ISSUES_DETAILS.md#oi-19) | High | cuBLAS≥13.5 pin requires driver R590+; CUDA 13.0 / R580 unsupported |
| [OI-20](OPEN_ISSUES_DETAILS.md#oi-20) | Low | cuDNN minor mismatch warning (wheel built vs 9.23, pin resolves 9.22) — harmless |

## Engine / concurrency

| ID | Sev | Summary |
|----|-----|---------|
| [OI-21](OPEN_ISSUES_DETAILS.md#oi-21) | Med | Rare long-running inference hang (A6) — instrumented, needs an aarch64 repro |
| [OI-22](OPEN_ISSUES_DETAILS.md#oi-22) | Low | `LazyAllocArray::Get()` lock-free read is a benign data race (needs C++20 atomics) |
| [OI-23](OPEN_ISSUES_DETAILS.md#oi-23) | Low | CUB global-reduce input aliasing flagged but benign — **won't fix** |

## Ecosystem / packaging / CI

| ID | Sev | Summary |
|----|-----|---------|
| [OI-24](OPEN_ISSUES_DETAILS.md#oi-24) | Med | Wheel publication is manual; no conda/system packaging or release automation |
| [OI-25](OPEN_ISSUES_DETAILS.md#oi-25) | Med | No CUDA build-matrix CI (Ada/Hopper/Blackwell + CUDA 12.x) |
| [OI-26](OPEN_ISSUES_DETAILS.md#oi-26) | Low | Downstreams unverified (GluonNLP/Sockeye/AutoGluon, ps-lite, Py3.13+, NumPy 2.x, DLPack) |
| [OI-27](OPEN_ISSUES_DETAILS.md#oi-27) | Med | ONNX fixed in source but not shipped in wheels |

## D2L book compatibility

| ID | Sev | Summary |
|----|-----|---------|
| [OI-28](OPEN_ISSUES_DETAILS.md#oi-28) | Med | `train_ch13` multi-GPU DeadKernel, under investigation |
| [OI-29](OPEN_ISSUES_DETAILS.md#oi-29) | Low | Two convergence gaps are book-side fixes (scheduler `epoch_size`, FCN `trainer.step`) |

## Build warnings (informational)

| ID | Sev | Summary |
|----|-----|---------|
| [OI-30](OPEN_ISSUES_DETAILS.md#oi-30) | Low | Vendored dmlc/oneDNN build warnings (u32 sentinel, ITT exec-stack) — by policy, not patched |
