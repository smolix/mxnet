# Open issues — details

Deep context for every item in [`OPEN_ISSUES.md`](OPEN_ISSUES.md): root cause,
why it is deferred, the reproducer, and the shape of the real fix. This file
intentionally carries detail **only for items that are still open** — context for
resolved work is retired to `git log` (and summarized in [`FIXED.md`](FIXED.md)).

---

## Correctness / numeric

> OI-1 (integer `matmul`/`dot`/`tensordot`), OI-2 (`np.mean(int)` dtype), and OI-3
> (fp16 `var`/`std` overflow) are **resolved** — see [`FIXED.md`](FIXED.md) §7.

<a id="oi-4"></a>
### OI-4 — NumPy view-aliasing gaps (narrowed)
The Apache-issue regression sweep flagged a class of operations that NumPy returns as a
**strided view** but this fork materializes as a copy. Split by PyTorch's behavior (the
fork's tie-breaker for value/format decisions):
- **Resolved (match PyTorch):** `flip` / `flipud` / `fliplr` / `rot90` return
  independent copies — exactly what `torch.flip` & friends do (they copy; they are not
  views). Negative-step slicing (`a[::-1]`) is likewise not a view (PyTorch rejects
  negative steps outright). The values were always correct; these are now pinned as
  copy-semantics tests instead of view xfails (`FIXED.md` §7).
- **Still open:** positive **stepped slicing** (`a[::2]`) and **axis-moving** ops
  (`moveaxis` / `rollaxis`) return copies here, whereas PyTorch (and NumPy) return
  views, so in-place writes to the result do not alias the base. **Real fix:**
  signed/strided metadata in `NDArray` / `TBlob` plus operator/API plumbing to honor it
  — a substantial change (today `TBlob::CheckContiguous()` is a stub returning `true`
  and strides are discarded). **Workaround:** none needed for value-correctness
  (results are correct copies); only view-aliasing semantics differ.

---

## oneDNN INT8 quantization (CPU / x86)

> On Apple Silicon the entire INT8 path is gated off (see [OI-17](#oi-17)); the items
> below concern the x86 oneDNN path.

<a id="oi-5"></a>
### OI-5 — Asymmetric quantize loses the sub-integer shift (was H15)
oneDNN v3 folds the affine offset into integer zero-points, so a *fractional* `shift`
cannot be represented. The correct fix — fold the fractional shift as an input
pre-bias — risks aliasing the caller's buffer via `Reorder2Default()`, and a CPU
fallback would trigger on ~every calibrated call (perf cliff). **Recommended:** input
pre-bias into a *private* copy. Needs a quantization-accuracy harness to validate;
this is not on the GPU-wheel headline path. **Deferred (analyzed).**

<a id="oi-6"></a>
### OI-6 — Quantized concat / batch_norm affine fallback layout (was H16)
Two real sub-problems remain: (1) root-cause the u8→s8 f32-roundtrip in the affine
requant, and (2) consolidate the three hand-rolled affine-requant helpers. Note: an
`IsView()` CHECK is *not* the fix — MXNet default storage is always contiguous after
`Reorder2Default`, so such a check would over-reject valid axis-0 slices. Needs oneDNN
expertise + accuracy validation. **Deferred (analyzed).**

<a id="oi-7"></a>
### OI-7 — uint8 requantize CPU fallback (was M19)
uint8 requantize currently routes to a CPU fallback; needs an asymmetric reorder plus
an accuracy harness. **Deferred (oneDNN).** A latent related use-after-free
(`ConvertWeightBias2DNNL` deferred-submit with registered scale-memory locals) was
already closed defensively with `CHECK(submit || weight_scales.empty())` (inert today
since both callers pass `submit=true`).

<a id="oi-8"></a>
### OI-8 — Backward through quantized ops unvalidated
Forward INT8 inference through `_sg_onednn_fully_connected` / `_sg_onednn_conv` is
solid and tested. The **backward** pass through these fused quantized ops has not been
validated. Treat quantized training as unsupported until a QAT-backward acceptance run
exists. (The QAT-backward shard currently reports an expected mixed pass/xfail state.)

---

## Performance / refactor (deferred — results are correct, only speed/cleanliness)

<a id="oi-9"></a>
### OI-9 — RNN re-issues cuDNN descriptors every forward (was M4)
Descriptor / temp-size / clip caching and an async sequence-length memcpy live only in
the narrow `use_sequence_length` path. Stateful change; deferred to its own cycle.

<a id="oi-10"></a>
### OI-10 — Proposal ops do per-call `cudaMalloc`/`cudaFree` (was M7)
Faster-RCNN contrib proposal ops bypass `ctx.requested` scratch and malloc per call.
`FRCNN_CUDA_CHECK` already throws like `CUDA_CALL`; the real ask is the per-call
`cudaMalloc` → `ctx.requested` refactor on these rarely-tested legacy ops. Own cycle.

<a id="oi-11"></a>
### OI-11 — `SetTBlob()` mutates via `const_cast` (was M12)
A `const` method does an in-place oneDNN `SelfReorder2Default` via `const_cast`. This
was an *intentional* fork change that fixed a crash; reverting risks reintroducing it.
The correct fix (reorder at call sites under var serialization) needs a full caller
audit. **Deferred (analyzed).** Latent thread-safety only.

<a id="oi-12"></a>
### OI-12 — Eager per-op heap NDArray allocation (was M14)
Each eager op `new`s an NDArray wrapper per call. A by-value `ScopedDerefInputOutput`
+ `PushFCompute` rewrite was tried but **reverted**: it destroyed NDArray handles
during exception unwinding when `Engine::Push` throws (e.g. invalid-GPU device check),
so an NDArray dtor threw mid-unwind → `std::terminate` (process abort on *any*
synchronous engine error; caught by `test_incorrect_gpu`). Needs a redesign that does
not destroy handles during unwinding.

<a id="oi-13"></a>
### OI-13 — Reduction headroom
GPU axis (non-global) reductions still leave ~35–40% bandwidth on the table for common
cases; would need vectorized (`float4`) loads in the RTC reduce kernel. CPU `float64`
mean is still single-threaded (native path). Both are perf loose ends, not regressions.

<a id="oi-14"></a>
### OI-14 — Batch-size-1 inference slow on AVX2-only CPUs (was B8)
On AMD EPYC 7B12 (Zen 2, AVX2-only): Conv2D 64ch `(1,3,224,224)` runs 49.8 ms at
`OMP_NUM_THREADS=1` but **536.4 ms at OMP=64** (10× *negative* scaling). Root causes:
(1) `IC=3` is pathological for `brg_conv_fwd:avx2` — the Acdb16a weight format pads IC
to 16 (81% waste); (2) brgemm is throughput-designed and its overhead dominates at
bs=1; (3) oneDNN v3 picks brg_conv over v2's faster `jit:avx2` here. **Workarounds:**
set `OMP_NUM_THREADS=1` for bs=1 inference; consider `DNNL_DEFAULT_FPMATH_MODE`. Some
of the 512-channel slowdown is inherent to the padding + cache behavior. The
`OMP_NUM_THREADS=1` / fpmath guidance + root cause is now documented in `README.md`
(troubleshooting); the kernel-level perf fix itself remains deferred.

<a id="oi-15"></a>
### OI-15 — cuBLASLt follow-ups deferred
PR-A (fp32) and PR-B (fp16/fp64) landed (see `FIXED.md` §2). Deferred: mshadow
`dot_engine-inl.h` rewiring (possibly a separate submodule PR), INT8 via cuBLASLt, the
true-fp16 HMMA path (`MXNET_FC_TRUE_FP16=1` still uses legacy `cublasGemmEx`), batched
GEMM coverage, and the default-on flip. bf16 is not yet reachable via `mx.nd.dot`
(mshadow `MSHADOW_REAL_TYPE_SWITCH` limitation). Datacenter Blackwell (B100/B200) is
expected to show ~1.5–1.7× but has not been benchmarked (validation card is a 110 W
workstation SKU).

<a id="oi-16"></a>
### OI-16 — CUDA Graphs remaining exclusions
Host-generator `kRandom` ops (`np.random.*`, shuffle, image augmentation) are excluded
from capture — they would need a device-resident offset or a host-side per-replay bump.
(`tensordot` / `np.dot` now reroute through the capture-safe cuBLASLt path — `MatrixDot`
calls `linalg_gemm` on GPU instead of the legacy mshadow `dot()` — see `FIXED.md` §3.)
Everything else in the default-on static-shape regime captures.

---

## Platform

<a id="oi-17"></a>
### OI-17 — Apple Silicon oneDNN INT8 + subgraph fusion gated off
`src/operator/nn/dnnl/dnnl_base-inl.h` returns `false` from
`SupportDNNLAArch64JITPrimitives()` and `SupportDNNLQuantizedOps()` on `__aarch64__`:
oneDNN 3.x routes several AArch64 primitives through Xbyak_aarch64 paths that fail with
`ERR_INTERNAL` on Apple Silicon, and INT8 GEMM wants MKL (`cblas_gemm_s8u8s32`, absent
under Accelerate). Consequently every oneDNN subgraph fusion pass is disabled at runtime
and quantized ops fall back to MXNet's native kernels. **Effect:** the float oneDNN
backend works; the `tests/python/dnnl` *fusion + quantization* lane asserts fusion/
quant happened and therefore does not apply on arm64 (it is not a wheel defect). The
float path and ~14.9k unittest/operator/NumPy/Gluon tests pass on the macOS CPU wheel.

> OI-18 (bf16 emulated in fp32 on CPUs without AVX-512-BF16 — inherent) and OI-19
> (cuBLAS≥13.5 / driver R590+) are **accepted constraints**, and OI-20 (cuDNN
> minor-version mismatch warning) is **resolved** (the runtime check now warns only on a
> major-version difference) — see [`FIXED.md`](FIXED.md) §1 and §11.

---

## Engine / concurrency

<a id="oi-21"></a>
### OI-21 — Rare long-running inference hang (was A6/A7)
A long-running inference can hang in `WaitForVar`. Two plausible mechanisms: (1) a
missing notify edge in `CompleteWriteDependency` (most likely; needs a minimized
reproducer); (2) the queue receiving `SignalForKill` before an op completes. Now
*observable* via `MXNET_ENGINE_DIAG=1`, which adds a watchdog that logs the stuck var
pointer / `pending_ops` / shutdown phase / kill flag on timeout (it does not abort).
A reliable reproducer (seen on aarch64) is needed to land a fix. Distinct from the
cold-start deadlock, which **is** fixed (`FIXED.md` §5).

<a id="oi-22"></a>
### OI-22 — `LazyAllocArray<T>::Get()` lock-free read race
The lock-free fast path reads `head_[idx]` without synchronization while another thread
may be writing it — technically UB, pre-existing, benign in practice. A proper fix needs
C++20 `std::atomic<std::shared_ptr>` or a seqlock. (Note: the *cold-start deadlock* fix
already moved `ThreadPool` readiness out of the `create_mutex_` critical section; this
remaining item is the lock-free read itself.)

> OI-23 (CUB global-reduce input aliasing) is closed **won't-fix** — see
> [`FIXED.md`](FIXED.md) §11.

---

## Ecosystem / packaging / CI

<a id="oi-24"></a>
### OI-24 — Manual packaging, no release automation (was O4/O7)
Linux/macOS wheels are published manually to GitHub Releases; there is no conda package,
no system package, and no automated release pipeline. Expensive CUDA build automation is
deliberately deferred. The d2l side consumes wheels via `tools/update_mxnet_wheel.py`.

<a id="oi-25"></a>
### OI-25 — No CUDA build-matrix CI (was C4)
There is no CI matrix building/testing across Ada / Hopper / Blackwell and CUDA 12.x.
`sm_89` (Ada) is validated by hand here; CUDA 12.x coverage is deferred. The single-host
build/test target keeps OMP threads capped (1–4 per xdist lane; ~48–64 runnable tasks).

<a id="oi-26"></a>
### OI-26 — Downstream libraries unverified (was T2–T6/T11)
Not validated against this fork: GluonNLP / Sockeye / AutoGluon (T2); ps-lite distributed
rendezvous (T3); Python 3.13+ (T4); NumPy 2.x ABI (T5 — the operator shape/axis-param
rendering incompatibility is now fixed, see [`FIXED.md`](FIXED.md) §10, but a full-suite
NumPy 2.x run is not yet a gate); DLPack interop (T6); broader cross-platform process
lifecycle (T11). Strategic; revisit per concrete demand.

<a id="oi-27"></a>
### OI-27 — ONNX fixed in source but not shipped in wheels
ONNX export/import was repaired in PR #38 (opset-13 default, ONNX 1.21 / ORT 1.24
validated; see `FIXED.md` §2), but the **published wheels are built ONNX-free**, so
`import`-time the ONNX path is absent. To use ONNX, build from source with the ONNX
toolchain installed. Future opset bumps should be opened as new compatibility work only
when required.
