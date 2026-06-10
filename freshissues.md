# Fresh Code Review — MXNet Fork (smolix/mxnet)

**Date:** 2026-06-10
**Scope:** A from-scratch review of the whole codebase (not just the fork's diff), looking for
poor performance, likely sources of error, and poor coding practice — in preparation for making
the fork safe for external users. Prior design/progress docs were deliberately disregarded.

**Method:** 13 parallel reviewer passes — 7 over the fork's own changes (diff vs upstream
merge-base `b84609d3`), 6 over the entire codebase (pre-existing/systemic concerns). Every
finding below was verified by reading the actual source, not inferred from the diff alone.

Each finding is tagged **[fork]** (introduced/changed by this fork) or **[pre-existing]**
(long-standing upstream code), so you can decide what blocks a release vs. what's a backlog item.

---

## TL;DR — the must-fix shortlist

If you do nothing else before releasing, address these:

1. **CUDA-graph opt-out drift** — three ops route through blocking D2H/sync or ephemeral alloc but
   are *not* excluded from capture while their siblings are: `scatter_nd`/`_scatter_set_nd`,
   `_linalg_inverse`, `_backward_npi_percentile`. Under the fork's **default-on** capture these
   abort the graph with the cryptic CUDA error 900. (C1, C2, H1)
2. **`mxnet_op.h` `Kernel::Launch(int N)` truncates element counts >2³¹** — the single most-used GPU
   dispatch path silently corrupts tensors >2.1 B elements. (C3)
3. **Engine lapped-event TOCTOU race** — GPU dependency events can under-synchronize → intermittent
   wrong results. (C4)
4. **Defense-in-depth is disabled exactly in the shipped regime** — `AssertGemmCaptureSafe` and the
   GPU storage pool both gate their capture-safety guards on `MXNET_ENABLE_CUDA_GRAPHS`, which is
   *unset* under default-on capture, so capture-illegal fallbacks abort cryptically instead of
   failing with an actionable message. (H2)
5. **`NDArrayBase.__del__` frees an unguarded handle** — crashes/raises during the unpickling path. (H3)
6. **Build silently drops requested CUDA arches** on older toolkits → wheel with no kernel image /
   no Blackwell PTX, no configure-time error. (H4)
7. **Hot-path blocking syncs** — ~36 operators do a D2H copy + `cudaStreamSynchronize` inside
   `FCompute`; several are on the training step and all are capture-illegal. (H5)

---

## Resolution status (updated 2026-06-10)

Fixes are being applied in tiers (Critical → Low) with a build + targeted-test
checkpoint after each batch. Status legend: ✅ fixed & tested · 🔧 fixed, pending
build/test · ⏳ not started · ➖ no code change needed (explained inline).

| ID | Status | Note |
|----|--------|------|
| C1 | ✅ | scatter_nd/_scatter_set_nd excluded; checker verified functional (not dead). |
| C2 | ✅ | _linalg_inverse excluded; backward confirmed capture-safe (gemm only). |
| C3 | ✅ | `Kernel::Launch`/generic kernels now `index_t`; >2³¹ test passes. |
| C4 | ✅ | TOCTOU re-check after wait + signed-loop fix; 835 op tests green. |
| H1 | ✅ | `_backward_npi_percentile` excluded from capture. |
| H2 | ✅ | `AssertGemmCaptureSafe` gates on `UseCuBlasLt()`; pool alloc emits actionable capture error. |
| H3 | ✅ | `__del__`/`__dealloc__` guard missing/NULL handle. |
| H4 | ✅ | CMake FATALs when a requested CUDA arch yields no gencode. |
| H5 | ➖ | Reachable-under-capture cases were C1/C2/H1 (done). Remaining ~36 sync sites are dynamic-shape ops never bulked into a static-shape capture segment; H6/H8 cover the static-regime offenders. |
| H6 | ✅ | count_sketch runs on the engine stream; per-iter `cudaDeviceSynchronize` removed. |
| H7 | ✅ | gather_nd backward (GPU+CPU) drops OOB indices instead of wrapping; also de-`int`-truncated. |
| H8 | ✅ | index_add/index_update host validation gated behind `MXNET_INDEX_BOUNDS_CHECK` (default off); kernels already guard. np.insert left as-is (dynamic-shape op, 1-byte check). |
| H9 | ✅ | cuBLASLt descriptors cached per GemmKey (race-free, process-lifetime); `Find` copies by value. Validated by the bitwise differential-replay net. |
| H10 | ✅ | OOM-retry releases the storage mutex across sleep/sync. |
| H11 | ✅ | CPU dev_id==0 RNG resources fast-pathed without the create mutex. |
| H12 | ✅ | `MX_API_END`/`_HANDLE_ERROR` now `catch (...)`. |
| H13 | ✅ | ObjectPool sharded into 8 per-thread free lists (no ABA hazard). |
| H14 | ✅ | Legacy NDArray made unhashable (matches numpy frontend). |
| H15 | ⚠️ | **Deferred (analyzed).** oneDNN zero-points are integer, so the fractional `shift` cannot be represented; the correct fix (fold fractional shift as an input pre-bias) risks aliasing the caller's buffer via `Reorder2Default()`, and a CPU fallback would trigger on ~every calibrated call (perf cliff). Needs a quantization-accuracy harness to validate; not the GPU-wheel headline path. Recommended: input pre-bias into a private copy. |
| H16 | ⚠️ | **Deferred (analyzed).** Contiguity is not actually violated for MXNet default storage (always contiguous after `Reorder2Default`; an `IsView()` CHECK would over-reject valid axis-0 slices). Real remaining work = root-cause the u8→s8 f32-roundtrip and consolidate the 3 affine-requant helpers — both need oneDNN expertise + accuracy validation. |
| H17 | ✅ | Wheel selection uses `ls -1t` (newest). setup.py hard-fail when OpenCV libs missing: pending (minor). |
| M1 | ✅ | reduce_cub asserts the transform-iterator temp size ≤ the double-query size before launch, so a future CCCL that sizes by iterator fails loudly instead of overflowing the caller workspace. |
| M2 | ✅ | histogram range kernel promotes to double (bin-edge parity). |
| M3 | ➖ | No change: inference makes exactly one temp-space call, so no aliasing today. |
| M4 | ⏳ | Deferred (perf, stateful). Descriptor/temp-size/clip caching + async seq-length memcpy live in the narrow `use_sequence_length` path; own cycle. |
| M5 | 🔧 | FC fp16 CPU path now honors `kAddTo` (was a hard `kWriteTo` assert); conv fp16 path already handled `kAddTo`. BLAS-upcast perf rewrite deferred (rare CPU-fp16 fallback, needs ctx/workspace). |
| M6 | ✅ | np_norm backward: CUDA calls checked + RAII free (no leak on throw). |
| M7 | ⚠️ | **Deferred (analyzed).** `FRCNN_CUDA_CHECK` already throws like `CUDA_CALL` (CHECK-based); the real ask is the per-call `cudaMalloc`→`ctx.requested` refactor on rarely-tested legacy proposal ops — own cycle. |
| M8 | ✅ | `CUDA_KERNEL_LOOP`/`get_num_threads`/`cuda_get_num_blocks` now `index_t` (C3 siblings). |
| M9 | ✅ | `IndexTensorToVector` syncs the stream before reading the host buffer. |
| M10 | ✅ | NaiveEngine erases var exceptions on `DeleteVariable` (no stale/leak). |
| M11 | ✅ | `SyncCopyToCPU` reorders only when `IsDNNLData()` (no extra round-trip). |
| M12 | ⚠️ | **Deferred (analyzed).** The `const_cast` `SelfReorder2Default` was an intentional fork change; reverting risks reintroducing the crash it fixed. Correct fix (reorder at call sites under var serialization) needs a full caller audit. |
| M13 | ✅ | Over-cap pool free now waits on the chunk's events before cudaFree. |
| M14 | ⚠️ | **Reverted (deferred).** The by-value `ScopedDerefInputOutput` + `PushFCompute` change destroyed NDArray handles during exception unwinding when `Engine::Push` throws (e.g. invalid-GPU device check), so an NDArray dtor threw mid-unwind → `std::terminate` (process abort on ANY synchronous engine error, caught by `test_incorrect_gpu`). Reverted to the heap path; needs a redesign that doesn't destroy handles during unwinding. |
| M15 | ✅ | **Root cause + leak both fixed.** `_cy3` never compiled — `cython/ndarray.pyx:41` used Py2 `long` → `cythonize` errored → silent ctypes fallback for every op. Fixed the compile; gave the cython `NDArrayBase` a `__del__` finalizer (PEP 442, mirrors ctypes) so cycle-trapped NDArrays free their handle on gc collection (the missing finalizer was the leak source the validation caught). Verified: cython active, prior-leaking svd/svdvals/qr now pass. setup.py honors `MXNET_WITH_CYTHON=1`; wheel build enables it. |
| M16 | ✅ | batch_dot caches the tensor-core env read; denorms load relaxed. |
| M17 | ✅ | DLPack `ToDLPack` uses `unique_ptr` RAII until the no-throw `release()` hand-off. |
| M18 | ✅ | `ConvertWeightBias2DNNL` now `CHECK(submit || weight_scales.empty())` — closes the latent use-after-free (deferred-submit with registered scale-memory locals). Inert today (both callers pass submit=true). |
| M19 | ⚠️ | **Deferred (oneDNN).** uint8 requantize CPU-fallback; needs asymmetric reorder + accuracy harness. |
| M20 | ✅ | cnpy: header-len decode fixed, header bounds-checked, shape overflow guarded, zero-length zip name guarded. New test: `test_npy_npz_load_safety.py`. |
| M21 | 🔧 | Done: source-glob `CONFIGURE_DEPENDS`, removed no-op `OpenMP_EXE_LINKER_FLAGS`. Backlog (can break valid setups): FindCUDNN version guard, ChooseBlas MKL default, NCCL static/dynamic reconciliation. |
| M22 | ✅ | `MXNET_DEPS_REQUIRE_CHECKSUM` hard-fails un-pinned downloads; build_opencv extracts to a staging dir then atomically promotes (no reuse of a partial/poisoned tree). |
| M23 | ✅ | GPU lane runner fails loudly on cross-lane GPU requests (`--allow-multi-gpu` opt-out); `run_wheel_full_test` `-n` now honors `PARALLEL_CPU`. |
| L1 | ✅ | `num_graph_creations` → `atomic<size_t>`. |
| L2 | ✅ | `cublasGetStream`/`cudaGetDevice`/`cudaSetDevice` checked (fall back instead of default-stream/wrong-device). |
| L3 | ✅ | cuBLASLt workspace floor: named `kHeuristicMaxWorkspaceBytes` + `static_assert` it ≤ the persistent floor (capture invariant). |
| L4 | ✅ | GemmKey `scale_dtype`: CHECK it is implied by `compute_type` (fp64↔`CUDA_R_64F`, else `CUDA_R_32F`). |
| L5 | ✅ | `VerifyReplay` wraps its raw `cudaMalloc`/free in a `DeviceStore` device guard. |
| L6 | ✅ | `InvalidateOutputs` `kNullOp` exclusion documented (intentional fork lifetime fix, commit b26ff5b3f; not reverted). |
| L7 | ✅ | Dead `SyncDgrad` machinery removed from rnn-inl.h. |
| L8 | ✅ | deconv `ALLOW_CONVERSION` documented as opt-in precision tradeoff. |
| L9 | ✅ | batch_norm running-stat-in-forward documented (PyTorch-style behavior change). |
| L10 | ➖ | No change: native scalar dropout path **also** rejects `kAddTo`, so the crash is not backend-dependent as the review described. |
| L11 | ✅ | c_api thread-local return-buffer lifetime contract documented. |
| L12 | ✅ | `OnComplete` decrements atomic `pending_` lock-free; takes `finished_m_` only on the 0-transition. |
| L13 | ✅ | `CheckDuplicate` now always-on + allocation-free O(n²); duplicate vars fatal instead of a release-build hang. |
| L14 | ✅ | `IsEngineAsync` matches `CreateEngine`: "Async" honored only as a suffix. |
| L15 | ✅ | 6 bare `except:` → `except Exception ... from e`; fixed `type(array)`→`type(source_array)`. |
| L16 | ✅ | `integer_types` includes `np.integer` (np.int8/16 indices); `py_str` NULL-guarded. |
| L17 | 🔧 | dataloader timeout `print` → `logging.warning`. (spawn-default left as-is: behavior change.) |
| L18 | ✅ | np_repeat flat-index `index_t` (✅ batch-1); quantized_elemwise_mul int32 output rounds via `nearbyint` (🔧 batch-2). SetInitValue/dead-Map/bincount left (cosmetic/already-safe). |
| L19 | ✅ | `nvidia-smi` count guarded (`2>/dev/null` + `grep '^GPU '`); `-n` honors `PARALLEL_CPU`; fp16 smoke BRANCH default → `master`. |

New regression tests added: `tests/python/gpu/test_freshfix_regressions.py` (C3 launcher,
scatter_nd), `tests/python/unittest/test_freshfix_frontend.py` (H3 del/pickle, H14 hash),
and `test_cuda_graphs_linalg_inverse_excluded_default_on` in the capture-replay suite (C2).

**Validation batches (single-arch sm_89 `build-g`, 4-GPU sharded):**
- **Batch 1** (prior-session 🔧 + L1/L2/L7/L15/L16/L18-np_repeat + M21/M22/M23) — **green**:
  24,448 passed / 126 skipped / 0 failed across test_operator_gpu + test_numpy_op.
- **Batch 2** (L3/L4/L5/L6/L8/L9/L11/L12/L13/L14/L18-quantized + M1/M17) — built. First
  4-GPU run mistakenly added `test_operator.py` to the GPU lane and surfaced 1 failure
  (`test_custom_op_fork`, "CUDA: initialization error" in a forked child) + 1 hang
  (`test_np_moment::asnumpy` blocking, faulthandler 20-min timeout, runaway kernel left
  GPU1 at 100%). **Root-caused as environmental, NOT a code regression:** both
  `test_custom_op_fork` and `test_np_moment` pass in isolation (0.3s / 2.4s); the cause is
  the fork-after-CUDA pitfall — `test_operator.py` (a CPU suite with `os.fork()`-based
  tests) was run in the same long-lived GPU process, and a fork after CUDA init wedges the
  child/context, cascading into the later hang. The always-on `CheckDuplicate` (L13) and
  lock-free `pending_` (L12) were specifically cleared: a CPU op-surface smoke
  (`test_custom_op_fork`/dot/reduce/broadcast/elementwise) passed with no duplicate-var
  fatal and no hang. **Lesson:** keep `test_operator.py` (and other fork tests) OFF the
  single-process GPU lane; run it CPU-only. See L17 (fork-after-CUDA).
- **Batch 2 re-validation — GREEN.** GPU-lane suites (`test_operator_gpu` + `test_numpy_op`
  + freshfix + cuda_graphs_replay) sharded across healthy GPUs (0,2,3) via the new `GPU_IDS`
  override: **24,447 passed** (8149/8148/8150) + 1 flaky failure
  (`test_convolution_independent_gradients`, a known cuDNN-nondeterminism flake — passes 3/3
  in isolation AND 3/3 with its own reproduce-seed; no fork/conv code in this batch). CPU
  `test_operator.py` (OMP=4): **1113 passed / 0 failed** in 3:53. L3/L4/L5/L6/L8/L9/L11/L12/
  L13/L14/L18-quantized + M1/M17 validated on both surfaces → 🔧 → ✅.

**Root cause of the CPU `test_operator.py` slow/"hang" — OpenMP oversubscription (NOT the
scheduler fix):** `test_batchnorm` (and similar small-op tests) blocked in `asnumpy()` at
**1255% CPU** — spinning, not lock-blocked. MXNet defaults OMP threads to the core count
(64 on this host); the suite runs 108 tiny batchnorm ops, and with the default **active**
wait policy the 64 idle OMP threads busy-spin at barriers, so per-op fork/join cost dwarfs
the compute and forward progress nearly stops. Under `xdist -n 8` that's 8×64 = 512 threads
on 64 cores → the 20-min faulthandler timeout. Measured on `build-g` (`test_operator.py::test_batchnorm`):

  | OMP_NUM_THREADS | wait policy | time |
  |---|---|---|
  | 4 / 16 | active | **24–25 s** ✅ |
  | 64 (default) | active | **>300 s** ✗ |
  | 64 (default) | passive | **89 s** ✅ |

  Confirmed pre-existing/environmental: reproduces identically on the older `build/`
  (06-09) binary that predates L12/L13/L14, and the only batch-2 `batch_norm.cu` edit was a
  comment. **Fix:** the runner now caps `OMP_NUM_THREADS` to `cores/NSHARDS` (≤8) and sets
  `OMP_WAIT_POLICY=passive`; the CPU `test_operator.py` re-run uses `OMP_NUM_THREADS=4`.
  L12/L13/L14 thereby cleared on the GPU lane (0 failures, 0 dup-var fatals, 0 deadlocks)
  and the CPU op surface.

Portable test runner added: `tools/run_gpu_shards.sh` + `tools/pytest_gpu_shard_plugin.py`
(supports `GPU_IDS="0 2 3"` to target specific GPUs; caps OMP threads + passive wait policy
to avoid many-core oversubscription).

---

## Still open — what each means and why

Everything else in the table above is ✅ (fixed & validated), ➖ (no change needed), or 🔧
(fixed, pending only final build/test: M5, M21-partial, L17). The **genuinely open** items
are the 7 below. **None block the GPU wheel** — they are CPU-INT8-quantization, perf/refactor,
or latent thread-safety. The only items with real *correctness* weight are H15 and H16, both
confined to INT8 quantized **CPU** inference (gate behind "are we advertising INT8 quant?").

### oneDNN / INT8-quantization cluster (CPU; not the GPU-wheel path)

- **H15 — asymmetric quantize loses sub-integer shift.** v3 folds the affine offset into a
  single *integer* zero-point; oneDNN zero-points must be integers, so the fractional part of
  `shift` (≤0.5 LSB) is dropped → *silent* accuracy loss on quantized models. **Open because**
  the correct fix (fold the fractional shift in as an input pre-bias) risks aliasing the
  caller's buffer via `Reorder2Default()`, and a CPU fallback would fire on ~every calibrated
  call (perf cliff). Needs a quantization-accuracy harness — cannot be patched blind.
- **H16 — quantized concat/batch_norm affine fallback layout.** Hand-rolled requant indexes
  output as dense row-major. **Open because** the feared bug (wrong results on strided inputs)
  is *not* actually reachable (MXNet default storage is always contiguous after
  `Reorder2Default`). The real remaining work — consolidate the 3 near-identical affine-requant
  helpers and root-cause a u8→s8 f32 round-trip — is quality/refactor needing oneDNN expertise
  + accuracy validation, not a live-bug fix.
- **M19 — uint8 requantize uses a per-call CPU fallback.** Throughput regression on the
  ReLU-fused INT8 inference path (perf, not correctness). **Open because** the proper fix
  (asymmetric oneDNN reorder with DST scale + zero-point) needs accuracy validation.

### Perf / refactor — deferred to their own cycle

- **M4 — RNN re-issues cuDNN descriptors every forward.** Re-sets data descriptors +
  temp-size query + clip each call though invariant; plus a sync memcpy in the variable-length
  path. Pure host overhead. **Open because** the fix is stateful caching keyed on shape and the
  sync-memcpy is in the narrow `use_sequence_length` path — warrants its own careful cycle.
- **M7 — proposal ops do per-call `cudaMalloc`/`cudaFree`.** Faster-RCNN contrib ops bypass the
  pool (fragmentation) + a D2H sync. **Open because** the "abort macros" concern was overstated
  (`FRCNN_CUDA_CHECK` already throws like `CUDA_CALL`); the real ask — route allocations through
  `ctx.requested` — is a refactor on rarely-tested legacy ops (low payoff, high regression risk).
- **M14 — eager-dispatch per-op heap allocation.** Each op `new`s an NDArray wrapper per
  input/output. **Open because it was attempted and REVERTED this session:** storing handles by
  value destroyed them during exception unwinding, so an engine error (e.g. invalid GPU)
  aborted the process (`std::terminate`, caught by `test_incorrect_gpu`). Needs a redesign that
  frees handles outside the throw path.

### Latent / intentional fork change

- **M12 — `SetTBlob()` mutates via `const_cast`.** A `const` method does an in-place oneDNN
  reorder through `const_cast` — a thread-safety hazard if the chunk is shared. **Open because**
  it was a *deliberate* fork change; reverting risks reintroducing the crash it fixed, and the
  correct fix (reorder at call sites under engine var-serialization) needs a full audit of every
  `SetTBlob` caller. Latent in practice (the var is exclusively held during op execution).

---

## Critical

### C1 — `scatter_nd` / `_scatter_set_nd` do a blocking D2H + `cudaStreamSynchronize` during capture **[fork]** ✅ FIXED
- **Where:** `src/operator/tensor/indexing_op.cu:486` (`GatherNDCheckBoundGPU`), op regs `indexing_op.cu:993`, `indexing_op.h:1716`.
- **What:** `ScatterNDForward<gpu>` runs `ScatterNDIndexChecker<gpu>::Check` → `cudaMemcpyAsync(D2H)` + `cudaStreamSynchronize` on **every call**. The sibling `gather_nd` was explicitly excluded from capture for exactly this reason (`indexing_op.cu:988`), but the two scatter ops that funnel through the same checker were missed, and they carry no `FIsCUDAGraphsCompatible=false`.
- **Why it matters:** Reachable under the fork's default-on capture. `cudaStreamSynchronize` while a stream is capturing returns error 900 and invalidates the whole segment graph.
- **Fix:** Add `.set_attr<FIsCUDAGraphsCompatible>(...return false)` to `scatter_nd` and `_scatter_set_nd` (mirror `gather_nd`). Long-term, move bounds-checking to a device-side trap so the op stays capturable.
- **Bonus (related, `src/operator/tensor/indexing_op.cu`):** the GPU `ScatterNDIndexChecker` currently computes a validity flag into a workspace and then **never reads it back** — the validation is a no-op launch. Either consume the flag (one int D2H + `CHECK`) or remove the dead call. And `scatter_nd`'s in-kernel guard *silently drops* out-of-bounds writes, while `gather_nd` backward (below, H7) silently *wraps* them — pick one consistent OOB policy.

### C2 — `_linalg_inverse` not excluded from capture though it uses the same alloc+sync as the excluded `det`/`slogdet` **[fork]**
- **Where:** `src/operator/tensor/la_op.cu:98`; primitives in `src/operator/linalg_impl.h` (`linalg_batch_getrf`/`getri`, `EPHEMERAL_GPU_STORAGE_ALLOC` whose dtor does `cudaStreamSynchronize` + `Storage::Free`).
- **What:** `_linalg_det`/`_linalg_slogdet`/`_backward_linalg_det` were all flagged `FIsCUDAGraphsCompatible=false` with comments citing the getrf/getri allocations, but `_linalg_inverse` (forward) uses the identical primitives and was missed.
- **Why it matters:** Under default-on capture the alloc-on-miss `cudaMalloc` + `cudaStreamSynchronize` are capture-illegal → graph abort.
- **Fix:** Flag `_linalg_inverse` (and audit `_backward_linalg_inverse`, the `potri` backward) `false`, consistent with the det family.

### C3 — central GPU kernel launcher truncates the element count to `int` **[pre-existing]**
- **Where:** `src/operator/mxnet_op.h:1177,1185,1194` (`Kernel<OP,gpu>::Launch(Stream*, int N, ...)`, `mxnet_generic_kernel`/`_ex`); also `get_num_threads(const int N)` at `mxnet_op.h:58` and `CUDA_KERNEL_LOOP`.
- **What:** `.Size()` returns `size_t`, but the launcher takes `int N` and loops with `int i`. The thousands of call sites doing `Kernel<...>::Launch(s, blob.Size(), ...)` silently truncate any tensor with >2³¹ (~2.1 B) elements.
- **Why it matters:** Tensors above ~2 G-elements wrap to negative/garbage `N` → OOB writes or silent under-computation, with no error. MXNet uses `index_t` (int64) for shapes everywhere else, so this is purely a dispatch-boundary truncation.
- **Fix:** Change `Launch`/`LaunchEx`/`mxnet_generic_kernel*` and `get_num_threads` to `index_t N`/`index_t i`; compute grid size in 64-bit.

### C4 — engine lapped-event check is a TOCTOU race that can under-synchronize GPU work **[pre-existing]**
- **Where:** `src/engine/threaded_engine.cc:708-713, 789-793` (`OnStartGPU`/`OnStartCPU`); counter in `engine.h:88,102`.
- **What:** The code decides a dependency event is safe via `ei.pool->IsLapped(ei.pool_index)` (reads `counter_.load()`), then waits on the cached `cudaEvent_t`. `counter_` is incremented independently in `OnCompleteGPU` → `GetNextEvent()` with no lock spanning the check-then-wait. Between the `IsLapped`==false check and `cudaStreamWaitEvent`, another worker can lap the slot and record a newer, unrelated op onto the same event.
- **Why it matters:** The consumer then waits on an event that no longer represents its true producer → a dependent kernel can launch before its input is ready. Silent, intermittent, load-dependent GPU data corruption — the worst kind. (The signed/unsigned loop at `threaded_engine.cc:821-833` in the same path is a related latent hazard.)
- **Fix:** Make the slot-validity check and the wait atomic w.r.t. event reuse — snapshot the event generation under `sync_obj.mutex` with the index, or hand out per-record `shared_ptr` events whose identity (not a recycled slot) is waited on. At minimum re-check `IsLapped` after queuing the wait and fall back to `cudaStreamSynchronize` on change.

---

## High

### H1 — `_backward_npi_percentile` does a CPU round-trip (D2H/H2D + sync) but isn't excluded from capture **[fork]**
- **Where:** `src/operator/numpy/np_percentile_op.cu:61` (backward), forward excluded at `:121`.
- **What:** `NumpyPercentileBackward<gpu>` copies all inputs to host, runs on CPU, copies back, with `cudaStreamSynchronize`. The forward `_npi_percentile` is flagged `false`; the backward isn't. Backward graphs *are* captured (the graph cache is keyed on `is_train`).
- **Why it matters:** Same class as C1/C2 — a CPU-roundtrip op inside the captured training segment hits capture-illegal sync/memcpy.
- **Fix:** Add `FIsCUDAGraphsCompatible=false` to `_backward_npi_percentile`. (Separately, the percentile GPU backward copying the *entire data tensor* to host for a single-threaded `std::sort` is a real perf cliff — flag for a proper GPU segmented-sort kernel like the interp backward got.)

### H2 — capture-safety guards self-disable in the default-on regime **[fork]**
- **Where:** `src/operator/linalg_impl.h:327` (`AssertGemmCaptureSafe`); `src/storage/pooled_storage_manager.h` Alloc miss path.
- **What:** Both guards gate on `dmlc::GetEnv("MXNET_ENABLE_CUDA_GRAPHS", false)`. The fork enables capture via `default_enable` (`cached_op.cc`, `static_alloc && static_shape`) with that env var **unset**. So precisely when default-on capture is active, the friendly "you hit a capture-illegal gemm/alloc" assert is skipped and you instead get the raw CUDA error 900 that aborts the graph.
- **Why it matters:** The intended defense-in-depth is off in the exact configuration you ship. Every future opt-out miss (C1/C2/H1 class) degrades to a cryptic abort instead of an actionable message.
- **Fix:** Drop the env gate (or OR-in the default-on condition) and always run the cheap `cudaStreamIsCapturing()` check. Mirror the cuBLASLt workspace pool, which already refuses to allocate during capture. **Strongly recommended companion:** a CI test that asserts the exact set of capturable ops, so opt-out drift is caught mechanically rather than by review.

### H3 — `NDArrayBase.__del__` frees a possibly-NULL/garbage handle **[pre-existing]**
- **Where:** `python/mxnet/_ctypes/ndarray.py:50`; same in `python/mxnet/cython/ndarray.pyx:64`.
- **What:** `__init__` permits `handle=None` and `__reduce__` reconstructs via `_ndarray_cls(None,)` during unpickling, but `__del__` unconditionally calls `MXNDArrayFree(self.handle)`. If construction/unpickling raises before `handle` is set, `__del__` runs on a half-built object → `AttributeError` inside `__del__` or a NULL passed to the C free.
- **Why it matters:** Lifetime hazard at the ctypes boundary; the pickling path is a realistic trigger.
- **Fix:** `handle = getattr(self, 'handle', None); if handle is not None: check_call(_LIB.MXNDArrayFree(handle))`. Mirror in the `.pyx`.

### H4 — build silently drops requested CUDA arches on older toolkits **[fork-adjacent / build]**
- **Where:** `CMakeLists.txt:118` default `MXNET_CUDA_ARCH`; `cmake/upstream/select_compute_arch.cmake:135`.
- **What:** The default arch list includes `sm_89/sm_90/sm_120`, which are unknown to CUDA toolkits older than 11.8/12.8. `CUDA_SELECT_NVCC_ARCH_FLAGS` silently filters unknown entries, so `12.0+PTX` can produce *no* PTX fallback and the wheel ships with no forward-compat image — failing at runtime on newer GPUs with "no kernel image is available." No configure-time error.
- **Why it matters:** A wheel built on an older toolkit silently omits promised Blackwell PTX.
- **Fix:** After arch selection, assert every requested arch produced a gencode flag (compare against `nvcc_archs_readable`) and `FATAL_ERROR` otherwise; or gate the default arch string on `CUDAToolkit_VERSION`. Also reconcile `config/build_gpu.cmake` (pins single `8.9`) with the root's multi-arch intent.

### H5 — blocking D2H + `cudaStreamSynchronize` inside `FCompute` (≈36 sites) **[pre-existing, some fork]**
- **Where (confirmed memcpy-then-sync):** `contrib/boolean_mask.cu:73`, `numpy/np_nonzero_op.cu:85`, `tensor/histogram.cu:96`, `tensor/dot-inl.cuh:704/821`, `tensor/cast_storage-inl.cuh:165/559`, plus `np_unique_op.cu`, `np_bincount_op.cu`, `np_percentile_op.cu`, `np_multinomial_op.cu`, `indexing_op.cu` (4), `np_lstsq/np_eig/np_eigvals-inl.h`, and others.
- **What:** Each fully stalls the GPU pipeline; all are capture-illegal. Some (`boolean_mask`, `nonzero`, `unique`, `histogram`) are dynamic-shape ops where it's hard to avoid, but the optimizer/dot/cast_storage ones serialize training steps unnecessarily.
- **Why it matters:** Lost overlap on the hot path + capture incompatibility.
- **Fix:** Where the value only gates a `CHECK` (histogram monotonic-bins, constraint checks), defer/skip under capture or move validation to an async path. Document the genuinely-unavoidable dynamic-shape cases and make sure they're all flagged `FIsCUDAGraphsCompatible=false`.

### H6 — `contrib/count_sketch.cu` calls `cudaDeviceSynchronize()` inside forward/backward **[pre-existing]**
- **Where:** `src/operator/contrib/count_sketch.cu:142,183` (the only two device-wide syncs in the operator tree).
- **What:** Full-device sync (not even stream-scoped) every loop iteration.
- **Why it matters:** Serializes *all* GPU work and all other streams; hard error under capture. Pure porting leftover.
- **Fix:** Remove; rely on stream ordering + `MSHADOW_CUDA_POST_KERNEL_CHECK`.

### H7 — `gather_nd` backward wraps OOB indices via modulo with no validation **[fork]**
- **Where:** `src/operator/tensor/indexing_op.cu:580-588`, `indexing_op.h:1550-1566`.
- **What:** `backward_gather_nd_gpu` computes `offset += strides[j] * ((idx + mshape[j]) % mshape[j])` with no bounds check (the `ScatterNDIndexChecker` is wired only into the forward). Any OOB index is wrapped into a valid-looking offset and `atomicAdd`'ed to the wrong cell.
- **Why it matters:** Silently corrupted gradients instead of an error — very hard to debug.
- **Fix:** Route gather_nd-backward indices through the same bound validation, or add an explicit in-kernel range guard; decide consistently whether OOB errors or is dropped (see C1).

### H8 — `index_add` / `index_update` / `np.insert` copy the whole index tensor to host + sync on every call **[fork]**
- **Where:** `src/operator/tensor/index_add_forward.cu:89-108`, `index_update.cu:93-112`, `numpy/np_insert_op_tensor-inl.h:42-72`.
- **What:** Each `cudaMemcpyAsync(D2H)` of the *entire* index buffer + `cudaStreamSynchronize` purely to produce a nicer error — even though an in-kernel `if (idx < 0 || idx >= dim) return;` guard was *also* added, making the kernel already memory-safe. `np.insert` additionally has a second, now-dead `invalid_ptr` check from `ObjToIndices` that's never read back.
- **Why it matters:** A mandatory blocking sync + full-index DtoH per call serializes the stream and scales with index size — a real training-loop perf regression, and capture-illegal.
- **Fix:** Validate on-device (small flag kernel + single-int D2H, like the histogram path) or gate the host check behind a debug flag; delete the dead second check.

### H9 — cuBLASLt descriptors created and destroyed on *every* GEMM call **[fork]**
- **Where:** `src/common/cuda/cublaslt_gemm.cc:291-325,383` (`MaybeCublasLtGemmImpl`).
- **What:** `op_desc`, `a_lay`, `b_lay`, `c_lay` (and on miss, `pref`) are built with `cublasLtMatmul*Create` at the top of every call and torn down at the bottom. Only the selected *algo* is cached; the descriptors are not, undermining the heuristic cache.
- **Why it matters:** This is the GEMM hot path (FC, matmul, batch_dot). 5 creates + sets + 4–5 destroys per GEMM is pure host overhead on top of the legacy path — measurable for small/medium GEMMs and warm-up.
- **Fix:** Cache the descriptors alongside the algo in the `GemmKey`-keyed entry (they're immutable for a given key); steady-state should be just `cublasLtMatmul` + workspace fetch.

### H10 — GPU storage pool OOM-retry sleeps while holding the per-device storage mutex **[fork]**
- **Where:** `src/storage/pooled_storage_manager.h:231-285`.
- **What:** `Alloc()` holds `Storage::Get()->GetMutex(dev_type_)` across the whole retry loop, which calls `cudaDeviceSynchronize()` and `std::this_thread::sleep_for(backoff)` (up to ~1s/attempt). `Free`/`DirectFree`/`ReleaseAll` contend on the same mutex.
- **Why it matters:** During the backoff sleep no other thread can `Free` memory back — so the very mechanism that could relieve the OOM is blocked. Serializes all device alloc/free for up to a second per failing alloc.
- **Fix:** Use a `unique_lock` and `unlock()`/`lock()` around the sleep+sync (or move retry/backoff outside the locked region).

### H11 — CPU random-resource handout now takes a mutex on every op **[fork]**
- **Where:** `src/resource.cc:151-167` + `src/common/lazy_alloc_array.h:96-99`.
- **What:** `GetResource()` for `kRandom`/`kParallelRandom` on CPU changed from a plain pointer deref to `cpu_rand_.Get(...)`, and `LazyAllocArray::Get()` unconditionally takes `create_mutex_` even for the already-created common case.
- **Why it matters:** Every CPU op using a random resource (dropout, samplers) now serializes on a global mutex → hurts multi-threaded CPU throughput. Previously zero synchronization here.
- **Fix:** Fast-path the pre-created `dev_id==0` resources without the array lock, or add a lock-free double-checked path in `LazyAllocArray::Get`.

### H12 — C-API exception guard only catches `std::exception` **[pre-existing]**
- **Where:** `src/c_api/c_api_error.h:42-57` (`MX_API_END`/`MX_API_END_HANDLE_ERROR`); ~151 sites in `c_api.cc` alone.
- **What:** No trailing `catch (...)`. Any non-`std::exception` throw (raw int, third-party/driver type, some dmlc throws) propagates out of `extern "C"`.
- **Why it matters:** Throwing across `extern "C"` into Python/other-language callers is UB → typically immediate `std::terminate` with no recoverable error string. Affects the entire public surface.
- **Fix:** Add `catch (...) { ... return MXAPIHandleException(dmlc::Error("unknown non-standard exception")); }` to both macros.

### H13 — `ObjectPool<T>` global per-type mutex on the engine's hottest allocation path **[pre-existing]**
- **Where:** `src/common/object_pool.h:143-165`.
- **What:** Every `OprBlock`, `VersionedVarBlock` (one per read/write dependency edge), `ThreadedOpr`, `ThreadedVar`, `GPUWorkerSyncInfo` is allocated/freed through a single process-wide mutex per type.
- **Why it matters:** These are created/destroyed for every op pushed and every dependency edge; all workers contend on one mutex per pool — serializing the scheduler's fast path, defeating the "fast allocation" purpose.
- **Fix:** Thread-local freelist with batch refill from a shared pool, or a lock-free Treiber stack; failing that, shard per worker thread.

### H14 — `NDArray.__hash__` returns `id(self)//16` while `__eq__` is element-wise **[pre-existing]**
- **Where:** `python/mxnet/ndarray/ndarray.py:436,440`.
- **What:** Legacy NDArray hashes by truncated identity but compares element-wise; the numpy frontend correctly sets `__hash__ = None`. Hash/eq contract violated, and `//16` can collide distinct live objects.
- **Why it matters:** Surprising behavior when NDArrays are used as dict keys / set members.
- **Fix:** Either make it unhashable (`__hash__ = None`, like the numpy frontend) or use plain `id(self)` and document that equality is element-wise.

### H15 — quantized asymmetric quantize loses sub-integer shift precision **[fork]**
- **Where:** `src/operator/quantization/dnnl/dnnl_quantize_asym-inl.h:213`.
- **What:** The v3 rewrite folds the affine offset into a single integer `dst_zp = nearbyint(shift)`; the old code kept the fractional part of `shift` per element. oneDNN zero-points must be integers, so the fractional bias (up to 0.5 LSB) is genuinely lost.
- **Why it matters:** Silent accuracy regression on quantized models (passes smoke tests, degrades quality). The common case has non-integer `shift = -min*scale`.
- **Fix:** Apply the reorder for scale only and fold the fractional shift into a pre-bias, or fall back to the elementwise CPU kernel when `shift` is non-integral.

### H16 — quantized concat / batch_norm affine fallbacks assume dense contiguous layout **[fork]**
- **Where:** `src/operator/quantization/dnnl/dnnl_quantized_concat.cc:30,89,225-284`; `dnnl_quantized_batch_norm.cc:31`.
- **What:** New hand-rolled affine-requant fallbacks index `Reorder2Default()` output as dense row-major with channel math `(i/inner)%C`, with no contiguity `CHECK`. For strided/view inputs (which the quantized graph does produce) channels get mixed. The mixed-dtype concat path also added a 2-reorder + 2-temp-buffer f32 round-trip per input on the hot path.
- **Why it matters:** Silent wrong results on strided inputs (highest-risk new quantization code); measurable perf regression on int8 concat graphs.
- **Fix:** Assert contiguity (or copy to contiguous first); unify the three near-identical affine-quant helpers (concat/bn/elemwise_add) into one tested utility; root-cause the single-reorder u8→s8 scaling instead of the f32 workaround.

### H17 — wheel build can package a stale `.whl`; OpenCV bundling split across two files with no enforced contract **[fork]**
- **Where:** `tools/build_cleanup_wheel.sh:193` (`ls -1 dist/*.whl | head -n1` picks lexical-first, not newest); `tools/build_cleanup_wheel.sh:96-147` vs `python/setup.py:178-186,302`.
- **What:** Wheel selection isn't `ls -1t`, so a leftover wheel can be validated/shipped instead of the one just built. Separately, the shell stages `libopencv_*` into `python/mxnet/lib/` and patches RUNPATH, but `setup.py` sets `bundled_libs=[]` and relies solely on a glob; building without the shell stager yields a wheel with `USE_OPENCV=ON` metadata + `$ORIGIN/lib` RUNPATH but no `.so` → import-time failure.
- **Why it matters:** Supply-chain artifact correctness depends on invocation order that isn't enforced.
- **Fix:** Select with `ls -1t` (or capture `python -m build`'s printed path). Consolidate OpenCV staging+RUNPATH into one place, and hard-fail in `setup.py` when `USE_OPENCV=ON` but `lib/libopencv_*` is absent. (`release_provenance.py` is good defense-in-depth but only runs because the script happens to call it.)

---

## Medium

### M1 — CUB global-reduce workspace queried with a different iterator type than the real reduce **[fork]**
- `src/operator/tensor/reduce_cub.cu:114-116`, `broadcast_reduce_op.cc:62-74`. Temp-storage size is queried with `double*` but the reduce runs over a `transform_iterator`. Holds for current CCCL but is an undocumented invariant; a future CCCL that sizes temp storage by iterator could overflow the caller-supplied workspace. Query with the actual iterator type, or assert sizes match before launch.

### M2 — histogram range-path bin target computed in reduced precision **[fork]**
- `src/operator/tensor/histogram.cu:39-45`. The no-range kernel was fixed to compute the target in `double`, but the range (`has_cnt`) kernel still mixes `DType` data with `double` edges → off-by-one bin placement for fp16 near an edge, diverging from NumPy/CPU. Promote `data` to `double` here too.

### M3 — cuDNN BatchNorm inference temp-gamma can alias the workspace **[fork]**
- `src/operator/nn/cudnn/cudnn_batch_norm.cu:185-196`. Training carves workspace+gamma from one allocation; inference/backward call `get_space_internal` a second time on the same resource. Correct today (one consumer per call) but fragile — a future second temp-space user silently corrupts the gamma buffer. Use the single-allocation-with-offsets pattern uniformly.

### M4 — RNN re-issues data descriptors + temp-size queries + clip on every forward **[fork]**
- `src/operator/rnn-inl.h:893-900`. The four RNN-data descriptors, `cudnnGetRNNTempSpaceSizes`, and clip state are re-set on every call though they're invariant for fixed-shape non-`use_sequence_length` cases. Cache keyed on (seq_len, batch, is_train); set clip once at Init. (Also `rnn-inl.h:806` uses a fully-synchronous `cudaMemcpy` on the legacy default stream for seq-lengths — use the async form on the engine stream like `:890` does.)

### M5 — CPU fp16 conv/FC fall back to a naive O(MNK) triple loop **[fork]**
- `src/operator/nn/convolution-inl.h:592-627`, `fully_connected-inl.h:296-360`. After removing the `LOG(FATAL)`, the only CPU fp16 path is a hand-rolled loop with cache-hostile transposed access — 1–2 orders of magnitude below BLAS. Acceptable as a correctness fallback, but prefer upcasting to fp32 and dispatching through `linalg_gemm` (as the bf16 path does). Also `FCForwardCPUHalf` hard-asserts `req==kWriteTo` though the helper supports `kAddTo`.

### M6 — `np_norm` backward leaks a `cudaMalloc` on exception + unchecked CUDA calls **[pre-existing]**
- `src/operator/numpy/linalg/np_norm-inl.h:411`. `cudaMalloc(&mapper_instance,...)` is unwrapped, `cudaStreamSynchronize` unchecked, and the matching `cudaFree` is only on the normal path — a throw in between leaks device memory. Per-call tiny `cudaMalloc`/`cudaFree` on a hot backward path; use `ctx.requested[].get_space_typed` + `CUDA_CALL`.

### M7 — `contrib/proposal*` use abort-on-error CUDA macros + raw per-call alloc **[pre-existing]**
- `src/operator/contrib/proposal.cu`, `multi_proposal.cu`. ~20 CUDA calls wrapped only in a file-local `FRCNN_CUDA_CHECK` (LOG(FATAL)/abort, not the project-standard throwing `CUDA_CALL`), plus per-forward `cudaMalloc`/`cudaFree` bypassing the pool and a DtoH+sync. Inconsistent error semantics, memory fragmentation. Replace with `CUDA_CALL`; route allocations through `ctx.requested`.

### M8 — `int`-typed flat thread indices in ~23 hand-written kernels **[pre-existing]**
- `contrib/proposal.cu` (7), `multi_proposal.cu` (8), `multibox_target.cu`, `multibox_prior.cu`, `count_sketch.cu:70`, `cudnn_batch_norm.cu:84`, `linalg_impl.h:1634`. Same overflow class as C3 but in bespoke kernels. Use `index_t` for loop var and bound.

### M9 — `IndexTensorToVector` (GPU) reads an async-copied host buffer without syncing **[pre-existing]**
- `src/operator/sequence_op_common.h:42`. `malloc`s a host buffer, `cudaMemcpyAsync(D2H)`, then loops on host with no stream sync — relies on default-stream semantics that don't hold for non-default streams → can read stale data. Used by SequenceLast/Mask/Reverse. Add a sync (or make it synchronous); cache the buffer.

### M10 — engine var-keyed exception map can grow unbounded and alias reused VarHandles **[fork]**
- `src/engine/naive_engine.cc:265-320`. `var_exceptions_[var]` (keyed by a raw `VarHandle` pointer) is only cleared by WaitForAll/WaitForVar; a never-drained failing op leaks, and a freed-then-reused var address can throw a stale, unrelated exception. Erase the var's entry in `DeleteVariable`.

### M11 — `SyncCopyToCPU` adds an unconditional reorder + extra `WaitToRead` on oneDNN builds **[fork]**
- `src/ndarray/ndarray.cc:2410-2414`, `serialization/cnpy.cc:273,479`. The old code reordered only `IsDNNLData()`; now every default-storage CPU array gets an engine push + extra full `WaitToRead` round-trip per `.asnumpy()` (and mutates the source layout as a side effect of a logically read-only copy). Guard with `if (this->IsDNNLData())`.

### M12 — `SetTBlob()` (const) silently reorders in place via `const_cast` **[fork]**
- `src/ndarray/ndarray.cc:966-984`. A hard `CHECK(!IsDNNLData())` became `const_cast<NDArray*>(this)->SelfReorder2Default()` — a synchronous storage realloc + reorder inside a `const` method on the per-op execution path, outside var serialization. Thread-safety hazard if the chunk is shared. Keep the reorder explicit/var-serialized; don't mutate through const.

### M13 — over-cap pool Free skips the reuse-event sync before `cudaFree` **[fork]**
- `src/storage/pooled_storage_manager.h:165-183`. When a bucket is at its retention cap, `Free()` calls `contextHelper_->Free` immediately, discarding `handle.sync_obj`. On GPU this can free memory still referenced by in-flight kernels → device use-after-free. Honor `handle.sync_obj` (await recorded events) before the direct free, as the event-aware release path does.

### M14 — per-op heap allocation of NDArray wrappers on the imperative invoke path **[pre-existing]**
- `src/imperative/imperative_utils.h:518-544`. `new NDArray(*i)` per input/output for essentially every eager op → multiple heap allocs + shared_ptr atomics purely to build temp arg vectors. At high op rates this dominates eager dispatch. Use a reusable thread-local arena / `reserve`d vector with emplace-by-value.

### M15 — per-op ctypes marshaling rebuilds lists/strings every imperative call **[pre-existing]**
- `python/mxnet/_ctypes/ndarray.py:108`, `register.py:218-254`. Each op call does `c_handle_array`, `c_str_array`, and `str(s)` over every kwarg, rebuilding ctypes arrays. Dominant Python-side cost for small ops. Ensure the cython/`_ffi` fast path is actually compiled & selected in the wheel (verify `_cy3` is built); fast-path empty-keys and avoid `str()` on already-strings.

### M16 — `dmlc::GetEnv` on hot paths (per-op env reads) **[fork + pre-existing]**
- `src/operator/tensor/dot-inl.h:~232` (`MXNET_CUDA_ALLOW_TENSOR_CORE` re-read on every batch_dot forward); also `src/common/denorms.h:111` does a seq_cst atomic load per executed op. Cache via `static const bool` (the standard `GetEnvAllowTensorCore()` idiom) / use `memory_order_relaxed`.

### M17 — DLPack manager uses raw `new`/`delete` with hand-written deleters **[pre-existing]**
- `src/ndarray/ndarray.cc:485-503`. DLPack handles cross to PyTorch/CuPy; manual `new`/`delete` with no RAII guard between `new` and hand-off is a double-free/leak surface. Use a single owning struct with `unique_ptr`, release only at the no-throw hand-off.

### M18 — runtime DNNL scale-memory lifetime relies on implicit handle-sharing **[fork]**
- `src/operator/subgraph/dnnl/dnnl_common.h:58`, `dnnl_quantized_concat.cc:259`. Local `dnnl::memory scale_mem` is registered into the args map and the function returns; `ConvertWeightBias2DNNL(..., submit=false)` would submit after the local is destroyed → use-after-free. Both current callers pass `submit=true`, so it's latent. Cache scale memory as a member (as the dequantize/quantize ops now do) or `CHECK(submit)` when scales are present; document the contract.

### M19 — uint8 requantize abandons the cached oneDNN primitive for a per-call CPU fallback **[fork]**
- `src/operator/quantization/dnnl/dnnl_requantize-inl.h:175`. uint8 now always routes to `RequantizeForward<cpu>` (scalar/OMP loop + forced `Reorder2Default()` + `InvalidateDNNLData()` each call), while int8 keeps the JIT reorder. uint8 requantize is on the ReLU-fused inference hot path → real throughput regression + format churn. Implement the uint8 reorder with DST scale + zero-point (oneDNN v3 supports asymmetric reorder).

### M20 — npy/npz loader: untrusted-input hazards **[pre-existing]**
- `src/serialization/cnpy.cc`. (a) `:556-557` `string_view{name.data(), name.size()-1}` underflows to ~2⁶⁴ on a zero-length zip entry name → OOB read (this is the highest-priority security item — fix with `if (name.size() < 5) continue;`). (b) `:254` `header[loc+9]` / `:233` `substr(loc+16,4)` index past the header on crafted files (UB / throw) — bounds-check first. (c) `:241-247` shape dims via `stoi` with no range check feed `TShape::Size()` which multiplies with no overflow check → size confusion / huge-alloc DoS. (d) `:219-225,525-533` header-length decode uses `get() >> 8` (always 0) so only the low byte is read — headers >255 bytes are mis-parsed, weakening all the validation above. Also `NDArray::Load` reads `type_flag` from the stream straight into `mshadow_sizeof` (controlled abort, not corruption — but whitelist it for a graceful error). These are reachable from loading any untrusted `.npz`/`.params`.

### M21 — build-system robustness gaps **[fork/build]**
- `CMakeLists.txt:651` appends a non-existent `${OpenMP_EXE_LINKER_FLAGS}` (silent no-op — rely on `OpenMP::OpenMP_CXX`). `cmake/Modules/FindCUDNN.cmake:20` defaults `CUDNN_ROOT` to an *include* dir and has **no cuDNN version guard** (ABI-mismatched cuDNN links silently). `CMakeLists.txt:734-775` globs sources without `CONFIGURE_DEPENDS` → adding/removing files doesn't reconfigure (stale/wrong incremental builds). `cmake/ChooseBlas.cmake:27-34` auto-switches to MKL whenever oneAPI is installed, overriding the documented `Open` default → non-reproducible builds. `FindNCCL.cmake:63` matches versioned `libnccl.so.2`; dev links dynamic while Distribution links `nccl_static` — different artifacts.

### M22 — supply-chain / dependency pinning **[fork]**
- `tools/dependencies/download_utils.py:35`: missing-checksum silently degrades to *no verification* with only a stderr warning; any `--version` override disables integrity checks. `build_opencv.py:135`: extracted source dir is cache-keyed on existence only — a partial/poisoned extraction is reused forever (no re-validation against the verified archive). `3rdparty/mshadow` is a plain tracked dir, not a pinned submodule; `dmlc-core` points at a personal fork branch, `onednn`/`tvm` at untagged dev tips. For release/CD, make missing-pin a hard error, extract-to-temp-then-rename, and pin deps to upstream tags.

### M23 — GPU test lane runner only remaps `mx.gpu(0)` and won't use `CUDA_VISIBLE_DEVICES` **[fork]**
- `tools/run_pytest_gpu_lane.py:39-46`. Monkeypatches only `mx.gpu(0)`; any test using `mx.gpu(1)`+ escapes the lane and concurrent lanes collide on shared physical devices → flaky/false passes. Multi-GPU tests aren't serialized/skipped. Document the limitation and skip multi-GPU tests under this runner, or fix the underlying error so `CUDA_VISIBLE_DEVICES` masking can be used (the correct mechanism). Also `run_wheel_full_test.sh` pip-install steps aren't error-checked, so an env failure masquerades as a fleet of test failures.

---

## Low (condensed)

- **L1 [fork]** `src/imperative/cuda_graphs.h` `MakeGraphExec`: `static int num_graph_creations++` is a non-atomic RMW raced across concurrent multi-GPU captures (cosmetic: diagnostic IDs/dotfile names). Make it `std::atomic<int>`.
- **L2 [fork]** `cublaslt_gemm.cc:154,199,275` unchecked `cudaGetDevice`/`cudaSetDevice`/`cublasGetStream` — a failed `cublasGetStream` silently runs the matmul on the default stream. Check status.
- **L3 [fork]** `cublaslt_gemm.cc:137-168` persistent per-stream workspace is keyed on raw `cudaStream_t` (reused after stream destroy), never freed, and capture-safety depends on an undocumented "never grows after capture" invariant. Add a `static_assert`/CHECK that the alloc floor ≥ the heuristic cap; comment it as a capture invariant.
- **L4 [fork]** `cublaslt_gemm.cc:58-76` `GemmKey` omits `scale_dtype` (safe today via an implicit invariant). Fold it in or CHECK the relationship.
- **L5 [fork]** `cuda_graphs.h:423-589` `VerifyReplay` does raw `cudaMalloc`/sync with no device guard → wrong-device alloc on multi-GPU (debug-only). Wrap in a device guard.
- **L6 [fork]** `imperative_utils.h:59` `InvalidateOutputs` dropped `kNullOp` with no rationale, bundled into the graphs branch — a oneDNN stale-buffer correctness risk. Document or revert.
- **L7 [fork]** `rnn-inl.h` `SyncDgrad()` is now an empty no-op but `dgrad_sync_event_`/`dgrad_sync_needed_` machinery is retained as dead code. Remove.
- **L8 [fork]** `cudnn_deconvolution-inl.h:427` re-enables `TENSOR_OP_MATH_ALLOW_CONVERSION` for fp32 deconv (opt-in via env, but a silent precision change vs. default math). Document.
- **L9 [fork]** `batch_norm.cu:326` moved running-stat update from backward to forward → training-mode forward-only now updates running stats (matches PyTorch; observable behavior change). Note in changelog.
- **L10 [fork]** cuDNN dropout `CHECK_NE(kAddTo)` while MKL/native paths support it → backend-dependent crash on `kAddTo`. Implement or fall back to native.
- **L11 [pre-existing]** `c_api_common.h:60-90` thread-local return buffers return interior pointers invalidated by the next API call on that thread, with duplicated `*2`/`*3` offset magic. Document the lifetime contract; factor the packing into a helper.
- **L12 [pre-existing]** `threaded_engine.cc:585` takes `finished_m_` on every `OnComplete` solely to decrement an already-atomic `pending_`. Decrement lock-free; lock only to notify when it hits 0.
- **L13 [pre-existing]** `CheckDuplicate` (engine duplicate-var validation) is compiled out of all release builds (`ENGINE_DEBUG=0`) → duplicate vars surface as nondeterministic hangs instead of a clear fatal. Make a cheap O(n) dedup check always-on.
- **L14 [pre-existing]** `engine/engine.cc` vs `threaded_engine.cc` parse `MXNET_ENGINE_TYPE` "Async" with divergent matching → engine could be built non-async while `IsEngineAsync()` returns true. Resolve once at construction, store on the engine.
- **L15 [pre-existing]** Multiple bare `except:` swallow root causes and `KeyboardInterrupt`: `ndarray/ndarray.py:805,1389` (the latter also formats `type(array)` instead of `type(source_array)`), `numpy/multiarray.py:1130`, `io/utils.py:56`, `ndarray/contrib.py:332,460`, `sparse.py:817`. Use `except Exception as e: ... from e`.
- **L16 [pre-existing]** `base.py:62` `py_str` crashes on a NULL `MXGetLastError()` (AttributeError inside the error handler). `base.py:48` `integer_types` omits `np.integer` subtypes → cryptic failures on `np.int16` indices. Guard `py_str`; use `(int, np.integer)`.
- **L17 [pre-existing]** `gluon/data/dataloader.py:130` registers `ForkingPickler` reducers as an import side effect (process-global), and the fork-based worker pool after CUDA init is the classic fork-after-GPU pitfall. Prefer `spawn` when a GPU context may exist; replace `print(msg)` at `:603` with logging.
- **L18 [fork]** Op-layer minor: `broadcast_reduce-inl.h:330` redundant triple `SetInitValue`; `np_broadcast_reduce_op.h:36` dead nullary `set_to_nan::Map()`; `np_repeat_op-inl.h:178` `int` flat-index arithmetic; `np_bincount_op-inl.h:140` weight-backward lacks a bin-range guard; `quantized_elemwise_mul.cc:185` int32/float output paths unrounded while int8 was fixed.
- **L19 [fork]** `tools/run_fp16_remote_smoke.sh` defaults `REPO_URL`/`BRANCH` to a personal fork + a dated branch that will rot; `nvidia-smi -L | wc -l` without `2>/dev/null` can count an error line as a GPU. `run_wheel_full_test.sh:214` hardcodes `-n 4` ignoring `PARALLEL_CPU`. `build_gpu.cmake` lacks ccache/unity/PCH wiring (build-time bloat).

---

## Cross-cutting themes (what to fix structurally, not one-by-one)

1. **Opt-out drift under default-on capture.** Capture-safety is a hand-maintained `FIsCUDAGraphsCompatible=false` denylist, but capture is default-*on*. Every confirmed miss (C1, C2, H1) is the same shape: a forward/sibling was excluded, its twin wasn't. Fix the two structural gaps (H2: make `AssertGemmCaptureSafe` + the GPU pool capture-aware unconditionally) **and add a CI test asserting the exact capturable-op set** so drift is caught mechanically.
2. **Blocking host syncs inside `FCompute`.** ~36 sites (H5/H6/H8/M9). They cost overlap in eager mode and are illegal under capture. Audit them: defer validation off the hot path, or flag + document the unavoidable dynamic-shape ones.
3. **64-bit index discipline at the GPU dispatch boundary.** C3 + M8 are the same bug (truncating `size_t`→`int`) at the most-used launcher and in bespoke kernels. Fixing `mxnet_op.h` covers most of the tree.
4. **Coarse locks on the engine fast path.** H13 (object pool), H10/H11 (storage/resource mutexes), L12 (completion path) all serialize per-op work. They undercut the per-device threading design.
5. **Duplicated, untested quantization fallbacks.** H15/H16/M18/M19 — a lot of hand-written affine-requant code was added that duplicates itself and assumes contiguous layout. Consolidate into one tested helper and add per-OC + strided-input tests.
6. **C-API / ctypes boundary hardening.** H3, H12, H14, L11, L15, L16 — exception safety, handle lifetime, and the hash/eq contract all need attention before third parties build on this.

## Verified clean (checked, not problems)

The CUDA-graphs core itself is well-built: the capture-aware cuBLASLt workspace pool, warm-up
tempspace pointer-stability checks, the differential-replay verifier, and the RNG resource
allow-list (device-resident philox/dropout counters advance under replay; host curand stays
excluded) are all correct. The macOS `shm_unlink` double-unlink fix
(`cpu_shared_storage_manager.h`) is correct and leak-free. The bf16 CPU fallback upcasts
in/out to fp32 consistently. `release_provenance.py` is solid defense-in-depth (verifies CMake
flags, commit-hash match, RUNPATH, SONAME bundling). `tests/run_unit_test_shards.sh.in` and the
hardened `make_shared_dependencies.sh` download path are clean. The core C-API is uniformly
`API_BEGIN/END`-wrapped and image/recordio loaders bounds-check their headers — the npy/npz
loader (M20) is the one real untrusted-input gap.
