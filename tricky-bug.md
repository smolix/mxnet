# Tricky scheduling/threading/memory bug report

Date: 2026-05-27

Scope: scheduling, worker lifecycle, C API validation, resource management,
profiler/runtime/storage races, CUDA stream/workspace/cache issues, leak-prone
exception paths, distributed KVStore/ps-lite encoding, and selected
out-of-bounds operator kernels.

Current local build context:

- `USE_CUDA=ON`
- `USE_DIST_KVSTORE=OFF`

Verification policy: a bug is counted as directly verified only when this
change contains a test that triggers the bad behavior. Some tests are
sanitizer, GPU, or dist-KVStore gated; those are listed separately so the
report does not overclaim what ran in this build.

## Fix Summary

- Engine scheduling now propagates async callback failures from threaded
  read-only/no-var ops and from `NaiveEngine`.
- Engine lifecycle and C API push wrappers now reject invalid inputs before
  allocation and reliably hit `API_END`.
- `MXNET_CPU_TEMP_COPY` and `MXNET_CPU_PARALLEL_RAND_COPY` now reject zero and
  negative values before divide-by-zero or huge allocation paths.
- Packed NumPy `CachedOp(out=...)` now passes and returns caller-owned outputs
  correctly.
- C API stream handling now exposes pointer-sized `MXPushStreamDepEx` and
  `MXGetCurrentStreamEx`, and Python DLPack paths use them.
- Lazy/runtime/profiler/storage singletons and maps now have locking where
  they had racy first-use or concurrent read/write paths.
- CPU and GPU index/ROI/sparse operators now validate indices and ROI batch ids
  before touching memory.
- CUDA multi-LAMB/LANS host buffers, cuBLASLt workspace use, and cuDNN op-cache
  iterator use were fixed for concurrent/sanitizer-sensitive paths.
- Distributed KVStore/ps-lite encoders now validate byte and lens overflow,
  serialize cache mutation, and return deep snapshots instead of exposing
  mutable cached `SArray` storage.
- SyncBatchNorm's reusable barrier now has a generation counter.
- Custom-op profiler begin/end now uses RAII so exceptions cannot leave stale
  thread-local profiler state.

## Test Code Added

- `tests/cpp/engine/threaded_engine_test.cc`
  - Async callback error propagation for threaded and naive engines.
  - Negative bulk-size validation.
  - idempotent `ThreadedEnginePerDevice::Start()`.
  - `MXEnginePushAsyncND` / `MXEnginePushSyncND` negative-count and `API_END`
    coverage.
- `tests/cpp/misc/tricky_bug_test.cc`
  - `.npy` C API name out-param cleanup.
  - TSAN-style stress tests for `LazyAllocArray`, runtime type registration,
    `CustomOpProfiler::Get`, `ProfilerScope`, storage profiler, and CPU
    `all_finite`.
  - Custom-op profiler exception cleanup proof.
  - SyncBatchNorm barrier reuse proof.
  - CUDA stream ABI proof.
  - `#if MXNET_USE_DIST_KVSTORE` KVStore/P3Store ps-lite overflow, snapshot,
    and concurrency tests.
- `tests/python/unittest/test_tricky_bug_repros.py`
  - CPU repros for `CachedOp(out=...)`, `_contrib_index_copy`, `sparse.retain`,
    ROIAlign, PSROIPooling, DeformablePSROIPooling, RROIAlign, and CUDA-gated
    `npx.index_add` / `npx.index_update`.
- `tests/python/unittest/test_resource_env_validation.py`
  - Process-isolated resource env-var repros.
- `tests/python/unittest/test_tricky_bug_sanitizer_repros.py`
  - LSAN-oriented repros for NumPy random temporary pointer arrays and Python
    wrapper exception cleanup paths.
- `tests/python/gpu/test_tricky_bug_gpu_repros.py`
  - CUDA/ASAN/LSAN/compute-sanitizer repros for stream dependency, GPU bounds
    checks, multi-LAMB/LANS, cuBLASLt workspace, and cuDNN op-cache stress.
- `tests/CMakeLists.txt`
  - Adds `cpp/misc/tricky_bug_test.cc` to the `misc_storage` unit-test binary.

## Verification Performed

Successful targeted checks:

```bash
cmake --build build --target libmxnet.so
cmake --build build --target mxnet_unit_tests_misc_storage
ctest --test-dir build --output-on-failure -R AllTestsInmxnetUnitTests_misc_storage
env MXNET_LIBRARY_PATH=/home/smola/mxnet/build/libmxnet.so .venv-mxnet/bin/python -m pytest -q tests/python/unittest/test_tricky_bug_repros.py tests/python/unittest/test_resource_env_validation.py tests/python/unittest/test_tricky_bug_sanitizer_repros.py
env MXNET_LIBRARY_PATH=/home/smola/mxnet/build/libmxnet.so .venv-mxnet/bin/python -m pytest -q tests/python/gpu/test_tricky_bug_gpu_repros.py
python3 -m py_compile python/mxnet/dlpack.py python/mxnet/numpy/multiarray.py tests/python/unittest/test_tricky_bug_repros.py tests/python/unittest/test_resource_env_validation.py tests/python/unittest/test_tricky_bug_sanitizer_repros.py tests/python/gpu/test_tricky_bug_gpu_repros.py
git diff --check
```

Results:

- Python CPU tricky repros: `18 passed, 4 skipped`.
- Python GPU repro file: `13 skipped` in this Python environment.
- Rebuilt `misc_storage` CTest target: passed after final C++ test edits.
- `git diff --check`: clean.

Full sweep:

```bash
ctest --test-dir build --output-on-failure -E tests_NOT_BUILT
```

The first sandboxed full sweep exposed CUDA error 304 on several C++ targets.
Per user guidance, those were rerun outside the sandbox and passed. A full
outside-sandbox sweep then passed:

- `15/15` CTest targets passed.
- Total time: `1206.45 sec`.

After that full sweep, the affected rebuilt `misc_storage` target and Python
tests were rerun as listed above.

## Directly Verified Bugs

1. Threaded engines dropped async callback errors for read-only ops.
   - Fix: `src/engine/threaded_engine.cc`, `src/engine/threaded_engine.h`
   - Proof: `Engine.ThreadedReadOnlyAsyncExceptionReachesWaitForAll`

2. Threaded engines dropped async callback errors for no-var ops.
   - Fix: `src/engine/threaded_engine.cc`, `src/engine/threaded_engine.h`
   - Proof: `Engine.ThreadedNoVarAsyncExceptionReachesWaitForAll`

3. `NaiveEngine` ignored async callback errors.
   - Fix: `src/engine/naive_engine.cc`
   - Proof: `Engine.NaiveAsyncCallbackExceptionReachesWaitForAll`

4. Negative engine bulk size poisoned thread-local bulk state.
   - Fix: `src/engine/threaded_engine.h`
   - Proof: `Engine.NegativeBulkSizeIsRejectedAndDoesNotPoisonThreadLocalState`

5. `ThreadedEnginePerDevice::Start()` was not idempotent.
   - Fix: `src/engine/threaded_engine_perdevice.cc`
   - Proof: `Engine.ThreadedEnginePerDeviceStartIsIdempotent`

6. `MXEnginePushAsyncND` / `MXEnginePushSyncND` allocated vectors before
   validating signed counts.
   - Fix: `src/c_api/c_api.cc`
   - Proof: `Engine.PushFuncNDRejectsNegativeCountsBeforeAllocatingVectors`

7. `MXEnginePushAsyncND` / `MXEnginePushSyncND` returned before `API_END`.
   - Fix: `src/c_api/c_api.cc`
   - Proof: `Engine.PushFuncNDReachesApiEndForProfiling`

8. `.npy` `MXNDArrayLoad` left name out-params stale.
   - Fix: `src/c_api/c_api.cc`
   - Proof: `TrickyBug.MXNDArrayLoadNpyClearsNameOutputs`

9. CPU resource copy-count env vars accepted zero and negative values.
   - Fix: `src/resource.cc`
   - Proof: `test_resource_env_validation.py`

10. Packed NumPy `CachedOp(out=...)` ignored/miswrapped caller outputs.
    - Fix: `src/api/cached_op_api.cc`
    - Proof: `test_numpy_cached_op_invoke_with_out_uses_caller_output`

11. `_contrib_index_copy` trusted CPU row indices.
    - Fix: `src/operator/contrib/index_copy.cc`
    - Proof: `test_contrib_index_copy_rejects_out_of_bounds_indices_cpu`

12. `sparse.retain` trusted signed CPU row indices.
    - Fix: `src/operator/tensor/sparse_retain-inl.h`
    - Proof: `test_sparse_retain_rejects_out_of_bounds_indices_cpu`

13. CPU ROIAlign accepted invalid batch ids and zero spatial input.
    - Fix: `src/operator/contrib/roi_align.cc`
    - Proof: ROIAlign tests in `test_tricky_bug_repros.py`

14. CPU PSROIPooling, DeformablePSROIPooling, and RROIAlign trusted invalid
    ROI batch ids.
    - Fixes: `src/operator/contrib/psroi_pooling.cc`,
      `src/operator/contrib/deformable_psroi_pooling.cc`,
      `src/operator/contrib/rroi_align.cc`
    - Proof: pooling/align tests in `test_tricky_bug_repros.py`

15. SyncBatchNorm reusable barrier could release a later generation early.
    - Fix: `src/operator/contrib/sync_batch_norm-inl.h`
    - Proof: `TrickyBug.SyncBatchNormBarrierIsReusableAcrossGenerations`

16. Custom-op profiler begin/end could remain imbalanced after exceptions.
    - Fix: `src/operator/custom/custom-inl.h`,
      `src/profiler/custom_op_profiler.h`
    - Proof: `TrickyBug.CustomOpProfilerScopeIsClearedWhenCustomOpThrows`

## Sanitizer-Oriented Proof Tests

These tests exercise real race/leak paths but need TSAN, ASAN, LSAN, or
compute-sanitizer for the failure signal.

- `LazyAllocArray::Get` concurrent first access.
  - Fix: `src/common/lazy_alloc_array.h`
  - Proof: `TrickyBug.LazyAllocArrayConcurrentFirstGetIsTsanClean`
- Runtime type-key lookup racing lazy registration.
  - Fix: `src/runtime/object.cc`
  - Proof: `TrickyBug.RuntimeTypeKeyLookupDuringLazyRegistrationIsTsanClean`
- `CustomOpProfiler::Get` unsafe first-use/singleton/map access.
  - Fix: `src/profiler/custom_op_profiler.h`
  - Proof: `TrickyBug.CustomOpProfilerSingletonFirstUseIsTsanClean`
- `ProfilerScope` shared string race.
  - Fix: `src/profiler/profiler.cc`, `src/profiler/profiler.h`
  - Proof: `TrickyBug.ProfilerScopeConcurrentSetGetIsTsanClean`
- CPU/GPU storage profiler map/counter races.
  - Fix: `src/profiler/storage_profiler.cc`,
    `src/profiler/storage_profiler.h`
  - Proof: storage profiler tests in `tricky_bug_test.cc`
- CPU `all_finite` parallel writes to one scalar.
  - Fix: `src/operator/all_finite.cc`
  - Proof: `TrickyBug.AllFiniteCpuParallelWritesAreTsanClean`
- NumPy random temporary pointer arrays leaked on NDArray parameter paths.
  - Fix: `src/api/operator/numpy/random/np_laplace_op.cc`,
    `src/api/operator/numpy/random/np_multinomial_op.cc`
  - Proof: LSAN tests in `test_tricky_bug_sanitizer_repros.py`
- Python ctypes multi-output wrapper exception cleanup.
  - Fix: `python/mxnet/_ctypes/ndarray.py`,
    `python/mxnet/_ctypes/cached_op.py`
  - Proof: `test_multi_output_wrapper_exception_path_is_lsan_clean`

## CUDA/GPU-Gated Proof Tests

The C++ CUDA-enabled CTest suite passed outside the sandbox. The Python GPU
proof file was skipped in the current Python environment, but contains tests
for these fixes:

- Pointer-sized stream C API and Python DLPack call sites.
  - Fix: `include/mxnet/c_api.h`, `src/c_api/c_api.cc`,
    `include/mxnet/ndarray.h`, `src/ndarray/ndarray.cc`,
    `python/mxnet/dlpack.py`, `python/mxnet/numpy/multiarray.py`
  - Proof: `TrickyBug.StreamHandleApiUsesPointerSizedStorage`,
    `test_mx_push_stream_dep_does_not_use_freed_ndarray_handle`,
    `test_mx_get_current_stream_is_lsan_clean`
- GPU bounds checks for `index_copy`, `sparse.retain`, ROIAlign,
  PSROIPooling, DeformablePSROIPooling, `npx.index_add`, and
  `npx.index_update`.
  - Fix: corresponding `.cu` files under `src/operator`
  - Proof: GPU tests in `test_tricky_bug_gpu_repros.py` and CUDA-gated tests
    in `test_tricky_bug_repros.py`
- GPU multi-LAMB/multi-LANS host allocation cleanup.
  - Fix: `src/operator/contrib/multi_lamb.cu`,
    `src/operator/contrib/multi_lans.cu`
  - Proof: `test_multi_lamb_lans_gpu_host_allocations_are_lsan_clean`
- cuBLASLt concurrent workspace use.
  - Fix: `src/common/cuda/cublaslt_gemm.cc`
  - Proof hook:
    `test_cublaslt_shared_workspace_concurrent_gemm_is_sanitizer_clean`
- cuDNN op-cache iterator invalidation.
  - Fix: `src/operator/cudnn_ops.h`
  - Proof hook:
    `test_cudnn_op_cache_concurrent_insert_is_sanitizer_clean`

## Dist-KVStore / ps-lite Status

Fixes were applied for the distributed ps-lite/KVStore bugs:

- ps-lite `lens` `int` overflow and byte-size overflow checks in
  `src/kvstore/kvstore_dist.h` and `src/kvstore/p3store_dist.h`.
- Cache mutation protected by `mu_` during encoding.
- `KVStoreDist` and `P3StoreDist` return cloned encoding snapshots instead of
  shallow `SArray` aliases to cached state.
- Row-sparse encoding clears and returns snapshots under lock.

Added guarded tests:

- `TrickyBug.KVStoreDistDefaultKeyEncodingRejectsPsLiteLensOverflow`
- `TrickyBug.P3StoreDistDefaultKeyEncodingRejectsPsLiteLensOverflow`
- `TrickyBug.P3StoreDistDefaultKeyEncodingReturnsSnapshot`
- `TrickyBug.KVStoreDistRowSparseEncodingReturnsSnapshot`
- `TrickyBug.KVStoreDistDefaultKeyEncodingConcurrentAccessIsTsanClean`

These tests are behind `#if MXNET_USE_DIST_KVSTORE`. The current local build has
`USE_DIST_KVSTORE=OFF`, so they compiled out and did not execute here. The
ordinary `AllTestsInmxnetUnitTests_kvstore` CTest target passed in the current
build, but it does not compile the distributed ps-lite whitebox tests.

## Not Counted As Directly Verified

- The custom sparse-output dependency index fix in
  `src/operator/custom/custom-inl.h` is applied, but is not counted as directly
  verified. A standalone C++ proof through `CustomOperator::Push` timed out
  before callback completion in both sandboxed and outside-sandbox runs, so it
  was not left in the suite. The code fix is small and direct: the loop now
  advances the tag index for every array, including default-storage outputs and
  non-output arguments.
- The dist-KVStore whitebox tests are present but require a
  `USE_DIST_KVSTORE=ON` build to compile and run.
- Sanitizer-oriented tests pass in a normal build but need the corresponding
  sanitizer runtime to prove the race/leak signal.
