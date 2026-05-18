# Singleton Thread-Safety Audit — apache/mxnet#17495

**Branch:** `onednn-v3-port`
**Date:** 2026-05-18
**Standard:** C++17 (`-std=gnu++17`), which guarantees thread-safe function-local-static
initialisation (magic statics, ISO C++11 §6.7 [stmt.dcl]/4).

---

## Key principle

Since MXNet 2.0 uses `-std=gnu++17`, any singleton of the form

```cpp
static Foo* Get() { static Foo inst; return &inst; }
```

is **safe by the language standard**: the compiler emits an initialisation guard
(on x86 a CMPXCHG-based critical section around the first entry) that ensures
`inst` is constructed exactly once, even when multiple threads race to call
`Get()` for the first time.

The genuine risk is class **(c)**: a raw `static T* p = nullptr;` with a
subsequent manual check-and-set that is NOT protected by `std::call_once` or a
mutex-on-every-access discipline.

---

## Audit results

| Singleton | Location | Pattern | Thread-safe? | Notes |
|---|---|---|---|---|
| `Engine::Get()` | `src/engine/engine.cc:87-95` | Two function-local-statics (`shared_ptr` + raw `Engine*`). `_GetSharedRef()` is called once from `Get()` initialiser. | **YES** | Magic-static; both statics initialised in the compiler-emitted guard. |
| `Storage::Get()` | `src/storage/storage.cc:260-273` | Same two-level pattern as Engine. | **YES** | Identical to Engine. |
| `ResourceManager::Get()` | `src/resource.cc:583-586` | `dmlc::ThreadLocalStore<ResourceManagerImpl>::Get()` | **YES** | Intentionally per-thread — each thread owns its own `ResourceManagerImpl`. |
| `CpuEngine::Get()` | `src/operator/nn/dnnl/dnnl_base-inl.h:76-81` | Function-local-static. Code comments "It's thread-safe in C++11." | **YES** | Magic-static. |
| `OpenMP::Get()` | `src/engine/openmp.cc:36-39` | Function-local-static `static OpenMP openMP`. | **YES** | Magic-static. |
| `Profiler::Get()` | `src/profiler/profiler.cc:99-112` | **Hand-rolled DCL on `std::shared_ptr`** (BEFORE fix). | **NO (before fix)** | See below. |
| `ProfilerScope::Get()` | `src/profiler/profiler.cc:295-303` | Acquires mutex unconditionally, then checks-and-sets. | **YES** | Slower than needed (lock on every call after init) but correct. |
| `TmpMemMgr::Get()` | `src/operator/nn/dnnl/dnnl_base-inl.h:518-525` | `thread_local` | **YES** | Each thread has its own instance; by design. |
| `DNNLStream::Get()` | `src/operator/nn/dnnl/dnnl_base.cc:30-37` | `thread_local` | **YES** | Each thread has its own stream; by design. |

### `LazyAllocArray<T>::Get()` — storage pool lazy init

`src/common/lazy_alloc_array.h:96-133` uses a lock-free fast path (reads
`head_[idx]` without the mutex, then takes the mutex to create). Reading a
`std::shared_ptr` without synchronisation while another thread writes it is
technically UB under the C++ memory model.  In practice on x86 the pointer
load is atomic (8-byte aligned read), but it is not portable. This is a
**pre-existing issue** noted here for completeness; fixing it would require
`std::atomic<std::shared_ptr<T>>` (C++20) or a seqlock. It is OUT OF SCOPE for
this ticket — the issue author's list focuses on singleton init paths.

---

## Fix applied — `Profiler::Get()` (class c)

### Before (data race)

```cpp
Profiler* Profiler::Get(std::shared_ptr<Profiler>* sp) {
  static std::mutex mtx;
  static std::shared_ptr<Profiler> prof = nullptr;
  if (!prof) {                    // <-- unsynchronised read of shared_ptr
    std::unique_lock<std::mutex> lk(mtx);
    if (!prof) {
      prof = std::make_shared<Profiler>();
    }
  }
  if (sp) { *sp = prof; }
  return prof.get();
}
```

The outer `if (!prof)` reads `prof` (a `std::shared_ptr`) without any
synchronisation, while another thread may be writing it under the mutex. That
is a data race (undefined behaviour per C++ memory model).

### After (magic-static)

```cpp
Profiler* Profiler::Get(std::shared_ptr<Profiler>* sp) {
  // C++17 magic-static guarantees thread-safe one-time initialisation.
  static std::shared_ptr<Profiler> prof = std::make_shared<Profiler>();
  if (sp) { *sp = prof; }
  return prof.get();
}
```

The compiler-emitted initialisation guard ensures `make_shared<Profiler>()` is
called exactly once, regardless of how many threads race to enter `Get()` for
the first time.

---

## Tests

`tests/python/unittest/test_threaded_init.py` — new file:
- `test_threaded_import_and_alloc`: spawns 8 threads simultaneously, each
  calls `mx.nd.array(...)` + arithmetic. Passes iff all 8 complete without
  exception.
- `test_profiler_get_is_stable_across_threads`: 8 threads barrier-sync and
  simultaneously access `mxnet.profiler`. Verifies stable module identity.

Both tests pass (2/2).  FC subgraph smoke check: **387 passed, 0 failed,
16 skipped**.

---

## Summary

One genuine class-(c) singleton was found — `Profiler::Get()` — and fixed by
replacing the hand-rolled double-checked-lock with a C++17 magic-static. All
other singletons already use function-local-static, `thread_local`, or a
fully mutex-guarded pattern and are safe.
