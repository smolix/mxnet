/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tricky_bug_test.cc
 * \brief Repro tests for concurrency bugs that need sanitizer builds to fail
 *        reliably. In a normal build these tests exercise the same paths but
 *        may pass because the underlying bug is C++ undefined behavior.
 */
#include <gtest/gtest.h>
#include <mxnet/c_api.h>
#include <mxnet/resource.h>
#include <mxnet/runtime/container.h>
#include <mxnet/storage.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#if MXNET_USE_DIST_KVSTORE
#define private public
#include "ps/internal/env.h"
#include "ps/internal/postoffice.h"
#include "../../../src/kvstore/kvstore_dist.h"
#include "../../../src/kvstore/p3store_dist.h"
#undef private
#endif
#include "../../../src/common/lazy_alloc_array.h"
#include "../../../src/operator/custom/custom-inl.h"
#include "../../../src/operator/contrib/sync_batch_norm-inl.h"
#include "../../../src/profiler/custom_op_profiler.h"
#include "../../../src/profiler/profiler.h"
#include "../../../src/profiler/storage_profiler.h"

namespace {

struct SlowInitObject {
  explicit SlowInitObject(int value) : value(value) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  int value;
};

void AssertMXSuccess(int status) {
  ASSERT_EQ(status, 0) << MXGetLastError();
}

void SetProcessEnv(const char* key, const char* value) {
#if defined(_WIN32)
  _putenv_s(key, value);
#else
  setenv(key, value, 1);
#endif
}

void UnsetProcessEnv(const char* key) {
#if defined(_WIN32)
  _putenv_s(key, "");
#else
  unsetenv(key);
#endif
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    const char* old = std::getenv(key);
    if (old != nullptr) {
      had_old_ = true;
      old_     = old;
    }
    SetProcessEnv(key, value);
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      SetProcessEnv(key_.c_str(), old_.c_str());
    } else {
      UnsetProcessEnv(key_.c_str());
    }
  }

 private:
  std::string key_;
  std::string old_;
  bool had_old_{false};
};

struct CompletionState {
  std::mutex mutex;
  std::condition_variable cv;
  bool done{false};
  std::string error;
};

void CompleteOp(mxnet::Engine*, void* param, const dmlc::Error* error) {
  auto* state = static_cast<CompletionState*>(param);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->done = true;
    if (error != nullptr) {
      state->error = error->what();
    }
  }
  state->cv.notify_one();
}

mxnet::OpContext MakeOpContext(CompletionState* state) {
  mxnet::OpContext ctx;
  ctx.need_grad          = false;
  ctx.is_train           = true;
  ctx.run_ctx.ctx        = mxnet::Context::CPU();
  ctx.async_on_complete  = mxnet::Engine::Get()->CreateCallback(CompleteOp, state);
  return ctx;
}

void WaitForCompletion(CompletionState* state) {
  std::unique_lock<std::mutex> lock(state->mutex);
  ASSERT_TRUE(state->cv.wait_for(lock, std::chrono::seconds(10), [&] { return state->done; }));
}

AtomicSymbolCreator FindAtomicSymbolCreator(const char* op_name) {
  uint32_t size = 0;
  AtomicSymbolCreator* creators = nullptr;
  AssertMXSuccess(MXSymbolListAtomicSymbolCreators(&size, &creators));
  for (uint32_t i = 0; i < size; ++i) {
    const char* name = nullptr;
    AssertMXSuccess(MXSymbolGetAtomicSymbolName(creators[i], &name));
    if (std::strcmp(name, op_name) == 0) {
      return creators[i];
    }
  }
  return nullptr;
}

struct RuntimeRaceObj0 : public mxnet::runtime::Object {
  static constexpr const char* _type_key = "mxnet.test.RuntimeRaceObj0";
  MXNET_DECLARE_FINAL_OBJECT_INFO(RuntimeRaceObj0, mxnet::runtime::Object);
};

struct RuntimeRaceObj1 : public mxnet::runtime::Object {
  static constexpr const char* _type_key = "mxnet.test.RuntimeRaceObj1";
  MXNET_DECLARE_FINAL_OBJECT_INFO(RuntimeRaceObj1, mxnet::runtime::Object);
};

struct RuntimeRaceObj2 : public mxnet::runtime::Object {
  static constexpr const char* _type_key = "mxnet.test.RuntimeRaceObj2";
  MXNET_DECLARE_FINAL_OBJECT_INFO(RuntimeRaceObj2, mxnet::runtime::Object);
};

#if MXNET_USE_DIST_KVSTORE
class ScopedPSEnvironment {
 public:
  explicit ScopedPSEnvironment(const std::unordered_map<std::string, std::string>& env)
      : old_(ps::Environment::Get()->kvs) {
    ps::Environment::Init(env);
  }

  ~ScopedPSEnvironment() {
    ps::Environment::Init(old_);
  }

 private:
  std::unordered_map<std::string, std::string> old_;
};

class ScopedServerKeyRanges {
 public:
  explicit ScopedServerKeyRanges(const std::vector<ps::Range>& ranges) {
    auto* postoffice = ps::Postoffice::Get();
    std::lock_guard<std::mutex> lock(postoffice->server_key_ranges_mu_);
    old_                              = postoffice->server_key_ranges_;
    postoffice->server_key_ranges_    = ranges;
  }

  ~ScopedServerKeyRanges() {
    auto* postoffice = ps::Postoffice::Get();
    std::lock_guard<std::mutex> lock(postoffice->server_key_ranges_mu_);
    postoffice->server_key_ranges_ = old_;
  }

 private:
  std::vector<ps::Range> old_;
};

class ScopedKVStoreDistTestEnv {
 public:
  ScopedKVStoreDistTestEnv()
      : env_({{"DMLC_ROLE", "server"}, {"DMLC_NUM_WORKER", "0"}, {"DMLC_NUM_SERVER", "1"}}),
        ranges_({ps::Range(0, ps::kMaxKey)}) {}

 private:
  ScopedPSEnvironment env_;
  ScopedServerKeyRanges ranges_;
};

class ExposedP3StoreDist : public mxnet::kvstore::P3StoreDist {
 public:
  using P3StoreDist::P3StoreDist;
  using P3StoreDist::EncodeDefaultKey;

  void MutateCachedEncoding(const int key) {
    auto& pskv = ps_kv_[key];
    ASSERT_FALSE(pskv.keys.empty());
    ASSERT_FALSE(pskv.lens.empty());
    pskv.keys[0] += 1;
    pskv.lens[0] += 1;
  }
};
#endif

}  // namespace

TEST(TrickyBug, MXNDArrayLoadNpyClearsNameOutputs) {
  NDArrayHandle source = nullptr;
  const uint32_t shape[] = {2};
  const std::vector<float> values = {1.0f, 2.0f};
  AssertMXSuccess(MXNDArrayCreate(shape, 1, 1, 0, 0, 0, &source));
  AssertMXSuccess(MXNDArraySyncCopyFromCPU(source, values.data(), values.size()));

#if defined(_WIN32)
  const std::string path = "mxnet_tricky_bug_load.npy";
#else
  const std::string path = "/tmp/mxnet_tricky_bug_load_" + std::to_string(getpid()) + ".npy";
#endif
  AssertMXSuccess(MXNDArraySave(path.c_str(), 1, &source, nullptr));
  AssertMXSuccess(MXNDArrayFree(source));

  uint32_t out_size = 0;
  NDArrayHandle* out_arr = nullptr;
  uint32_t out_name_size = 0xdeadbeefu;
  const char** out_names = reinterpret_cast<const char**>(static_cast<uintptr_t>(1));
  AssertMXSuccess(MXNDArrayLoad(path.c_str(), &out_size, &out_arr, &out_name_size, &out_names));

  EXPECT_EQ(out_size, 1U);
  EXPECT_EQ(out_name_size, 0U) << ".npy loads have no names and must clear the out name count.";
  EXPECT_EQ(out_names, nullptr) << ".npy loads have no names and must clear the out names pointer.";

  for (uint32_t i = 0; i < out_size; ++i) {
    AssertMXSuccess(MXNDArrayFree(out_arr[i]));
  }
  std::remove(path.c_str());
}

TEST(TrickyBug, LazyAllocArrayConcurrentFirstGetIsTsanClean) {
  mxnet::common::LazyAllocArray<SlowInitObject> array;
  constexpr int kThreads = 32;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&array, &start]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iter = 0; iter < 200; ++iter) {
        std::shared_ptr<SlowInitObject> ptr = array.Get(0, []() { return new SlowInitObject(7); });
        ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(ptr->value, 7);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(TrickyBug, RuntimeTypeKeyLookupDuringLazyRegistrationIsTsanClean) {
  const uint32_t adt_type_index = mxnet::runtime::ADTObj::RuntimeTypeIndex();
  ASSERT_EQ(mxnet::runtime::Object::TypeKey2Index(mxnet::runtime::ADTObj::_type_key),
            adt_type_index);

  constexpr int kIterations = 20000;
  std::atomic<bool> start{false};
  std::thread lookup_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kIterations; ++i) {
      ASSERT_EQ(mxnet::runtime::Object::TypeKey2Index(mxnet::runtime::ADTObj::_type_key),
                adt_type_index);
    }
  });

  std::thread registration_thread([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kIterations; ++i) {
      switch (i % 3) {
        case 0:
          ASSERT_NE(RuntimeRaceObj0::RuntimeTypeIndex(), mxnet::runtime::TypeIndex::kRoot);
          break;
        case 1:
          ASSERT_NE(RuntimeRaceObj1::RuntimeTypeIndex(), mxnet::runtime::TypeIndex::kRoot);
          break;
        default:
          ASSERT_NE(RuntimeRaceObj2::RuntimeTypeIndex(), mxnet::runtime::TypeIndex::kRoot);
          break;
      }
    }
  });

  start.store(true, std::memory_order_release);
  lookup_thread.join();
  registration_thread.join();
}

TEST(TrickyBug, CustomOpProfilerSingletonFirstUseIsTsanClean) {
  constexpr int kThreads = 32;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto* profiler = mxnet::profiler::CustomOpProfiler::Get();
      ASSERT_NE(profiler, nullptr);
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(TrickyBug, CustomOpProfilerScopeIsClearedWhenCustomOpThrows) {
  ScopedEnvVar engine_type("MXNET_ENGINE_TYPE", "ThreadedEnginePerDevice");
  auto* custom_operator = mxnet::op::custom::CustomOperator::Get();
  custom_operator->Stop();
  custom_operator->Start();
  struct CustomOperatorGuard {
    explicit CustomOperatorGuard(mxnet::op::custom::CustomOperator* op) : op(op) {}
    ~CustomOperatorGuard() {
      op->Stop();
      op->Start();
    }
    mxnet::op::custom::CustomOperator* op;
  } custom_operator_guard(custom_operator);

  auto* profiler = mxnet::profiler::Profiler::Get();
  profiler->SetConfig(mxnet::profiler::Profiler::kImperative,
                      "/tmp/mxnet_custom_profiler_scope_test.json",
                      false,
                      0.0f,
                      false);
  profiler->SetState(mxnet::profiler::Profiler::kRunning);

  CompletionState first;
  auto first_ctx = MakeOpContext(&first);
  custom_operator->Push(
      []() { throw dmlc::Error("intentional custom-op failure"); },
      first_ctx,
      false,
      true,
      {},
      {},
      {},
      {},
      "throwing_custom");
  WaitForCompletion(&first);
  ASSERT_NE(first.error.find("intentional custom-op failure"), std::string::npos);

  profiler->SetState(mxnet::profiler::Profiler::kNotRunning);

  CompletionState second;
  auto second_ctx = MakeOpContext(&second);
  std::string display_name;
  custom_operator->Push(
      [&display_name]() {
        display_name = mxnet::profiler::CustomOpProfiler::Get()->GenerateDisplayName("child_op");
      },
      second_ctx,
      false,
      true,
      {},
      {},
      {},
      {},
      "plain_custom");
  WaitForCompletion(&second);
  ASSERT_TRUE(second.error.empty()) << second.error;
  EXPECT_EQ(display_name, "child_op")
      << "A throwing profiled custom op must not leave its op type attached to the worker thread.";
}

TEST(TrickyBug, ProfilerScopeConcurrentSetGetIsTsanClean) {
  mxnet::profiler::ProfilerScope* scope = mxnet::profiler::ProfilerScope::Get();
  constexpr int kReaders                = 8;
  constexpr int kIterations             = 20000;
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::vector<std::thread> threads;
  threads.reserve(kReaders + 1);

  threads.emplace_back([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kIterations; ++i) {
      // Long strings force heap allocation and make unsynchronized string
      // mutation visible to TSAN and, occasionally, to ASAN/libstdc++ debug
      // builds as use-after-free or torn reads.
      scope->SetCurrentProfilerScope(std::string(256, static_cast<char>('a' + (i % 26))));
    }
    done.store(true, std::memory_order_release);
  });

  for (int reader = 0; reader < kReaders; ++reader) {
    threads.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!done.load(std::memory_order_acquire)) {
        std::string current = scope->GetCurrentProfilerScope();
        if (!current.empty()) {
          ASSERT_GE(current.size(), 1U);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(TrickyBug, DeviceStorageProfilerConcurrentFirstUseIsTsanClean) {
  const char* keys[] = {"profile_memory", "filename"};
  const char* vals[] = {"1", "/tmp/mxnet_storage_profiler_tsan.json"};
  AssertMXSuccess(MXSetProfilerConfig(2, keys, vals));
  AssertMXSuccess(MXSetProfilerState(1));

  mxnet::profiler::DeviceStorageProfiler profiler("TSAN Device Storage");
  mxnet::Storage::Handle handle;
  handle.ctx  = mxnet::Context::CPU(0);
  handle.size = 64;
  handle.dptr = reinterpret_cast<void*>(static_cast<uintptr_t>(1));

  constexpr int kThreads    = 32;
  constexpr int kIterations = 1000;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        profiler.OnAlloc(handle);
        profiler.OnFree(handle);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  AssertMXSuccess(MXSetProfilerState(0));
}

#if MXNET_USE_DIST_KVSTORE
TEST(TrickyBug, KVStoreDistDefaultKeyEncodingRejectsPsLiteLensOverflow) {
  ScopedKVStoreDistTestEnv env;
  mxnet::kvstore::KVStoreDist store(false);
  store.bigarray_bound_ = std::numeric_limits<size_t>::max();

  const size_t too_many_float_elems =
      static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float) + 1;
  EXPECT_THROW(store.EncodeDefaultKey(7, too_many_float_elems, sizeof(float)), dmlc::Error);
}

TEST(TrickyBug, P3StoreDistDefaultKeyEncodingRejectsPsLiteLensOverflow) {
  ScopedKVStoreDistTestEnv env;
  mxnet::kvstore::P3StoreDist store(false);
  store.slice_threshold_ = std::numeric_limits<size_t>::max() / sizeof(float);

  const size_t too_many_float_elems =
      static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float) + 1;
  EXPECT_THROW(store.EncodeDefaultKey(11, too_many_float_elems, sizeof(float)), dmlc::Error);
}

TEST(TrickyBug, P3StoreDistDefaultKeyEncodingReturnsSnapshot) {
  ScopedKVStoreDistTestEnv env;
  ExposedP3StoreDist store(false);
  store.slice_threshold_ = std::numeric_limits<size_t>::max() / sizeof(float);

  auto first            = store.EncodeDefaultKey(13, 16, sizeof(float));
  const auto first_key  = first.keys[0];
  const auto first_lens = first.lens[0];

  store.MutateCachedEncoding(13);

  EXPECT_EQ(first.keys[0], first_key);
  EXPECT_EQ(first.lens[0], first_lens);
}

TEST(TrickyBug, KVStoreDistRowSparseEncodingReturnsSnapshot) {
  ScopedKVStoreDistTestEnv env;
  mxnet::kvstore::KVStoreDist store(false);
  store.bigarray_bound_ = std::numeric_limits<size_t>::max();

  const int64_t first_offsets[] = {0, 2};
  const auto& first             = store.EncodeRowSparseKey(
      17, 6, 2, first_offsets, 3, 4, sizeof(float));
  ASSERT_EQ(first.keys.size(), 3U);
  ASSERT_EQ(first.lens.size(), 3U);

  const int64_t second_offsets[] = {1};
  const auto& second             = store.EncodeRowSparseKey(
      17, 3, 1, second_offsets, 3, 4, sizeof(float));
  ASSERT_EQ(second.keys.size(), 2U);
  ASSERT_EQ(second.lens.size(), 2U);

  EXPECT_EQ(first.keys.size(), 3U)
      << "Row-sparse encoding must return a snapshot; a cached reference is mutated by later calls.";
  EXPECT_EQ(first.lens.size(), 3U);
  EXPECT_EQ(first.size, 6U * sizeof(float));
}

TEST(TrickyBug, KVStoreDistDefaultKeyEncodingConcurrentAccessIsTsanClean) {
  ScopedKVStoreDistTestEnv env;
  mxnet::kvstore::KVStoreDist store(false);
  store.bigarray_bound_ = std::numeric_limits<size_t>::max();

  constexpr int kThreads    = 16;
  constexpr int kIterations = 1000;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&store, &start, t]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        auto pskv = store.EncodeDefaultKey(100 + ((t + i) % 4), 16, sizeof(float));
        ASSERT_EQ(pskv.keys.size(), 1U);
        ASSERT_EQ(pskv.lens.size(), 1U);
        ASSERT_EQ(pskv.size, 16U * sizeof(float));
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
}
#endif

TEST(TrickyBug, AllFiniteCpuParallelWritesAreTsanClean) {
  AtomicSymbolCreator creator = FindAtomicSymbolCreator("all_finite");
  ASSERT_NE(creator, nullptr);

  NDArrayHandle input = nullptr;
  const uint32_t shape[] = {1U << 20};
  AssertMXSuccess(MXNDArrayCreate(shape, 1, 1, 0, 0, 0, &input));
  std::vector<float> values(shape[0], std::numeric_limits<float>::infinity());
  AssertMXSuccess(MXNDArraySyncCopyFromCPU(input, values.data(), values.size()));

  const char* keys[] = {"init_output"};
  const char* vals[] = {"True"};
  for (int i = 0; i < 100; ++i) {
    int num_outputs = 0;
    NDArrayHandle* outputs = nullptr;
    AssertMXSuccess(MXImperativeInvoke(creator, 1, &input, &num_outputs, &outputs, 1, keys, vals,
                                       nullptr));
    ASSERT_EQ(num_outputs, 1);
    AssertMXSuccess(MXNDArrayWaitToRead(outputs[0]));
    float result = 1.0f;
    AssertMXSuccess(MXNDArraySyncCopyToCPU(outputs[0], &result, 1));
    EXPECT_EQ(result, 0.0f);
    AssertMXSuccess(MXNDArrayFree(outputs[0]));
  }
  AssertMXSuccess(MXNDArrayFree(input));
}

#if !defined(_WIN32)
TEST(TrickyBug, SyncBatchNormBarrierIsReusableAcrossGenerations) {
  EXPECT_EXIT(
      {
        std::signal(SIGALRM, [](int) { _exit(4); });
        alarm(5);
        mxnet::op::Barrier barrier(2);
        std::atomic<bool> start{false};
        constexpr int kIterations = 200000;
        std::thread slow_waiter([&]() {
          while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
          for (int i = 0; i < kIterations; ++i) {
            barrier.Wait();
          }
        });
        std::thread fast_waiter([&]() {
          while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
          for (int i = 0; i < kIterations; ++i) {
            barrier.Wait();
          }
        });
        start.store(true, std::memory_order_release);
        slow_waiter.join();
        fast_waiter.join();
        _exit(0);
      },
      ::testing::ExitedWithCode(0),
      "")
      << "The reusable barrier needs a generation counter; otherwise a waiter can miss "
         "the previous generation's notify and deadlock.";
}
#endif

#if MXNET_USE_CUDA
TEST(TrickyBug, GpuDeviceStorageProfilerConcurrentMapAccessIsTsanClean) {
  const char* keys[] = {"profile_memory", "gpu_memory_profile_filename_prefix"};
  const char* vals[] = {"1", "/tmp/mxnet_gpu_storage_profiler_tsan"};
  AssertMXSuccess(MXSetProfilerConfig(2, keys, vals));
  AssertMXSuccess(MXSetProfilerState(1));

  auto* profiler = mxnet::profiler::GpuDeviceStorageProfiler::Get();
  constexpr int kThreads    = 16;
  constexpr int kIterations = 2000;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads + 1);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        mxnet::Storage::Handle handle;
        handle.ctx            = mxnet::Context::GPU(0);
        handle.size           = 64;
        handle.dptr           = reinterpret_cast<void*>(
            (static_cast<uintptr_t>(t + 1) << 32) | static_cast<uintptr_t>(i + 1));
        handle.profiler_scope = "scope";
        handle.name           = "name";
        profiler->OnAlloc(handle, handle.size, false);
        profiler->UpdateStorageInfo(handle);
        profiler->OnFree(handle);
      }
    });
  }

  threads.emplace_back([&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 100; ++i) {
      profiler->DumpProfile();
    }
  });

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  AssertMXSuccess(MXSetProfilerState(0));
}

TEST(TrickyBug, StreamHandleApiUsesPointerSizedStorage) {
  EXPECT_GE(sizeof(uintptr_t), sizeof(cudaStream_t))
      << "CUDA stream handles must fit in the pointer-sized C API overloads.";
  static_assert(std::is_same<decltype(&MXPushStreamDepEx),
                             int (*)(NDArrayHandle, uintptr_t)>::value,
                "MXPushStreamDepEx must expose a pointer-sized stream argument.");
  static_assert(std::is_same<decltype(&MXGetCurrentStreamEx),
                             int (*)(int, uintptr_t*)>::value,
                "MXGetCurrentStreamEx must expose a pointer-sized stream output.");
}
#endif
