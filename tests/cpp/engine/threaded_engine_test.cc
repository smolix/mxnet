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
 * \file threaded_engine_test.cc
 * \brief threaded engine tests
 */
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <dmlc/omp.h>
#include <gtest/gtest.h>
#include <mxnet/c_api.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <dmlc/timer.h>
#include <atomic>
#include <csignal>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "../src/engine/engine_impl.h"
#include "../include/test_util.h"

/**
 * present the following workload
 *  n = reads.size()
 *  data[write] = (data[reads[0]] + ... data[reads[n]]) / n
 *  std::this_thread::sleep_for(std::chrono::microseconds(time));
 */
struct Workload {
  std::vector<int> reads;
  int write;
  int time;
};

/**
 * generate a list of workloads
 */
void GenerateWorkload(int num_workloads,
                      int num_var,
                      int min_read,
                      int max_read,
                      int min_time,
                      int max_time,
                      std::vector<Workload>* workloads) {
  workloads->clear();
  workloads->resize(num_workloads);
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<int> distribution_var(0, num_var - 1);
  std::uniform_int_distribution<int> distribution_time(min_time, max_time - 1);
  std::uniform_int_distribution<int> distribution_read(min_read, max_read - 1);
  for (int i = 0; i < num_workloads; ++i) {
    auto& wl     = workloads->at(i);
    wl.write     = distribution_var(generator);
    int num_read = distribution_read(generator);
    for (int j = 0; j < num_read; ++j) {
      wl.reads.push_back(distribution_var(generator));
    }
    wl.time = distribution_time(generator);
  }
}

/**
 * evaluate a single workload
 */
void EvaluateWorkload(const Workload& wl, std::vector<double>* data) {
  double tmp = 0;
  for (int i : wl.reads)
    tmp += data->at(i);
  data->at(wl.write) = tmp / (wl.reads.size() + 1);
  if (wl.time > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(wl.time));
  }
}

void AsyncCompleteWithError(mxnet::RunContext,
                            mxnet::Engine::CallbackOnStart on_start,
                            mxnet::Engine::CallbackOnComplete on_complete) {
  on_start();
  dmlc::Error error("expected async callback failure");
  on_complete(&error);
}

mxnet::Engine::AsyncFn AsyncCompleteWithNamedError(const std::string& message) {
  return [message](mxnet::RunContext,
                   mxnet::Engine::CallbackOnStart on_start,
                   mxnet::Engine::CallbackOnComplete on_complete) {
    on_start();
    dmlc::Error error(message);
    on_complete(&error);
  };
}

void NoOpSync(mxnet::RunContext) {}

/**
 * evaluate a list of workload, return the time used
 */
double EvaluateWorkloads(const std::vector<Workload>& workloads,
                         mxnet::Engine* engine,
                         std::vector<double>* data) {
  using namespace mxnet;
  double t = dmlc::GetTime();
  std::vector<Engine::VarHandle> vars;
  if (engine) {
    for (size_t i = 0; i < data->size(); ++i) {
      vars.push_back(engine->NewVariable());
    }
  }

  for (const auto& wl : workloads) {
    if (wl.reads.size() == 0)
      continue;
    if (engine == nullptr) {
      EvaluateWorkload(wl, data);
    } else {
      auto func = [wl, data](RunContext ctx,
                             Engine::CallbackOnStart on_start,
                             Engine::CallbackOnComplete cb) {
        on_start();
        EvaluateWorkload(wl, data);
        cb();
      };
      std::vector<Engine::VarHandle> reads;
      for (auto i : wl.reads) {
        if (i != wl.write)
          reads.push_back(vars[i]);
      }
      engine->PushAsync(func, Context::CPU(), reads, {vars[wl.write]});
    }
  }

  if (engine) {
    engine->WaitForAll();
  }
  return dmlc::GetTime() - t;
}

TEST(Engine, start_stop) {
  const int num_engine = 3;
  std::vector<mxnet::Engine*> engine(num_engine);
  engine[0]                 = mxnet::engine::CreateNaiveEngine();
  engine[1]                 = mxnet::engine::CreateThreadedEnginePooled();
  engine[2]                 = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[3] = {"NaiveEngine", "ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  for (int i = 0; i < num_engine; ++i) {
    LOG(INFO) << "Stopping: " << type_names[i];
    engine[i]->Stop();
    LOG(INFO) << "Stopped: " << type_names[i] << " Starting...";
    engine[i]->Start();
    LOG(INFO) << "Started: " << type_names[i] << " Done...";
  }
}

TEST(Engine, ThreadedAsyncExceptionsAreReportedOnce) {
  const int num_engine = 2;
  std::vector<mxnet::Engine*> engines(num_engine);
  engines[0]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[1]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  constexpr int kThreads       = 4;
  constexpr int kOpsPerThread  = 32;
  constexpr int kNumOperations = kThreads * kOpsPerThread;

  for (int e = 0; e < num_engine; ++e) {
    auto engine = engines[e];
    LOG(INFO) << "Testing async exception clearing in " << type_names[e];

    std::vector<mxnet::Engine::VarHandle> vars(kNumOperations);
    for (auto& var : vars) {
      var = engine->NewVariable();
    }

    std::atomic<bool> start(false);
    std::vector<std::thread> pushers;
    pushers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      pushers.emplace_back([engine, &vars, &start, t]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        const int offset = t * kOpsPerThread;
        for (int i = 0; i < kOpsPerThread; ++i) {
          const int op_id = offset + i;
          engine->PushAsync(
              [op_id](mxnet::RunContext,
                      mxnet::Engine::CallbackOnStart on_start,
                      mxnet::Engine::CallbackOnComplete on_complete) {
                on_start();
                dmlc::Error error("expected async engine failure " + std::to_string(op_id));
                on_complete(&error);
              },
              mxnet::Context::CPU(),
              {},
              {vars[op_id]},
              mxnet::FnProperty::kAsync,
              0,
              "ThreadedAsyncExceptionRegression");
        }
      });
    }

    start.store(true, std::memory_order_release);
    for (auto& pusher : pushers) {
      pusher.join();
    }

    EXPECT_THROW(engine->WaitForAll(), dmlc::Error);
    EXPECT_NO_THROW(engine->WaitForAll());

    for (auto var : vars) {
      engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
    }
    EXPECT_NO_THROW(engine->WaitForAll());
  }
}

TEST(Engine, ThreadedReadOnlyAsyncExceptionReachesWaitForAll) {
  const int num_engine = 2;
  std::vector<mxnet::Engine*> engines(num_engine);
  engines[0]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[1]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  for (int e = 0; e < num_engine; ++e) {
    auto engine = engines[e];
    LOG(INFO) << "Testing read-only async exception propagation in " << type_names[e];

    auto var = engine->NewVariable();
    engine->PushAsync(AsyncCompleteWithError,
                      mxnet::Context::CPU(),
                      {var},
                      {},
                      mxnet::FnProperty::kAsync,
                      0,
                      "ReadOnlyAsyncExceptionRegression");

    EXPECT_THROW(engine->WaitForAll(), dmlc::Error)
        << "Callback errors from read-only ops must be visible to WaitForAll.";
    engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
    EXPECT_NO_THROW(engine->WaitForAll());
  }
}

TEST(Engine, ThreadedNoVarAsyncExceptionReachesWaitForAll) {
  const int num_engine = 2;
  std::vector<mxnet::Engine*> engines(num_engine);
  engines[0]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[1]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  for (int e = 0; e < num_engine; ++e) {
    auto engine = engines[e];
    LOG(INFO) << "Testing no-var async exception propagation in " << type_names[e];

    engine->PushAsync(AsyncCompleteWithError,
                      mxnet::Context::CPU(),
                      {},
                      {},
                      mxnet::FnProperty::kAsync,
                      0,
                      "NoVarAsyncExceptionRegression");

    EXPECT_THROW(engine->WaitForAll(), dmlc::Error)
        << "Callback errors from ops without dependency vars must be visible to WaitForAll.";
    EXPECT_NO_THROW(engine->WaitForAll());
  }
}

TEST(Engine, WaitForAllAggregatesMultipleAsyncExceptions) {
  const int num_engine = 3;
  std::vector<mxnet::Engine*> engines(num_engine);
  engines[0]                = mxnet::engine::CreateNaiveEngine();
  engines[1]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[2]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[3] = {"NaiveEngine", "ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  for (int e = 0; e < num_engine; ++e) {
    auto engine = engines[e];
    LOG(INFO) << "Testing WaitForAll exception aggregation in " << type_names[e];

    engine->PushAsync(AsyncCompleteWithNamedError("first aggregated engine failure"),
                      mxnet::Context::CPU(),
                      {},
                      {},
                      mxnet::FnProperty::kAsync,
                      0,
                      "FirstAggregatedFailure");
    engine->PushAsync(AsyncCompleteWithNamedError("second aggregated engine failure"),
                      mxnet::Context::CPU(),
                      {},
                      {},
                      mxnet::FnProperty::kAsync,
                      0,
                      "SecondAggregatedFailure");

    try {
      engine->WaitForAll();
      FAIL() << "WaitForAll should report the aggregated async errors";
    } catch (const dmlc::Error& err) {
      const std::string message = err.what();
      EXPECT_NE(message.find("Multiple asynchronous engine errors"), std::string::npos)
          << message;
      EXPECT_NE(message.find("first aggregated engine failure"), std::string::npos) << message;
      EXPECT_NE(message.find("second aggregated engine failure"), std::string::npos) << message;
    }
    EXPECT_NO_THROW(engine->WaitForAll());
  }
}

TEST(Engine, NaiveAsyncCallbackExceptionReachesWaitForAll) {
  mxnet::Engine* engine = mxnet::engine::CreateNaiveEngine();
  auto var              = engine->NewVariable();

  engine->PushAsync(AsyncCompleteWithError,
                    mxnet::Context::CPU(),
                    {},
                    {var},
                    mxnet::FnProperty::kAsync,
                    0,
                    "NaiveAsyncExceptionRegression");

  EXPECT_THROW(engine->WaitForAll(), dmlc::Error)
      << "NaiveEngine must not drop errors passed to CallbackOnComplete.";
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
}

TEST(Engine, NegativeBulkSizeIsRejectedAndDoesNotPoisonThreadLocalState) {
  mxnet::Engine* engine = mxnet::engine::CreateThreadedEnginePooled();

  bool rejected_negative_bulk_size = false;
  bool follow_up_push_succeeded    = false;
  std::string follow_up_error;

  // Bulk status is thread-local. Run the invalid state transition in a throwaway
  // thread so the test process remains usable even when the current bug poisons
  // that thread's bulk state.
  std::thread worker([&]() {
    try {
      engine->set_bulk_size(-1);
    } catch (const dmlc::Error&) {
      rejected_negative_bulk_size = true;
    } catch (const std::exception& e) {
      follow_up_error = std::string("unexpected set_bulk_size exception: ") + e.what();
    }

    try {
      engine->PushSync(NoOpSync, mxnet::Context::CPU(), {}, {});
      follow_up_push_succeeded = true;
    } catch (const std::exception& e) {
      follow_up_error = e.what();
    }
  });
  worker.join();

  EXPECT_TRUE(rejected_negative_bulk_size)
      << "set_bulk_size(-1) should fail before mutating thread-local bulk state.";
  EXPECT_TRUE(follow_up_push_succeeded)
      << "A rejected bulk-size update must not make the next PushSync throw: " << follow_up_error;
  EXPECT_NO_THROW(engine->WaitForAll());
}

#if !defined(_WIN32)
TEST(Engine, ThreadedEnginePerDeviceStartIsIdempotent) {
  EXPECT_EXIT(
      {
        std::signal(SIGALRM, [](int) { _exit(2); });
        alarm(3);
        mxnet::Engine* engine = mxnet::engine::CreateThreadedEnginePerDevice();
        engine->Start();
        _exit(0);
      },
      ::testing::ExitedWithCode(0),
      "")
      << "Calling Start() on an already-started per-device engine must not deadlock.";
}
#endif

TEST(Engine, WaitForVarClearsThreadedAsyncException) {
  const int num_engine = 2;
  std::vector<mxnet::Engine*> engines(num_engine);
  engines[0]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[1]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  for (int e = 0; e < num_engine; ++e) {
    auto engine = engines[e];
    LOG(INFO) << "Testing WaitForVar exception clearing in " << type_names[e];

    auto var = engine->NewVariable();
    engine->PushAsync(
        [](mxnet::RunContext,
           mxnet::Engine::CallbackOnStart on_start,
           mxnet::Engine::CallbackOnComplete on_complete) {
          on_start();
          dmlc::Error error("expected WaitForVar async failure");
          on_complete(&error);
        },
        mxnet::Context::CPU(),
        {},
        {var},
        mxnet::FnProperty::kAsync,
        0,
        "WaitForVarExceptionRegression");

    EXPECT_THROW(engine->WaitForVar(var), dmlc::Error);
    EXPECT_NO_THROW(engine->WaitForVar(var));
    EXPECT_NO_THROW(engine->WaitForAll());

    engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
    EXPECT_NO_THROW(engine->WaitForAll());
  }
}

TEST(Engine, RandSumExpr) {
  std::vector<Workload> workloads;
  int num_repeat       = 5;
  const int num_engine = 4;

  std::vector<double> t(num_engine, 0.0);
  std::vector<mxnet::Engine*> engine(num_engine);

  engine[0] = nullptr;
  engine[1] = mxnet::engine::CreateNaiveEngine();
  engine[2] = mxnet::engine::CreateThreadedEnginePooled();
  engine[3] = mxnet::engine::CreateThreadedEnginePerDevice();

  for (int repeat = 0; repeat < num_repeat; ++repeat) {
    srand(time(nullptr) + repeat);
    int num_var = 100;
    GenerateWorkload(10000, num_var, 2, 20, 1, 10, &workloads);
    std::vector<std::vector<double>> data(num_engine);
    for (int i = 0; i < num_engine; ++i) {
      data[i].resize(num_var, 1.0);
      t[i] += EvaluateWorkloads(workloads, engine[i], &data[i]);
    }

    for (int i = 1; i < num_engine; ++i) {
      for (int j = 0; j < num_var; ++j)
        EXPECT_EQ(data[0][j], data[i][j]);
    }
    LOG(INFO) << "data: " << data[0][1] << " " << data[0][2] << "...";
  }

  LOG(INFO) << "baseline\t\t" << t[0] << " sec";
  LOG(INFO) << "NaiveEngine\t\t" << t[1] << " sec";
  LOG(INFO) << "ThreadedEnginePooled\t" << t[2] << " sec";
  LOG(INFO) << "ThreadedEnginePerDevice\t" << t[3] << " sec";
}

void Foo(mxnet::RunContext, int i) {
  printf("The fox says %d\n", i);
}

void FooAsyncFunc(void*, void*, void* cb_ptr, void* param) {
  if (param == nullptr) {
    LOG(INFO) << "The fox asynchronously says receiving nothing.";
  } else {
    auto num = static_cast<int*>(param);
    EXPECT_EQ(*num, 100);
    LOG(INFO) << "The fox asynchronously says receiving " << *num;
  }
  auto cb = *static_cast<mxnet::engine::CallbackOnComplete*>(cb_ptr);
  cb();
}

void FooSyncFunc(void*, void* param) {
  if (param == nullptr) {
    LOG(INFO) << "The fox synchronously says receiving nothing.";
  } else {
    auto num = static_cast<int*>(param);
    EXPECT_EQ(*num, 101);
    LOG(INFO) << "The fox synchronously says receiving " << *num;
  }
}

void FooFuncDeleter(void* param) {
  if (param != nullptr) {
    auto num = static_cast<int*>(param);
    LOG(INFO) << "The fox says deleting " << *num;
    delete num;
  }
}

TEST(Engine, PushFunc) {
  auto var = mxnet::Engine::Get()->NewVariable();
  auto ctx = mxnet::Context{};

  // Test #1
  LOG(INFO) << "===== Test #1: PushAsync param and deleter =====";
  int* a  = new int(100);
  int res = MXEnginePushAsync(FooAsyncFunc, a, FooFuncDeleter, &ctx, &var, 1, nullptr, 0);
  EXPECT_EQ(res, 0);

  // Test #2
  LOG(INFO) << "===== Test #2: PushAsync NULL param and NULL deleter =====";
  res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, nullptr, 0, &var, 0);
  EXPECT_EQ(res, 0);

  // Test #3
  LOG(INFO) << "===== Test #3: PushAsync invalid number of const vars =====";
  res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, &var, -1, nullptr, 0);
  EXPECT_EQ(res, -1);

  // Test #4
  LOG(INFO) << "===== Test #4: PushAsync invalid number of mutable vars =====";
  res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, nullptr, 0, &var, -1);
  EXPECT_EQ(res, -1);

  // Test #5
  LOG(INFO) << "===== Test #5: PushSync param and deleter =====";
  int* b = new int(101);
  res    = MXEnginePushSync(FooSyncFunc, b, FooFuncDeleter, &ctx, &var, 1, nullptr, 0);
  EXPECT_EQ(res, 0);

  // Test #6
  LOG(INFO) << "===== Test #6: PushSync NULL param and NULL deleter =====";
  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, nullptr, 0, &var, 1);
  EXPECT_EQ(res, 0);

  // Test #7
  LOG(INFO) << "===== Test #7: PushSync invalid number of const vars =====";
  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, &var, -1, nullptr, 0);
  EXPECT_EQ(res, -1);

  // Test #8
  LOG(INFO) << "===== Test #8: PushSync invalid number of mutable vars =====";
  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, nullptr, 0, &var, -1);
  EXPECT_EQ(res, -1);

  // Test #9
  LOG(INFO) << "===== Test #9: PushAsync positive const count with null array =====";
  res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, nullptr, 1, &var, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("const_vars_handle"), std::string::npos);

  // Test #10
  LOG(INFO) << "===== Test #10: PushAsync positive mutable count with null array =====";
  res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, &var, 1, nullptr, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("mutable_vars_handle"), std::string::npos);

  // Test #11
  LOG(INFO) << "===== Test #11: PushSync positive const count with null array =====";
  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, nullptr, 1, &var, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("const_vars_handle"), std::string::npos);

  // Test #12
  LOG(INFO) << "===== Test #12: PushSync positive mutable count with null array =====";
  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, &var, 1, nullptr, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("mutable_vars_handle"), std::string::npos);
}

TEST(Engine, PushFuncDeduplicatesReadWriteVars) {
  auto engine = mxnet::Engine::Get();
  auto var    = engine->NewVariable();
  auto ctx    = mxnet::Context{};

  int res = MXEnginePushAsync(FooAsyncFunc, nullptr, nullptr, &ctx, &var, 1, &var, 1);
  EXPECT_EQ(res, 0);
  engine->WaitForAll();

  res = MXEnginePushSync(FooSyncFunc, nullptr, nullptr, &ctx, &var, 1, &var, 1);
  EXPECT_EQ(res, 0);
  engine->WaitForAll();

  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
}

TEST(Engine, PushFuncND) {
  auto ctx = mxnet::Context{};
  std::vector<mxnet::NDArray*> nds;
  const int num_nds = 5;
  for (int i = 0; i < num_nds; ++i) {
    mxnet::NDArray* pnd = new mxnet::NDArray(ctx);
    nds.push_back(pnd);
  }
  for (int num_const_nds = 0; num_const_nds <= num_nds; ++num_const_nds) {
    int num_mutable_nds     = num_nds - num_const_nds;
    void** const_nds_handle = num_const_nds > 0 ? reinterpret_cast<void**>(nds.data()) : nullptr;
    void** mutable_nds_handle =
        num_mutable_nds > 0 ? reinterpret_cast<void**>(nds.data() + num_const_nds) : nullptr;

    // Test #1
    LOG(INFO) << "===== Test #1: PushAsyncND param and deleter =====";
    int* a  = new int(100);
    int res = MXEnginePushAsyncND(FooAsyncFunc,
                                  a,
                                  FooFuncDeleter,
                                  &ctx,
                                  const_nds_handle,
                                  num_const_nds,
                                  mutable_nds_handle,
                                  num_mutable_nds);
    EXPECT_EQ(res, 0);

    // Test #2
    LOG(INFO) << "===== Test #2: PushAsyncND NULL param and NULL deleter =====";
    res = MXEnginePushAsyncND(FooAsyncFunc,
                              nullptr,
                              nullptr,
                              &ctx,
                              const_nds_handle,
                              num_const_nds,
                              mutable_nds_handle,
                              num_mutable_nds);
    EXPECT_EQ(res, 0);

    // Test #3
    LOG(INFO) << "===== Test #3: PushAsyncND invalid number of const nds =====";
    res = MXEnginePushAsyncND(FooAsyncFunc,
                              nullptr,
                              nullptr,
                              &ctx,
                              const_nds_handle,
                              -1,
                              mutable_nds_handle,
                              num_mutable_nds);
    EXPECT_EQ(res, -1);

    // Test #4
    LOG(INFO) << "===== Test #4: PushAsyncND invalid number of mutable nds =====";
    res = MXEnginePushAsyncND(FooAsyncFunc,
                              nullptr,
                              nullptr,
                              &ctx,
                              const_nds_handle,
                              num_const_nds,
                              mutable_nds_handle,
                              -1);
    EXPECT_EQ(res, -1);

    // Test #5
    LOG(INFO) << "===== Test #5: PushSyncND param and deleter =====";
    int* b = new int(101);
    res    = MXEnginePushSyncND(FooSyncFunc,
                             b,
                             FooFuncDeleter,
                             &ctx,
                             const_nds_handle,
                             num_const_nds,
                             mutable_nds_handle,
                             num_mutable_nds);
    EXPECT_EQ(res, 0);

    // Test #6
    LOG(INFO) << "===== Test #6: PushSyncND NULL param and NULL deleter =====";
    res = MXEnginePushSyncND(FooSyncFunc,
                             nullptr,
                             nullptr,
                             &ctx,
                             const_nds_handle,
                             num_const_nds,
                             mutable_nds_handle,
                             num_mutable_nds);
    EXPECT_EQ(res, 0);

    // Test #7
    LOG(INFO) << "===== Test #7: PushSyncND invalid number of const nds =====";
    res = MXEnginePushSyncND(FooSyncFunc,
                             nullptr,
                             nullptr,
                             &ctx,
                             const_nds_handle,
                             -1,
                             mutable_nds_handle,
                             num_mutable_nds);
    EXPECT_EQ(res, -1);

    // Test #8
    LOG(INFO) << "===== Test #8: PushSyncND invalid number of mutable nds =====";
    res = MXEnginePushSyncND(FooSyncFunc,
                             nullptr,
                             nullptr,
                             &ctx,
                             const_nds_handle,
                             num_const_nds,
                             mutable_nds_handle,
                             -1);
    EXPECT_EQ(res, -1);
  }
  for (mxnet::NDArray* pnd : nds) {
    delete pnd;
  }
}

TEST(Engine, PushFuncNDRejectsNegativeCountsBeforeAllocatingVectors) {
  auto ctx = mxnet::Context{};
  mxnet::NDArray nd(ctx);
  void* nd_handle = &nd;

  int res = MXEnginePushAsyncND(
      FooAsyncFunc, nullptr, nullptr, &ctx, &nd_handle, -1, &nd_handle, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("Non-negative number of const vars"),
            std::string::npos)
      << MXGetLastError();

  res = MXEnginePushAsyncND(
      FooAsyncFunc, nullptr, nullptr, &ctx, &nd_handle, 1, &nd_handle, -1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("Non-negative number of mutable vars"),
            std::string::npos)
      << MXGetLastError();

  res = MXEnginePushSyncND(FooSyncFunc, nullptr, nullptr, &ctx, &nd_handle, -1, &nd_handle, 1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("Non-negative number of const vars"),
            std::string::npos)
      << MXGetLastError();

  res = MXEnginePushSyncND(FooSyncFunc, nullptr, nullptr, &ctx, &nd_handle, 1, &nd_handle, -1);
  EXPECT_EQ(res, -1);
  EXPECT_NE(std::string(MXGetLastError()).find("Non-negative number of mutable vars"),
            std::string::npos)
      << MXGetLastError();
}

#if !defined(_WIN32)
void RunPushFuncNDProfilingExitTest() {
  std::signal(SIGALRM, [](int) { _exit(5); });
  alarm(5);

  auto ctx = mxnet::Context{};
  const char* keys[] = {"profile_api",
                        "profile_symbolic",
                        "profile_imperative",
                        "profile_memory",
                        "aggregate_stats",
                        "continuous_dump",
                        "filename"};
  const char* vals[] = {"1", "0", "0", "0", "1", "0", "/tmp/mxnet_api_profile.json"};

  if (MXSetProfilerConfig(7, keys, vals) != 0) {
    _exit(1);
  }
  if (MXSetProfilerState(1) != 0) {
    _exit(2);
  }

  if (MXEnginePushAsyncND(FooAsyncFunc,
                          nullptr,
                          nullptr,
                          &ctx,
                          nullptr,
                          0,
                          nullptr,
                          0,
                          nullptr,
                          0,
                          "MXEnginePushAsyncNDProfileRegression",
                          true) != 0) {
    _exit(3);
  }
  if (MXEnginePushSyncND(FooSyncFunc,
                         nullptr,
                         nullptr,
                         &ctx,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         "MXEnginePushSyncNDProfileRegression") != 0) {
    _exit(4);
  }

  if (MXSetProfilerState(0) != 0) {
    _exit(6);
  }

  const char* stats = nullptr;
  if (MXAggregateProfileStatsPrint(&stats, 0, 1, 0, 1) != 0 || stats == nullptr) {
    _exit(7);
  }
  const std::string json(stats);
  if (json.find("MXEnginePushAsyncND") == std::string::npos ||
      json.find("MXEnginePushSyncND") == std::string::npos) {
    _exit(8);
  }
  _exit(0);
}

TEST(Engine, PushFuncNDReachesApiEndForProfiling) {
  EXPECT_EXIT(RunPushFuncNDProfilingExitTest(), ::testing::ExitedWithCode(0), "")
      << "MXEnginePushAsyncND/MXEnginePushSyncND must execute API_END so API profiling "
         "closes and records their outer C API duration tasks.";
}
#endif

TEST(Engine, basics) {
  auto&& engine = mxnet::Engine::Get();
  auto&& var    = engine->NewVariable();
  std::vector<mxnet::Engine::OprHandle> oprs;

  // Test #1
  printf("============= Test #1 ==============\n");
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx,
            mxnet::Engine::CallbackOnStart on_start,
            mxnet::Engine::CallbackOnComplete cb) {
          on_start();
          Foo(ctx, i);
          std::this_thread::sleep_for(std::chrono::seconds{1});
          cb();
        },
        {var},
        {}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  engine->WaitForAll();
  printf("Going to push delete\n");
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #2 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  for (int i = 0; i < 10; ++i) {
    oprs.push_back(engine->NewOperator(
        [i](mxnet::RunContext ctx,
            mxnet::Engine::CallbackOnStart on_start,
            mxnet::Engine::CallbackOnComplete cb) {
          on_start();
          Foo(ctx, i);
          std::this_thread::sleep_for(std::chrono::milliseconds{500});
          cb();
        },
        {},
        {var}));
    engine->Push(oprs.at(i), mxnet::Context{});
  }
  // std::this_thread::sleep_for(std::chrono::seconds{1});
  engine->WaitForAll();
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);

  printf("============= Test #3 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  engine->WaitForVar(var);
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #4 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  oprs.push_back(engine->NewOperator(
      [](mxnet::RunContext ctx,
         mxnet::Engine::CallbackOnStart on_start,
         mxnet::Engine::CallbackOnComplete cb) {
        std::this_thread::sleep_for(std::chrono::seconds{2});
        on_start();
        Foo(ctx, 42);
        cb();
      },
      {},
      {var},
      mxnet::FnProperty::kCopyFromGPU));
  engine->Push(oprs.at(0), mxnet::Context{});
  LOG(INFO) << "IO operator pushed, should wait for 2 seconds.";
  engine->WaitForVar(var);
  LOG(INFO) << "OK, here I am.";
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();

  printf("============= Test #5 ==============\n");
  var = engine->NewVariable();
  oprs.clear();
  oprs.push_back(engine->NewOperator(
      [](mxnet::RunContext ctx,
         mxnet::Engine::CallbackOnStart on_start,
         mxnet::Engine::CallbackOnComplete cb) {
        on_start();
        Foo(ctx, 42);
        std::this_thread::sleep_for(std::chrono::seconds{2});
        cb();
      },
      {var},
      {}));
  engine->Push(oprs.at(0), mxnet::Context{});
  LOG(INFO) << "Operator pushed, should not wait.";
  engine->WaitForVar(var);
  LOG(INFO) << "OK, here I am.";
  engine->WaitForAll();
  LOG(INFO) << "That was 2 seconds.";
  for (auto&& i : oprs) {
    engine->DeleteOperator(i);
  }
  engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
  engine->WaitForAll();
  var = nullptr;
  oprs.clear();
  LOG(INFO) << "All pass";
}

TEST(Engine, VarVersion) {
  const size_t num_engines = 3;
  std::vector<mxnet::Engine*> engines(num_engines);
  engines[0]                = mxnet::engine::CreateNaiveEngine();
  engines[1]                = mxnet::engine::CreateThreadedEnginePooled();
  engines[2]                = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[3] = {"NaiveEngine", "ThreadedEnginePooled", "ThreadedEnginePerDevice"};
  for (size_t k = 0; k < num_engines; ++k) {
    auto engine = engines[k];
    std::vector<mxnet::Engine::OprHandle> oprs;

    LOG(INFO) << "Testing var as a read dependency in " << type_names[k];
    auto var = engine->NewVariable();
    EXPECT_EQ(var->version(), 0U);
    for (int i = 0; i < 10; ++i) {
      oprs.push_back(engine->NewOperator(
          [i](mxnet::RunContext ctx,
              mxnet::Engine::CallbackOnStart on_start,
              mxnet::Engine::CallbackOnComplete cb) {
            on_start();
            Foo(ctx, i);
            cb();
          },
          {var},
          {}));
      engine->Push(oprs.at(i), mxnet::Context{});
    }
    engine->WaitForAll();
    EXPECT_EQ(var->version(), 0U);
    for (auto&& i : oprs) {
      engine->DeleteOperator(i);
    }
    engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
    engine->WaitForAll();

    LOG(INFO) << "Testing var as a write dependency in " << type_names[k];
    var = engine->NewVariable();
    EXPECT_EQ(var->version(), 0U);
    oprs.clear();
    for (int i = 0; i < 10; ++i) {
      oprs.push_back(engine->NewOperator(
          [i](mxnet::RunContext ctx,
              mxnet::Engine::CallbackOnStart on_start,
              mxnet::Engine::CallbackOnComplete cb) {
            on_start();
            Foo(ctx, i);
            cb();
          },
          {},
          {var}));
      engine->Push(oprs.at(i), mxnet::Context{});
    }
    engine->WaitForAll();
    EXPECT_EQ(var->version(), 10U);
    for (auto&& i : oprs) {
      engine->DeleteOperator(i);
    }
    engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
    engine->WaitForAll();

    var = nullptr;
    oprs.clear();
    LOG(INFO) << "All pass";
  }
}

#ifdef _OPENMP

struct TestSaveAndRestoreOMPState {
  TestSaveAndRestoreOMPState() {
    omp_set_dynamic(false);
  }
  ~TestSaveAndRestoreOMPState() {
    omp_set_num_threads(nthreads_);
    omp_set_dynamic(dynamic_);
  }
  const int nthreads_ = omp_get_max_threads();
  const int dynamic_  = omp_get_dynamic();
};

/*!
 * \brief This test checks that omp_set_num_threads implementation has thread-scope
 */
TEST(Engine, omp_threading_count_scope) {
  TestSaveAndRestoreOMPState omp_state;
  const int THREAD_COUNT                     = 10;
  std::shared_ptr<dmlc::ManualEvent> ready   = std::make_shared<dmlc::ManualEvent>();
  std::shared_ptr<dmlc::ThreadGroup> threads = std::make_shared<dmlc::ThreadGroup>();
  std::atomic<int> counter(0), correct(0);
  omp_set_dynamic(0);
  for (int x = 0; x < THREAD_COUNT; ++x) {
    std::string name = "thread: ";
    name += std::to_string(x + 1);
    ++counter;
    threads->create(
        name,
        false,
        [x, &counter, &correct](std::shared_ptr<dmlc::ManualEvent> ready_ptr) -> int {
          const int thread_count = x + 1;
          omp_set_num_threads(thread_count);
          --counter;
          ready_ptr->wait();
          CHECK_EQ(omp_get_max_threads(), thread_count);
#pragma omp parallel for
          for (int i = 0; i < 100; ++i) {
            if (i == 50) {
              const int current_threads = omp_get_num_threads();
              if (current_threads == thread_count) {
                ++correct;
              }
            }
          }
          return 0;
        },
        ready);
  }
  while (counter.load() > 0) {
    usleep(100);
  }
  ready->signal();
  threads->join_all();
  GTEST_ASSERT_EQ(correct.load(), THREAD_COUNT);
}
#endif  // _OPENMP
