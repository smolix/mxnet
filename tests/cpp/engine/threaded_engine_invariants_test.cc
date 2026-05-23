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
 * \file threaded_engine_invariants_test.cc
 * \brief Smoke tests guarding ThreadedVar/VersionedVarBlock invariants
 *        that were previously protected only by assert() and therefore
 *        stripped in release (-DNDEBUG) builds.
 */
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <mxnet/engine.h>
#include <atomic>
#include <thread>
#include <vector>

#include "../src/engine/engine_impl.h"

namespace {

void NoOp(mxnet::RunContext,
          mxnet::Engine::CallbackOnStart on_start,
          mxnet::Engine::CallbackOnComplete on_complete) {
  on_start();
  on_complete();
}

}  // namespace

// Push many read deps then a write dep concurrently on a shared variable.
// Exercises the AppendReadDependency / AppendWriteDependency /
// CompleteReadDependency / CompleteWriteDependency invariants that used
// to be assert()-only.  Acts as a smoke test for the var block chain
// manipulation paths; an actual corruption would manifest as a CHECK
// failure (process abort with diagnostic) rather than silent UB.
TEST(Engine, WriteAfterReadChainTermination) {
  const size_t num_engines = 2;
  std::vector<mxnet::Engine*> engines(num_engines);
  engines[0] = mxnet::engine::CreateThreadedEnginePooled();
  engines[1] = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  constexpr int kReaders = 64;
  constexpr int kRounds  = 32;

  for (size_t e = 0; e < num_engines; ++e) {
    auto* engine = engines[e];
    LOG(INFO) << "Testing read-then-write chain termination in " << type_names[e];

    for (int round = 0; round < kRounds; ++round) {
      auto* var = engine->NewVariable();

      std::atomic<bool> start(false);
      std::vector<std::thread> readers;
      readers.reserve(kReaders);
      for (int t = 0; t < kReaders; ++t) {
        readers.emplace_back([engine, var, &start]() {
          while (!start.load(std::memory_order_acquire)) {
          }
          engine->PushAsync(NoOp,
                            mxnet::Context::CPU(),
                            {var},
                            {},
                            mxnet::FnProperty::kNormal,
                            0,
                            "InvariantSmokeRead");
        });
      }

      start.store(true, std::memory_order_release);
      for (auto& th : readers) {
        th.join();
      }

      // Now push a write dependency, which forces the read chain to
      // terminate and triggers the assert() sites we replaced with CHECK.
      engine->PushAsync(NoOp,
                        mxnet::Context::CPU(),
                        {},
                        {var},
                        mxnet::FnProperty::kNormal,
                        0,
                        "InvariantSmokeWrite");

      engine->WaitForAll();
      engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
      engine->WaitForAll();
    }
  }
}

// Rapidly allocate and delete variables across many threads to exercise
// VersionedVarBlock chain manipulation under contention.  Any violation
// of the invariants (head_->next == nullptr, head_->trigger == nullptr,
// !head_->write) at append-time would surface as a CHECK failure now
// that the asserts have been promoted; previously, release builds would
// silently corrupt the linked list.
TEST(Engine, RapidVarAllocDelete) {
  const size_t num_engines = 2;
  std::vector<mxnet::Engine*> engines(num_engines);
  engines[0] = mxnet::engine::CreateThreadedEnginePooled();
  engines[1] = mxnet::engine::CreateThreadedEnginePerDevice();
  std::string type_names[2] = {"ThreadedEnginePooled", "ThreadedEnginePerDevice"};

  constexpr int kThreads      = 8;
  constexpr int kPerThreadOps = 1000;

  for (size_t e = 0; e < num_engines; ++e) {
    auto* engine = engines[e];
    LOG(INFO) << "Testing rapid New/DeleteVariable in " << type_names[e];

    std::atomic<bool> start(false);
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      workers.emplace_back([engine, &start]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        for (int i = 0; i < kPerThreadOps; ++i) {
          auto* var = engine->NewVariable();
          // Push a trivial read + write so AppendReadDependency and
          // AppendWriteDependency both run against this var.
          engine->PushAsync(NoOp,
                            mxnet::Context::CPU(),
                            {var},
                            {},
                            mxnet::FnProperty::kNormal,
                            0,
                            "RapidRead");
          engine->PushAsync(NoOp,
                            mxnet::Context::CPU(),
                            {},
                            {var},
                            mxnet::FnProperty::kNormal,
                            0,
                            "RapidWrite");
          engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
        }
      });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : workers) {
      th.join();
    }
    engine->WaitForAll();
  }
}

// XOP23: Repeatedly create, push trivial work onto, and delete a single
// variable in tight rotation across multiple threads.  Stresses the
// var-deletion fast path, which used to rely on assert() to guard the
// "delete-while-busy" + "deletion-already-scheduled" invariants.  A real
// double-delete or use-after-free would manifest as a sanitizer abort or
// data corruption that the next iteration's pointer reuse exposes.
TEST(Engine, ShutdownRaceCreateUseDeleteCycle) {
  mxnet::Engine* engine = mxnet::engine::CreateThreadedEnginePooled();
  constexpr int kThreads = 8;
  constexpr int kCycles  = 256;

  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([engine, &start]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kCycles; ++i) {
        auto* var = engine->NewVariable();
        // Mix push then immediate delete so the engine has to drain the
        // pending op before reclaiming the var.
        engine->PushAsync(NoOp,
                          mxnet::Context::CPU(),
                          {},
                          {var},
                          mxnet::FnProperty::kNormal,
                          0,
                          "ShutdownRaceWrite");
        engine->DeleteVariable([](mxnet::RunContext) {}, mxnet::Context{}, var);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& th : workers) {
    th.join();
  }
  engine->WaitForAll();
}
