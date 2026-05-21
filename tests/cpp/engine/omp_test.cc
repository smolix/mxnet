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

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "../include/test_util.h"
#include "../../src/engine/openmp.h"

#if defined(unix) || defined(__unix__) || defined(__unix)
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <dmlc/logging.h>

TEST(OMPBehaviour, after_fork) {
  /*
   * Check that after fork, OMP is disabled, and the recommended thread count is 1 to prevent
   * process fanout.
   */
  using namespace mxnet::engine;
  auto openmp = OpenMP::Get();
  pid_t pid   = fork();
  if (pid == 0) {
    EXPECT_FALSE(openmp->enabled());
    EXPECT_EQ(openmp->GetRecommendedOMPThreadCount(), 1);
    _exit(::testing::Test::HasFailure() ? 1 : 0);
  } else if (pid > 0) {
    int status;
    int ret = waitpid(pid, &status, 0);
    CHECK_EQ(ret, pid) << "waitpid failed";
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
  } else {
    CHECK(false) << "fork failed";
  }
}
#endif

TEST(OMPBehaviour, concurrent_state_access) {
  using namespace mxnet::engine;
  auto openmp = OpenMP::Get();

  const bool old_enabled     = openmp->enabled();
  const int old_thread_max   = openmp->thread_max();
  const int old_reserve_core = openmp->reserve_cores();

  constexpr int kThreads = 4;
  constexpr int kIters   = 1000;
  std::atomic<bool> start(false);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([openmp, &start, t]() {
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kIters; ++i) {
        openmp->set_enabled(((i + t) & 1) == 0);
        openmp->set_thread_max(1 + ((i + t) % 8));
        EXPECT_GE(openmp->thread_max(), 1);
        EXPECT_GE(openmp->reserve_cores(), 0);
        EXPECT_GE(openmp->GetRecommendedOMPThreadCount(false), 1);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  openmp->set_enabled(old_enabled);
  openmp->set_thread_max(old_thread_max);
  openmp->set_reserve_cores(old_reserve_core);
}
