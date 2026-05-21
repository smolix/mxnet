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
#include <mxnet/c_api.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kCPU = 1;
constexpr int kFloat32 = 0;

void AssertMXSuccess(int status) {
  ASSERT_EQ(status, 0) << MXGetLastError();
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

SymbolHandle CreateAddSymbol() {
  SymbolHandle lhs = nullptr;
  SymbolHandle rhs = nullptr;
  SymbolHandle add = nullptr;
  AssertMXSuccess(MXSymbolCreateVariable("lhs", &lhs));
  AssertMXSuccess(MXSymbolCreateVariable("rhs", &rhs));

  AtomicSymbolCreator add_creator = FindAtomicSymbolCreator("elemwise_add");
  EXPECT_NE(add_creator, nullptr);
  if (add_creator == nullptr) {
    MXSymbolFree(lhs);
    MXSymbolFree(rhs);
    return nullptr;
  }

  AssertMXSuccess(MXSymbolCreateAtomicSymbol(add_creator, 0, nullptr, nullptr, &add));
  SymbolHandle args[] = {lhs, rhs};
  AssertMXSuccess(MXSymbolCompose(add, "sum", 2, nullptr, args));
  MXSymbolFree(lhs);
  MXSymbolFree(rhs);
  return add;
}

NDArrayHandle CreateVector(const std::vector<float>& values) {
  NDArrayHandle handle = nullptr;
  const uint32_t shape[] = {static_cast<uint32_t>(values.size())};
  AssertMXSuccess(MXNDArrayCreate(shape, 1, kCPU, 0, 0, kFloat32, &handle));
  AssertMXSuccess(MXNDArraySyncCopyFromCPU(handle, values.data(), values.size()));
  return handle;
}

bool InvokeAdd(CachedOpHandle op, int thread_id, std::string* error) {
  const std::vector<float> lhs = {static_cast<float>(thread_id), static_cast<float>(thread_id + 1)};
  const std::vector<float> rhs = {10.0f, 20.0f};
  NDArrayHandle lhs_handle = CreateVector(lhs);
  NDArrayHandle rhs_handle = CreateVector(rhs);
  NDArrayHandle out_handle = CreateVector({0.0f, 0.0f});
  NDArrayHandle inputs[] = {lhs_handle, rhs_handle};
  NDArrayHandle outputs[] = {out_handle};
  NDArrayHandle* output_ptr = outputs;
  int num_outputs = 1;

  int status = MXInvokeCachedOp(op, 2, inputs, kCPU, 0, &num_outputs, &output_ptr, nullptr);
  if (status != 0) {
    *error = MXGetLastError();
    MXNDArrayFree(lhs_handle);
    MXNDArrayFree(rhs_handle);
    MXNDArrayFree(out_handle);
    return false;
  }

  std::vector<float> actual(2, 0.0f);
  status = MXNDArraySyncCopyToCPU(out_handle, actual.data(), actual.size());
  MXNDArrayFree(lhs_handle);
  MXNDArrayFree(rhs_handle);
  MXNDArrayFree(out_handle);

  if (status != 0) {
    *error = MXGetLastError();
    return false;
  }
  if (num_outputs != 1 || output_ptr != outputs) {
    std::ostringstream os;
    os << "MXInvokeCachedOp did not preserve caller-owned output array";
    *error = os.str();
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    const float expected = lhs[i] + rhs[i];
    if (std::fabs(actual[i] - expected) > 1e-5f) {
      std::ostringstream os;
      os << "output[" << i << "] expected " << expected << " got " << actual[i];
      *error = os.str();
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(CAPICachedOp, ThreadSafeInvokePreservesCallerOwnedOutputs) {
  SymbolHandle add = CreateAddSymbol();
  ASSERT_NE(add, nullptr);

  CachedOpHandle op = nullptr;
  AssertMXSuccess(MXCreateCachedOp(add, 0, nullptr, nullptr, &op, true));

  constexpr int kNumThreads = 4;
  constexpr int kIterations = 8;
  std::mutex errors_mutex;
  std::vector<std::string> errors;
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    threads.emplace_back([&, thread_id] {
      for (int iter = 0; iter < kIterations; ++iter) {
        std::string error;
        if (!InvokeAdd(op, thread_id * kIterations + iter, &error)) {
          std::lock_guard<std::mutex> lock(errors_mutex);
          errors.push_back(error);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  MXFreeCachedOp(op);
  MXSymbolFree(add);

  ASSERT_TRUE(errors.empty()) << errors.front();
}
