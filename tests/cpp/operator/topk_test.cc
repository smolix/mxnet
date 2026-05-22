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
#include <mxnet/ndarray.h>
#include <mxnet/tensor_blob.h>
#include <vector>
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"
#include "../include/test_util.h"

namespace mxnet {
namespace test {

namespace {

using op::CoreOpExecutor;
using op::kwargs_t;

NDArray CreateArray(const TShape& shape, const Context& ctx, const std::vector<float>& values) {
  CHECK_EQ(static_cast<size_t>(shape.Size()), values.size());
  NDArray array(shape, ctx, true, mshadow::kFloat32);
  array.CheckAndAlloc();
  size_t idx = 0;
  TBlob data = array.data();
  patternFill(RunContext{ctx, nullptr, nullptr}, &data, [&values, &idx]() { return values[idx++]; });
  return array;
}

void ExpectArrayEq(const NDArray& array, const std::vector<float>& expected) {
  ASSERT_EQ(static_cast<size_t>(array.shape().Size()), expected.size());
  AccessAsCPU(array, RunContext{array.ctx(), nullptr, nullptr}, [&expected](const NDArray& cpu_array) {
    const float* data = cpu_array.data().dptr<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_FLOAT_EQ(data[i], expected[i]) << " at flat index " << i;
    }
  });
}

void InitTopKMaskExecutor(const NDArray& data, const NDArray& out, CoreOpExecutor<float>* executor) {
  const kwargs_t args = {{COREOP_FWD_OP_NAME_KEY, "topk"},
                         {"axis", "1"},
                         {"k", "2"},
                         {"ret_typ", "mask"},
                         {"is_ascend", "False"},
                         {COREOP_BWD_OP_NAME_KEY, COREOP_BWD_OP_NAME_VALUE_NONE}};
  executor->Init(args, {data}, {out});
}

}  // namespace

TEST(TopK, MaskForwardNullOpPreservesOutput) {
  const Context ctx = Context::CPU();
  const TShape shape({2, 4});
  NDArray data      = CreateArray(shape, ctx, {1.f, 4.f, 2.f, 3.f, 7.f, 5.f, 6.f, 4.f});
  NDArray out       = CreateArray(shape, ctx, {7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f});

  CoreOpExecutor<float> executor(false, {data.shape()});
  InitTopKMaskExecutor(data, out, &executor);
  executor.set_requests({kNullOp});
  executor.Execute();

  ExpectArrayEq(out, {7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f});
}

TEST(TopK, MaskForwardAddToAccumulatesSelectedMaskEntries) {
  const Context ctx = Context::CPU();
  const TShape shape({2, 4});
  NDArray data      = CreateArray(shape, ctx, {1.f, 4.f, 2.f, 3.f, 7.f, 5.f, 6.f, 4.f});
  NDArray out       = CreateArray(shape, ctx, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f});

  CoreOpExecutor<float> executor(false, {data.shape()});
  InitTopKMaskExecutor(data, out, &executor);
  executor.set_requests({kAddTo});
  executor.Execute();

  ExpectArrayEq(out, {2.f, 3.f, 2.f, 3.f, 3.f, 2.f, 3.f, 2.f});
}

}  // namespace test
}  // namespace mxnet
