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

#include <dmlc/logging.h>
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

NDArray CreateFloatArray(const TShape& shape, const std::vector<float>& values) {
  CHECK_EQ(static_cast<size_t>(shape.Size()), values.size());
  NDArray array(shape, Context::CPU(), true, mshadow::kFloat32);
  array.CheckAndAlloc();
  size_t idx = 0;
  TBlob data = array.data();
  patternFill(RunContext{Context::CPU(), nullptr, nullptr}, &data, [&values, &idx]() {
    return values[idx++];
  });
  return array;
}

NDArray CreateFloatArray(const TShape& shape, float value) {
  NDArray array(shape, Context::CPU(), true, mshadow::kFloat32);
  array.CheckAndAlloc();
  TBlob data = array.data();
  patternFill(RunContext{Context::CPU(), nullptr, nullptr}, &data, [value]() { return value; });
  return array;
}

std::vector<float> ReadFloatArray(const NDArray& array) {
  std::vector<float> values;
  values.reserve(array.shape().Size());
  AccessAsCPU(array, RunContext{array.ctx(), nullptr, nullptr}, [&values](const NDArray& cpu_array) {
    const float* data = cpu_array.data().dptr<float>();
    values.assign(data, data + cpu_array.shape().Size());
  });
  return values;
}

void ExpectFloatArrayEq(const NDArray& array, const std::vector<float>& expected) {
  const std::vector<float> actual = ReadFloatArray(array);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << " at flat index " << i;
  }
}

void InitRandomCropExecutor(const NDArray& data,
                            const NDArray& out,
                            const NDArray& temp,
                            CoreOpExecutor<float>* executor) {
  const kwargs_t args = {{COREOP_FWD_OP_NAME_KEY, "_image_random_crop"},
                         {"width", "4"},
                         {"height", "4"},
                         {"xrange", "(0, 0)"},
                         {"yrange", "(0, 0)"},
                         {COREOP_BWD_OP_NAME_KEY, COREOP_BWD_OP_NAME_VALUE_NONE}};
  executor->Init(args, {data}, {out, temp});
}

}  // namespace

TEST(ImageRandomCrop, ResizePathNullOpPreservesOutput) {
  const NDArray data = CreateFloatArray(TShape({2, 2, 1}), {1.f, 2.f, 3.f, 4.f});
  NDArray out        = CreateFloatArray(TShape({4, 4, 1}), 7.f);
  NDArray temp       = CreateFloatArray(TShape({2, 2, 1}), -1.f);

  CoreOpExecutor<float> executor(false, {data.shape()});
  InitRandomCropExecutor(data, out, temp, &executor);
  executor.set_requests({kNullOp, kNullOp});
  executor.Execute();

  ExpectFloatArrayEq(out, std::vector<float>(16, 7.f));
}

TEST(ImageRandomCrop, ResizePathAddToRejectedBeforeOutputWrite) {
  const NDArray data = CreateFloatArray(TShape({2, 2, 1}), {1.f, 2.f, 3.f, 4.f});
  NDArray out        = CreateFloatArray(TShape({4, 4, 1}), 7.f);
  NDArray temp       = CreateFloatArray(TShape({2, 2, 1}), -1.f);

  CoreOpExecutor<float> executor(false, {data.shape()});
  InitRandomCropExecutor(data, out, temp, &executor);
  executor.set_requests({kAddTo, kNullOp});

  EXPECT_THROW(executor.Execute(), dmlc::Error);
  ExpectFloatArrayEq(out, std::vector<float>(16, 7.f));
}

}  // namespace test
}  // namespace mxnet
