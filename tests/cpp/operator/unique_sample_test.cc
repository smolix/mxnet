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
#include <mxnet/resource.h>
#include <mxnet/tensor_blob.h>
#include <vector>
#include "../../../src/operator/random/unique_sample_op.h"
#include "../include/test_util.h"

namespace mxnet {
namespace test {

namespace {

NDArray CreateInt64Array(const TShape& shape, int64_t value) {
  const Context ctx = Context::CPU();
  NDArray array(shape, ctx, true, mshadow::kInt64);
  array.CheckAndAlloc();
  TBlob data = array.data();
  patternFill(RunContext{ctx, nullptr, nullptr}, &data, [value]() { return value; });
  return array;
}

std::vector<int64_t> ReadInt64Array(const NDArray& array) {
  std::vector<int64_t> values;
  values.reserve(array.shape().Size());
  AccessAsCPU(
      array, RunContext{array.ctx(), nullptr, nullptr}, [&values](const NDArray& cpu_array) {
        const int64_t* data = cpu_array.data().dptr<int64_t>();
        values.assign(data, data + cpu_array.shape().Size());
      });
  return values;
}

nnvm::NodeAttrs UniqueZipfianAttrs(int range_max, const TShape& shape) {
  op::SampleUniqueZifpianParam param;
  param.range_max = range_max;
  param.shape     = shape;
  nnvm::NodeAttrs attrs;
  attrs.parsed = param;
  return attrs;
}

OpContext UniqueZipfianContext() {
  OpContext ctx;
  ctx.is_train             = false;
  ctx.run_ctx.ctx          = Context::CPU();
  ctx.run_ctx.stream       = nullptr;
  Resource random_resource = ResourceManager::Get()->Request(
      ctx.run_ctx.ctx, ResourceRequest(ResourceRequest::kParallelRandom));
  common::random::RandGenerator<cpu, double>::AllocState(
      random_resource.get_parallel_random<cpu, double>());
  ctx.requested.emplace_back(random_resource);
  ctx.requested.emplace_back(
      ResourceManager::Get()->Request(ctx.run_ctx.ctx,
                                      ResourceRequest(ResourceRequest::kTempSpace)));
  return ctx;
}

void RunUniqueZipfian(const std::vector<OpReqType>& req,
                      const NDArray& samples,
                      const NDArray& trials) {
  nnvm::NodeAttrs attrs = UniqueZipfianAttrs(20, TShape({2, 3}));
  OpContext ctx         = UniqueZipfianContext();
  std::vector<TBlob> inputs;
  std::vector<TBlob> outputs{samples.data(), trials.data()};
  op::SampleUniqueZifpian(attrs, ctx, inputs, req, outputs);
}

}  // namespace

TEST(UniqueSampleZipfian, MixedNullAndWriteRequestsPreserveSamplesOnly) {
  NDArray samples = CreateInt64Array(TShape({2, 3}), -9);
  NDArray trials  = CreateInt64Array(TShape({2}), -7);

  RunUniqueZipfian({kNullOp, kWriteTo}, samples, trials);

  for (const int64_t value : ReadInt64Array(samples)) {
    EXPECT_EQ(value, -9);
  }
  for (const int64_t value : ReadInt64Array(trials)) {
    EXPECT_GE(value, 3);
  }
}

TEST(UniqueSampleZipfian, MixedAddToAndNullRequestsAccumulateSamplesOnly) {
  NDArray samples = CreateInt64Array(TShape({2, 3}), 100);
  NDArray trials  = CreateInt64Array(TShape({2}), -7);

  RunUniqueZipfian({kAddTo, kNullOp}, samples, trials);

  for (const int64_t value : ReadInt64Array(samples)) {
    EXPECT_GE(value, 100);
    EXPECT_LT(value, 120);
  }
  for (const int64_t value : ReadInt64Array(trials)) {
    EXPECT_EQ(value, -7);
  }
}

}  // namespace test
}  // namespace mxnet
