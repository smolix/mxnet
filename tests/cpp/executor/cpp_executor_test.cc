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

#include <vector>

#include "mxnet-cpp/operator.hpp"
#include "mxnet-cpp/ndarray.hpp"
#include "mxnet-cpp/executor.hpp"
#include "mxnet-cpp/symbol.hpp"

namespace mxnet {
namespace cpp {

TEST(CppExecutor, AlignsGradientRequestsWithBoundGradArrays) {
  const Context ctx = Context::cpu();
  Symbol x("x");
  Symbol y("y");
  Symbol z = x * y;

  const mx_float x_value = 3.0f;
  const mx_float y_value = 4.0f;
  const mx_float grad_init = 0.0f;
  NDArray x_arg(&x_value, Shape(1), ctx);
  NDArray y_arg(&y_value, Shape(1), ctx);
  NDArray y_grad(&grad_init, Shape(1), ctx);

  Executor exec(
      z, ctx, {x_arg, y_arg}, {NDArray(), y_grad}, {OpReqType::kNullOp, OpReqType::kWriteTo}, {});
  exec.Forward(true);
  exec.Backward();
  NDArray::WaitAll();

  std::vector<mx_float> grad_data;
  exec.grad_arrays[1].SyncCopyToCPU(&grad_data, 1);
  ASSERT_EQ(grad_data.size(), 1U);
  EXPECT_NEAR(grad_data[0], x_value, 1e-5f);
}

}  // namespace cpp
}  // namespace mxnet
