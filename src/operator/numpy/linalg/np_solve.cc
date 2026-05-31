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
 * \file np_solve.cc
 * \brief CPU implementation placeholder of Solve Operator
 */
#include <mxnet/operator_util.h>
#include <vector>
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"

#include "./np_solve-inl.h"

namespace mxnet {
namespace op {

inline bool SolveOpShape(const nnvm::NodeAttrs& attrs,
                         std::vector<mxnet::TShape>* in_attrs,
                         std::vector<mxnet::TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& in_a_shape = (*in_attrs)[0];
  const mxnet::TShape& in_b_shape = (*in_attrs)[1];
  if (!ndim_is_known(in_a_shape)) {
    return false;
  }
  int in_a_ndim = in_a_shape.ndim(), in_b_ndim = in_b_shape.ndim();

  CHECK_GE(in_a_ndim, 2) << "Array must be at least two-dimensional";
  CHECK_EQ(in_a_shape[in_a_ndim - 2], in_a_shape[in_a_ndim - 1])
      << "Input A's last two dimension must be equal";

  const bool vector_rhs = in_b_ndim == 1 || in_a_ndim == in_b_ndim + 1;
  if (vector_rhs) {
    CHECK_EQ(in_a_shape[in_a_ndim - 1], in_b_shape[in_b_ndim - 1])
        << "Input A's and B's last dimension must be equal";
  } else if (in_a_ndim == in_b_ndim) {
    CHECK_EQ(in_a_shape[in_a_ndim - 1], in_b_shape[in_b_ndim - 2])
        << "Input A's and B's last second dimension must be equal";
  } else {
    dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "A's and B's dimensions don't match";
  }

  const int a_batch_ndim = in_a_ndim - 2;
  const int b_batch_ndim = vector_rhs ? in_b_ndim - 1 : in_b_ndim - 2;
  const int out_batch_ndim = std::max(a_batch_ndim, b_batch_ndim);
  const int out_core_ndim  = vector_rhs ? 1 : 2;
  mxnet::TShape out_shape(out_batch_ndim + out_core_ndim, 1);
  for (int i = 0; i < out_batch_ndim; ++i) {
    const int a_axis = a_batch_ndim - out_batch_ndim + i;
    const int b_axis = b_batch_ndim - out_batch_ndim + i;
    const dim_t a_dim = a_axis < 0 ? 1 : in_a_shape[a_axis];
    const dim_t b_dim = b_axis < 0 ? 1 : in_b_shape[b_axis];
    CHECK(a_dim == b_dim || a_dim == 1 || b_dim == 1) << "A's and B's dimensions don't match";
    out_shape[i] = std::max(a_dim, b_dim);
  }
  if (vector_rhs) {
    out_shape[out_batch_ndim] = in_b_shape[in_b_ndim - 1];
  } else {
    out_shape[out_batch_ndim]     = in_b_shape[in_b_ndim - 2];
    out_shape[out_batch_ndim + 1] = in_b_shape[in_b_ndim - 1];
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  return !mxnet::op::shape_is_none(in_b_shape) && !mxnet::op::shape_is_none(out_attrs->at(0));
}

inline bool SolveOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  int a_type = in_attrs->at(0);
  int b_type = in_attrs->at(1);
  // unsupport float16
  CHECK_NE(a_type, mshadow::kFloat16) << "array type float16 is unsupported in linalg";
  CHECK_NE(b_type, mshadow::kFloat16) << "array type float16 is unsupported in linalg";
  if (mshadow::kFloat32 == a_type && mshadow::kFloat32 == b_type) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  }
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_solve)
    .describe(R"code()code" ADD_FILELINE)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"A", "B"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", SolveOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SolveOpType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<FCompute>("FCompute<cpu>", LaOpForwardSolve<cpu, 2, 2, 2, 1, solve>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_npi_solve"})
    .add_argument("A", "NDArray-or-Symbol", "Tensor of square matrix")
    .add_argument("B", "NDArray-or-Symbol", "Tensor of right side vector");

NNVM_REGISTER_OP(_backward_npi_solve)
    .set_num_inputs(4)
    .set_num_outputs(2)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs&) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", LaOpBackwardSolve<cpu, 2, 2, 4, 2, solve_backward>);

}  // namespace op
}  // namespace mxnet
