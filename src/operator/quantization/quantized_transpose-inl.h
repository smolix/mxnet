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

#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_TRANSPOSE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_TRANSPOSE_INL_H_

#include <mxnet/op_attr_types.h>
#include "../tensor/matrix_op-inl.h"
#include "../numpy/np_matrix_op-inl.h"
#include "./quantized_range_utils.h"

namespace mxnet {
namespace op {

inline bool QuantizedTransposeType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

typedef bool (*TransposeShapeFunAny)(const nnvm::NodeAttrs&,
                                     mxnet::ShapeVector*,
                                     mxnet::ShapeVector*);
typedef void (*TransposeComputeFunAny)(const nnvm::NodeAttrs&,
                                       const OpContext&,
                                       const std::vector<TBlob>&,
                                       const std::vector<OpReqType>&,
                                       const std::vector<TBlob>&);

template <TransposeShapeFunAny TransposeShapeFun>
inline bool QuantizedTransposeShape(const nnvm::NodeAttrs& attrs,
                                    mxnet::ShapeVector* in_attrs,
                                    mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  mxnet::ShapeVector qin_attrs(1);
  mxnet::ShapeVector qout_attrs(1);
  SHAPE_ASSIGN_CHECK(qin_attrs, 0, (*in_attrs)[0]);
  SHAPE_ASSIGN_CHECK(qout_attrs, 0, (*out_attrs)[0]);
  bool ret = TransposeShapeFun(attrs, &qin_attrs, &qout_attrs);
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, qin_attrs[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, qout_attrs[0]);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, mxnet::TShape{1});
  return ret;
}

template <typename xpu, TransposeComputeFunAny TransposeComputeFun>
void QuantizedTransposeCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(req.size(), 3U);

  TransposeComputeFun(attrs, ctx, {inputs[0]}, {req[0]}, {outputs[0]});
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  AssignQuantizedRangeOutput<xpu>(s, outputs[1], inputs[1], req[1]);
  AssignQuantizedRangeOutput<xpu>(s, outputs[2], inputs[2], req[2]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_TRANSPOSE_INL_H_
