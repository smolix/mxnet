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

#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_

#include <dmlc/logging.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/tensor_blob.h>
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

inline void AssignQuantizedRangeOutput(float* output,
                                       const float* input,
                                       const OpReqType req,
                                       const char* op_name) {
  switch (req) {
    case kNullOp:
      return;
    case kWriteTo:
    case kWriteInplace:
      output[0] = input[0];
      return;
    case kAddTo:
      output[0] += input[0];
      return;
    default:
      LOG(FATAL) << "Unsupported request type for " << op_name << " range output";
  }
}

struct assign_quantized_range_output {
  MSHADOW_XINLINE static void Map(int,
                                  float* output,
                                  const float* input,
                                  const OpReqType req) {
    KERNEL_ASSIGN(output[0], req, input[0]);
  }
};

struct assign_quantized_range_output_value {
  MSHADOW_XINLINE static void Map(int,
                                  float* output,
                                  const float value,
                                  const OpReqType req) {
    KERNEL_ASSIGN(output[0], req, value);
  }
};

struct assign_quantized_zero_centered_range_output {
  MSHADOW_XINLINE static void Map(int,
                                  float* omin_range,
                                  float* omax_range,
                                  const float* imin_range,
                                  const float* imax_range,
                                  const OpReqType min_req,
                                  const OpReqType max_req) {
    const float real_range = MaxAbs(imin_range[0], imax_range[0]);
    KERNEL_ASSIGN(omin_range[0], min_req, -real_range);
    KERNEL_ASSIGN(omax_range[0], max_req, real_range);
  }
};

template <typename xpu>
inline void AssignQuantizedRangeOutput(mshadow::Stream<xpu>* s,
                                       const TBlob& output,
                                       const TBlob& input,
                                       const OpReqType req) {
  if (req == kNullOp)
    return;
  mxnet_op::Kernel<assign_quantized_range_output, xpu>::Launch(
      s, 1, output.dptr<float>(), input.dptr<float>(), req);
}

template <typename xpu>
inline void AssignQuantizedRangeOutput(mshadow::Stream<xpu>* s,
                                       const TBlob& output,
                                       const float value,
                                       const OpReqType req) {
  if (req == kNullOp)
    return;
  mxnet_op::Kernel<assign_quantized_range_output_value, xpu>::Launch(
      s, 1, output.dptr<float>(), value, req);
}

template <typename xpu>
inline void AssignQuantizedZeroCenteredRangeOutput(mshadow::Stream<xpu>* s,
                                                  const TBlob& omin_range,
                                                  const TBlob& omax_range,
                                                  const TBlob& imin_range,
                                                  const TBlob& imax_range,
                                                  const OpReqType min_req,
                                                  const OpReqType max_req) {
  if (min_req == kNullOp && max_req == kNullOp)
    return;
  mxnet_op::Kernel<assign_quantized_zero_centered_range_output, xpu>::Launch(
      s,
      1,
      omin_range.dptr<float>(),
      omax_range.dptr<float>(),
      imin_range.dptr<float>(),
      imax_range.dptr<float>(),
      min_req,
      max_req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_
