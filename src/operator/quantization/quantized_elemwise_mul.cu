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
 * \file quantized_elemwise_mul.cu
 * \brief GPU implementation of calibrated quantized elemwise multiplication.
 */
#include <cstdint>
#include <mxnet/op_attr_types.h>
#include "./quantized_elemwise_mul-inl.h"
#include "./quantized_range_utils.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct quantized_elemwise_mul_calibrated_int8_kernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int8_t* out,
                                  const int8_t* lhs,
                                  const int8_t* rhs,
                                  const float* lhs_min,
                                  const float* lhs_max,
                                  const float* rhs_min,
                                  const float* rhs_max,
                                  const float out_min,
                                  const float out_max,
                                  const OpReqType req) {
    const float out_data_scale = kInt8Range / MaxAbs(out_min, out_max);
    const float lhs_scale      = kInt8Range / MaxAbs(lhs_min[0], lhs_max[0]);
    const float rhs_scale      = kInt8Range / MaxAbs(rhs_min[0], rhs_max[0]);
    const float out_scale      = out_data_scale / lhs_scale / rhs_scale;
    const float scaled = nearbyintf(static_cast<float>(lhs[i]) * static_cast<float>(rhs[i]) *
                                    out_scale);
    const int8_t value = static_cast<int8_t>(Min(Max(scaled, static_cast<float>(INT8_MIN)),
                                                static_cast<float>(INT8_MAX)));
    KERNEL_ASSIGN(out[i], req, value);
  }
};

struct quantized_elemwise_mul_float_kernel {
  MSHADOW_XINLINE static void Map(int i,
                                  float* out,
                                  const int8_t* lhs,
                                  const int8_t* rhs,
                                  const float* lhs_min,
                                  const float* lhs_max,
                                  const float* rhs_min,
                                  const float* rhs_max,
                                  const OpReqType req) {
    const float lhs_scale = kInt8Range / MaxAbs(lhs_min[0], lhs_max[0]);
    const float rhs_scale = kInt8Range / MaxAbs(rhs_min[0], rhs_max[0]);
    const float out_scale = 1.0f / lhs_scale / rhs_scale;
    const float value = static_cast<float>(lhs[i]) * static_cast<float>(rhs[i]) * out_scale;
    KERNEL_ASSIGN(out[i], req, value);
  }
};

// Non-calibrated int8 * int8 -> int32 output. With out_scale == 1 (matching the
// CPU op), the result is the exact integer product; the output range is derived
// separately from the input ranges via QuantizationRangeForS8S8Multiplication.
struct quantized_elemwise_mul_int32_kernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int32_t* out,
                                  const int8_t* lhs,
                                  const int8_t* rhs,
                                  const OpReqType req) {
    const int32_t value = static_cast<int32_t>(lhs[i]) * static_cast<int32_t>(rhs[i]);
    KERNEL_ASSIGN(out[i], req, value);
  }
};

void QuantizedElemwiseMulOpForwardGPU(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
  const QuantizeElemwiseMulParam& params = nnvm::get<QuantizeElemwiseMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();

  CHECK_EQ(inputs[quantized_elemwise_mul::kLhs].type_flag_, mshadow::kInt8);
  CHECK_EQ(inputs[quantized_elemwise_mul::kRhs].type_flag_, mshadow::kInt8);

  const size_t out_size = outputs[quantized_elemwise_mul::kOut].Size();
  const int8_t* input_l = inputs[quantized_elemwise_mul::kLhs].dptr<int8_t>();
  const int8_t* input_r = inputs[quantized_elemwise_mul::kRhs].dptr<int8_t>();
  const float* lhs_min  = inputs[quantized_elemwise_mul::kLhsMin].dptr<float>();
  const float* lhs_max  = inputs[quantized_elemwise_mul::kLhsMax].dptr<float>();
  const float* rhs_min  = inputs[quantized_elemwise_mul::kRhsMin].dptr<float>();
  const float* rhs_max  = inputs[quantized_elemwise_mul::kRhsMax].dptr<float>();

  if (params.enable_float_output) {
    if (req[quantized_elemwise_mul::kOut] != kNullOp) {
      mxnet_op::Kernel<quantized_elemwise_mul_float_kernel, gpu>::Launch(
          s,
          out_size,
          outputs[quantized_elemwise_mul::kOut].dptr<float>(),
          input_l,
          input_r,
          lhs_min,
          lhs_max,
          rhs_min,
          rhs_max,
          req[quantized_elemwise_mul::kOut]);
    }
    return;
  }

  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    CHECK_EQ(outputs[quantized_elemwise_mul::kOut].type_flag_, mshadow::kInt8);
    if (req[quantized_elemwise_mul::kOut] != kNullOp) {
      mxnet_op::Kernel<quantized_elemwise_mul_calibrated_int8_kernel, gpu>::Launch(
          s,
          out_size,
          outputs[quantized_elemwise_mul::kOut].dptr<int8_t>(),
          input_l,
          input_r,
          lhs_min,
          lhs_max,
          rhs_min,
          rhs_max,
          params.min_calib_range.value(),
          params.max_calib_range.value(),
          req[quantized_elemwise_mul::kOut]);
    }
    AssignQuantizedRangeOutput<gpu>(
        s,
        outputs[quantized_elemwise_mul::kOutMin],
        params.min_calib_range.value(),
        req[quantized_elemwise_mul::kOutMin]);
    AssignQuantizedRangeOutput<gpu>(
        s,
        outputs[quantized_elemwise_mul::kOutMax],
        params.max_calib_range.value(),
        req[quantized_elemwise_mul::kOutMax]);
  } else {
    // Non-calibrated int8 * int8 -> int32 output. The exact integer product is
    // written on device; the output range is computed on device from the four
    // input range scalars (no host dereference of device pointers), matching the
    // CPU op and the quantized_fully_connected GPU precedent.
    CHECK_EQ(outputs[quantized_elemwise_mul::kOut].type_flag_, mshadow::kInt32);
    if (req[quantized_elemwise_mul::kOut] != kNullOp) {
      mxnet_op::Kernel<quantized_elemwise_mul_int32_kernel, gpu>::Launch(
          s,
          out_size,
          outputs[quantized_elemwise_mul::kOut].dptr<int32_t>(),
          input_l,
          input_r,
          req[quantized_elemwise_mul::kOut]);
    }
    if (req[quantized_elemwise_mul::kOutMin] != kNullOp ||
        req[quantized_elemwise_mul::kOutMax] != kNullOp) {
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, gpu>::Launch(
          s,
          1,
          outputs[quantized_elemwise_mul::kOutMin].dptr<float>(),
          outputs[quantized_elemwise_mul::kOutMax].dptr<float>(),
          lhs_min,
          lhs_max,
          rhs_min,
          rhs_max);
    }
  }
}

NNVM_REGISTER_OP(_contrib_quantized_elemwise_mul)
    .set_attr<FCompute>("FCompute<gpu>", QuantizedElemwiseMulOpForwardGPU);

}  // namespace op
}  // namespace mxnet
