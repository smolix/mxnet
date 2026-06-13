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
 * \file quantize_v2-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include <type_traits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"
#include "./quantized_range_utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct QuantizeV2Param : public dmlc::Parameter<QuantizeV2Param> {
  int out_type;
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;
  DMLC_DECLARE_PARAMETER(QuantizeV2Param) {
    DMLC_DECLARE_FIELD(out_type)
        .add_enum("auto", QuantizeOutType::kAuto)
        .add_enum("int8", QuantizeOutType::kInt8)
        .add_enum("uint8", QuantizeOutType::kUint8)
        .set_default(QuantizeOutType::kInt8)
        .describe(
            "Output data type. `auto` can be specified to automatically determine output type "
            "according to min_calib_range.");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32. If present, it will be used to "
            "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32. If present, it will be used to "
            "quantize the fp32 data into int8 or uint8.");
  }
};

// quantize float to uint8_t
struct quantize_v2_unsigned {
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  const SrcDType* in,
                                  const float imin_range,
                                  const float imax_range,
                                  const double min_limit,
                                  const double max_limit,
                                  const OpReqType req) {
    const float scale = (max_limit - min_limit) / (imax_range - imin_range);
    const float rounded = (static_cast<float>(in[i]) - imin_range) * scale + 0.5f;
    const DstDType quantized =
        static_cast<DstDType>(Min(Max(rounded, static_cast<float>(min_limit)),
                                  static_cast<float>(max_limit)));
    KERNEL_ASSIGN(out[i], req, quantized);
  }

  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  const SrcDType* in,
                                  const float* imin_range,
                                  const float* imax_range,
                                  const double min_limit,
                                  const double max_limit,
                                  const OpReqType req) {
    Map(i, out, in, *imin_range, *imax_range, min_limit, max_limit, req);
  }
};

// keep zero-center
struct quantize_v2_zero_centered {
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  const SrcDType* in,
                                  const float imin_range,
                                  const float imax_range,
                                  const float quantized_range,
                                  const OpReqType req) {
    float real_range = MaxAbs(imin_range, imax_range);
    float scale      = quantized_range / real_range;
    const float x    = static_cast<float>(in[i]);
    const DstDType quantized =
        static_cast<DstDType>(Sign(x) * Min(Abs(x) * scale + 0.5f, quantized_range));
    KERNEL_ASSIGN(out[i], req, quantized);
  }

  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  const SrcDType* in,
                                  const float* imin_range,
                                  const float* imax_range,
                                  const float quantized_range,
                                  const OpReqType req) {
    Map(i, out, in, *imin_range, *imax_range, quantized_range, req);
  }
};

static inline bool QuantizeV2Shape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_attrs,
                                   std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);

  mxnet::TShape dshape = (*in_attrs)[0];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape(1, 1));

  if ((*out_attrs)[0].ndim() > 0) {
    dshape[0] = ((*out_attrs)[0])[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, dshape);
  }

  return !shape_is_none(out_attrs->at(0));
}

static inline bool QuantizeV2Type(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);
  const QuantizeV2Param& param = nnvm::get<QuantizeV2Param>(attrs.parsed);

#if MXNET_USE_ONEDNN == 1
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    CHECK(in_attrs->at(0) == mshadow::kFloat32 || in_attrs->at(0) == mshadow::kBfloat16 ||
          in_attrs->at(0) == mshadow::kUint8 || in_attrs->at(0) == mshadow::kInt8);
  } else {
    CHECK(in_attrs->at(0) == mshadow::kFloat32 || in_attrs->at(0) == mshadow::kUint8 ||
          in_attrs->at(0) == mshadow::kInt8);
  }
#else
  CHECK(in_attrs->at(0) == mshadow::kFloat32 || in_attrs->at(0) == mshadow::kUint8 ||
        in_attrs->at(0) == mshadow::kInt8);
#endif

  auto out_type = GetQuantizeOutputType(param);
  if (out_type == mshadow::kUint8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  } else if (out_type == mshadow::kInt8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  } else {
    LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

template <typename xpu>
class QuantizeV2Operator {
 public:
  explicit QuantizeV2Operator(const nnvm::NodeAttrs& attrs) : attrs_(attrs) {}

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    Stream<xpu>* s               = ctx.get_stream<xpu>();
    const QuantizeV2Param& param = nnvm::get<QuantizeV2Param>(attrs_.parsed);
    if (inputs[0].type_flag_ == mshadow::kUint8 || inputs[0].type_flag_ == mshadow::kInt8) {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        AssignQuantizedRangeOutput<xpu>(s, outputs[1], param.min_calib_range.value(), req[1]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[2], param.max_calib_range.value(), req[2]);
      } else {
        if (inputs[0].type_flag_ == mshadow::kUint8) {
          AssignQuantizedRangeOutput<xpu>(s, outputs[1], 0, req[1]);
          AssignQuantizedRangeOutput<xpu>(s, outputs[2], 255, req[2]);
        } else {
          AssignQuantizedRangeOutput<xpu>(s, outputs[1], -127, req[1]);
          AssignQuantizedRangeOutput<xpu>(s, outputs[2], 127, req[2]);
        }
      }
      UnaryOp::IdentityCompute<xpu>(attrs_, ctx, {inputs[0]}, req, outputs);
    } else if (inputs[0].type_flag_ == mshadow::kFloat32) {
      ForwardImpl<float>(ctx, inputs, req, outputs);
    } else if (inputs[0].type_flag_ == mshadow::kBfloat16) {
      ForwardImpl<mshadow::bfloat::bf16_t>(ctx, inputs, req, outputs);
    } else {
      LOG(FATAL) << "quantize_v2 only supports float32, bfloat16, int8, and uint8 inputs";
    }
  }

 private:
  template <typename SrcDType>
  void ForwardImpl(const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    using mshadow::red::limits::MaxValue;
    using mshadow::red::limits::MinValue;
    Stream<xpu>* s               = ctx.get_stream<xpu>();
    const QuantizeV2Param& param = nnvm::get<QuantizeV2Param>(attrs_.parsed);
    auto out_type                = GetQuantizeOutputType(param);

    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      if (out_type == mshadow::kUint8) {
        if (req[0] != kNullOp) {
          Kernel<quantize_v2_unsigned, xpu>::Launch(s,
                                                    outputs[0].Size(),
                                                    outputs[0].dptr<uint8_t>(),
                                                    inputs[0].dptr<SrcDType>(),
                                                    param.min_calib_range.value(),
                                                    param.max_calib_range.value(),
                                                    MinValue<uint8_t>(),
                                                    MaxValue<uint8_t>(),
                                                    req[0]);
        }
        AssignQuantizedRangeOutput<xpu>(s, outputs[1], param.min_calib_range.value(), req[1]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[2], param.max_calib_range.value(), req[2]);
      } else if (out_type == mshadow::kInt8) {  // zero-centered quantization
        if (req[0] != kNullOp) {
          Kernel<quantize_v2_zero_centered, xpu>::Launch(
              s,
              outputs[0].Size(),
              outputs[0].dptr<int8_t>(),
              inputs[0].dptr<SrcDType>(),
              param.min_calib_range.value(),
              param.max_calib_range.value(),
              MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()),
              req[0]);
        }
        const float real_range = MaxAbs(param.min_calib_range.value(), param.max_calib_range.value());
        AssignQuantizedRangeOutput<xpu>(s, outputs[1], -real_range, req[1]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[2], real_range, req[2]);
      } else {
        LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
      }
    } else {  // model is not calibrated
      if constexpr (!std::is_same<SrcDType, float>::value) {
        LOG(FATAL) << "Uncalibrated native quantize_v2 only supports float32 inputs";
      } else {
        mxnet::TShape src_shape, dst_shape;
        const size_t actual_float_size = sizeof(float);
        const size_t temp_reduce_size  = ConfigReduce<xpu, SrcDType>(
            s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        Tensor<xpu, 1, char> temp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(2 * actual_float_size + temp_reduce_size), s);
        const int dev_id = ctx.run_ctx.ctx.dev_id;
        TBlob in_min_t(
            reinterpret_cast<SrcDType*>(temp_space.dptr_), Shape1(1), xpu::kDevMask, dev_id);
        TBlob in_max_t(
            reinterpret_cast<SrcDType*>(temp_space.dptr_) + 1, Shape1(1), xpu::kDevMask, dev_id);
        Tensor<xpu, 1, char> workspace(
            temp_space.dptr_ + 2 * actual_float_size, Shape1(temp_reduce_size), s);
#if !defined(__CUDACC__)
        broadcast::Reduce<red::minimum, 2, SrcDType, mshadow::op::identity>(
            s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
            s, in_max_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
#else
        broadcast::RTCReduce(ctx,
                             in_min_t.reshape(dst_shape),
                             kWriteTo,
                             workspace,
                             inputs[0].reshape(src_shape),
                             "red::minimum{}",
                             2,
                             "identity");
        broadcast::RTCReduce(ctx,
                             in_max_t.reshape(dst_shape),
                             kWriteTo,
                             workspace,
                             inputs[0].reshape(src_shape),
                             "red::maximum{}",
                             2,
                             "identity");
#endif
        if (out_type == mshadow::kUint8) {
          if (req[0] != kNullOp) {
            Kernel<quantize_v2_unsigned, xpu>::Launch(s,
                                                      outputs[0].Size(),
                                                      outputs[0].dptr<uint8_t>(),
                                                      inputs[0].dptr<SrcDType>(),
                                                      in_min_t.dptr<float>(),
                                                      in_max_t.dptr<float>(),
                                                      MinValue<uint8_t>(),
                                                      MaxValue<uint8_t>(),
                                                      req[0]);
          }
          AssignQuantizedRangeOutput<xpu>(s, outputs[1], in_min_t, req[1]);
          AssignQuantizedRangeOutput<xpu>(s, outputs[2], in_max_t, req[2]);
        } else if (out_type == mshadow::kInt8) {  // zero-centered quantization
          if (req[0] != kNullOp) {
            Kernel<quantize_v2_zero_centered, xpu>::Launch(
                s,
                outputs[0].Size(),
                outputs[0].dptr<int8_t>(),
                inputs[0].dptr<SrcDType>(),
                in_min_t.dptr<float>(),
                in_max_t.dptr<float>(),
                MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()),
                req[0]);
          }
          AssignQuantizedZeroCenteredRangeOutput<xpu>(
              s, outputs[1], outputs[2], in_min_t, in_max_t, req[1], req[2]);
        } else {
          LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
        }
      }
    }
  }

  nnvm::NodeAttrs attrs_;
};

template <typename xpu>
static void QuantizeV2Forward(const OpStatePtr& state_ptr,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  auto& op = state_ptr.get_state<QuantizeV2Operator<xpu>>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_
