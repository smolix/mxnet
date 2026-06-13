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
 * \file quantize_asym-inl.h
 * \brief implementation of asymmetric quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mshadow/tensor.h>
#include <mxnet/operator_util.h>
#include <vector>

#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "./quantized_range_utils.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct QuantizeAsymParam : public dmlc::Parameter<QuantizeAsymParam> {
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;

  DMLC_DECLARE_PARAMETER(QuantizeAsymParam) {
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32. If "
            "present, it will be used to "
            "quantize the fp32 data.");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32. If "
            "present, it will be used to "
            "quantize the fp32 data.");
  }
};

// quantize float to uint8_t
struct quantize_asymmetric {
  // Host-constant scale/shift overload (calibrated / int8 / uint8 paths).
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  float*,
                                  float*,
                                  const SrcDType* in,
                                  const float scale,
                                  const float shift,
                                  const OpReqType req) {
    const float rounded = in[i] * scale + shift + 0.5f;
    const DstDType quantized =
        static_cast<DstDType>(Min(Max(rounded, 0.0f),
                                  static_cast<float>(mshadow::red::limits::MaxValue<DstDType>())));
    KERNEL_ASSIGN(out[i], req, quantized);
  }

  // Device-resident scale/shift overload (uncalibrated path). The scale and
  // shift live in device memory so they are dereferenced inside the kernel
  // instead of on the host, which keeps the path correct on GPU.
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  DstDType* out,
                                  const SrcDType* in,
                                  const float* scale,
                                  const float* shift,
                                  const OpReqType req) {
    const float rounded = in[i] * scale[0] + shift[0] + 0.5f;
    const DstDType quantized =
        static_cast<DstDType>(Min(Max(rounded, 0.0f),
                                  static_cast<float>(mshadow::red::limits::MaxValue<DstDType>())));
    KERNEL_ASSIGN(out[i], req, quantized);
  }
};

// Derive asymmetric scale/shift from a device-resident [min, max] range. Runs
// as a single-thread kernel so the computation stays on the compute device and
// never dereferences device pointers on the host.
struct quantize_asym_scale_shift {
  MSHADOW_XINLINE static void Map(int,
                                  float* scale_out,
                                  float* shift_out,
                                  const float* in_min,
                                  const float* in_max,
                                  const float quantized_max) {
    const float scale = quantized_max / (in_max[0] - in_min[0]);
    scale_out[0]      = scale;
    shift_out[0]      = quantized_max - in_max[0] * scale;
  }
};

template <typename xpu>
class QuantizeAsymOp {
 public:
  explicit QuantizeAsymOp(const nnvm::NodeAttrs& attrs) : attrs_(attrs) {}

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    using mshadow::red::limits::MaxValue;
    using mshadow::red::limits::MinValue;

    CHECK_EQ(outputs[0].type_flag_, mshadow::kUint8)
        << "Asymmetric quantization only supports uint8 outputs.";
    mshadow::Stream<xpu>* s    = ctx.get_stream<xpu>();
    const int input_data_dtype = inputs[0].type_flag_;
    if (input_data_dtype == mshadow::kUint8) {
      AssignQuantizedRangeOutput<xpu>(s, outputs[1], 1.f, req[1]);
      AssignQuantizedRangeOutput<xpu>(s, outputs[2], 0.f, req[2]);
      UnaryOp::IdentityCompute<xpu>(attrs_, ctx, {inputs[0]}, req, outputs);
    } else if (input_data_dtype == mshadow::kInt8) {
      const float scale = 1;
      const float shift = 128;
      Kernel<quantize_asymmetric, xpu>::Launch(s,
                                               outputs[0].Size(),
                                               outputs[0].dptr<uint8_t>(),
                                               outputs[1].dptr<float>(),
                                               outputs[2].dptr<float>(),
                                               inputs[0].dptr<int8_t>(),
                                               scale,
                                               shift,
                                               req[0]);
      AssignQuantizedRangeOutput<xpu>(s, outputs[1], scale, req[1]);
      AssignQuantizedRangeOutput<xpu>(s, outputs[2], shift, req[2]);
    } else if (input_data_dtype == mshadow::kFloat32) {
      const QuantizeAsymParam& param = nnvm::get<QuantizeAsymParam>(attrs_.parsed);
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        const float scale =
            MaxValue<uint8_t>() / (param.max_calib_range.value() - param.min_calib_range.value());
        const float shift = MaxValue<uint8_t>() - param.max_calib_range.value() * scale;
        Kernel<quantize_asymmetric, xpu>::Launch(s,
                                                 outputs[0].Size(),
                                                 outputs[0].dptr<uint8_t>(),
                                                 outputs[1].dptr<float>(),
                                                 outputs[2].dptr<float>(),
                                                 inputs[0].dptr<float>(),
                                                 scale,
                                                 shift,
                                                 req[0]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[1], scale, req[1]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[2], shift, req[2]);
      } else {
        mxnet::TShape src_shape, dst_shape;
        const size_t float_bytes      = sizeof(float);
        const size_t temp_reduce_size = ConfigReduce<xpu, float>(
            s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
        // Scratch layout: [in_min, in_max, scale, shift, reduce-workspace].
        Tensor<xpu, 1, char> temp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(4 * float_bytes + temp_reduce_size), s);
        const int dev_id     = ctx.run_ctx.ctx.dev_id;
        float* const scratch = reinterpret_cast<float*>(temp_space.dptr_);
        TBlob in_min_t(scratch, Shape1(1), xpu::kDevMask, dev_id);
        TBlob in_max_t(scratch + 1, Shape1(1), xpu::kDevMask, dev_id);
        TBlob scale_t(scratch + 2, Shape1(1), xpu::kDevMask, dev_id);
        TBlob shift_t(scratch + 3, Shape1(1), xpu::kDevMask, dev_id);
        Tensor<xpu, 1, char> workspace(
            temp_space.dptr_ + 4 * float_bytes, Shape1(temp_reduce_size), s);
#if !defined(__CUDACC__)
        broadcast::Reduce<red::minimum, 2, float, mshadow::op::identity>(
            s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
        broadcast::Reduce<red::maximum, 2, float, mshadow::op::identity>(
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
        // Derive scale/shift on the compute device from the reduced [min, max]
        // range, then quantize using the device-resident scale/shift. This keeps
        // the uncalibrated path correct on GPU, where the reduced range lives in
        // device memory and must not be dereferenced on the host.
        Kernel<quantize_asym_scale_shift, xpu>::Launch(s,
                                                       1,
                                                       scale_t.dptr<float>(),
                                                       shift_t.dptr<float>(),
                                                       in_min_t.dptr<float>(),
                                                       in_max_t.dptr<float>(),
                                                       static_cast<float>(MaxValue<uint8_t>()));
        Kernel<quantize_asymmetric, xpu>::Launch(s,
                                                 outputs[0].Size(),
                                                 outputs[0].dptr<uint8_t>(),
                                                 inputs[0].dptr<float>(),
                                                 scale_t.dptr<float>(),
                                                 shift_t.dptr<float>(),
                                                 req[0]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[1], scale_t, req[1]);
        AssignQuantizedRangeOutput<xpu>(s, outputs[2], shift_t, req[2]);
      }
    } else {
      LOG(FATAL) << "Asymmetric quantizaiton only supports int8, uint8 and "
                    "float inputs";
    }
  }

 private:
  nnvm::NodeAttrs attrs_;
};

template <typename xpu>
void QuantizeAsymForward(const OpStatePtr& state_ptr,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  QuantizeAsymOp<xpu>& op = state_ptr.get_state<QuantizeAsymOp<xpu>>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_ASYM_INL_H_
