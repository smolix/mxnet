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
 * \file dnnl_quantized_conv.cc
 * \brief
 * \author Wenting Jiang, Xinyu Chen
 */

#if MXNET_USE_ONEDNN == 1
#include <cmath>

#include "operator/elemwise_op_common.h"
#include "operator/nn/convolution-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_convolution-inl.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

static void DNNLQuantizedConvForward(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<NDArray>& in_data,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  ConvolutionParam param = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_EQ(param.num_group, 1U) << "quantized_conv only supports num_group=1 for now";

  const size_t num_inputs = param.no_bias ? 2 : 3;
  NDArray data            = in_data[conv::kData];
  NDArray weight          = in_data[conv::kWeight];
  bool reordered          = false;
  if (data.IsDNNLData()) {
    data      = data.Reorder2Default();
    reordered = true;
  }
  if (weight.IsDNNLData()) {
    weight    = weight.Reorder2Default();
    reordered = true;
  }
  NDArray bias;
  if (!param.no_bias) {
    bias = in_data[conv::kBias];
    if (bias.IsDNNLData()) {
      bias      = bias.Reorder2Default();
      reordered = true;
    }
  }
  if (reordered) {
    DNNLStream::Get()->Submit();
  }

  CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8)
      << "QuantizedConv expects int8 or uint8 data";
  CHECK_EQ(weight.dtype(), mshadow::kInt8) << "QuantizedConv expects int8 weight";

  const float min_data   = in_data[num_inputs].data().dptr<float>()[0];
  const float max_data   = in_data[num_inputs + 1].data().dptr<float>()[0];
  const float min_weight = in_data[num_inputs + 2].data().dptr<float>()[0];
  const float max_weight = in_data[num_inputs + 3].data().dptr<float>()[0];
  const bool data_is_int8 = data.dtype() == mshadow::kInt8;
  if (!data_is_int8) {
    CHECK_GT(max_data, min_data) << "uint8 quantized conv expects max_data > min_data";
  }
  const float data_scale =
      data_is_int8 ? kInt8Range / MaxAbs(min_data, max_data) : kUint8Range / (max_data - min_data);
  const float weight_scale = kInt8Range / MaxAbs(min_weight, max_weight);
  const int32_t data_zero_point =
      data_is_int8 ? 0 : static_cast<int32_t>(std::nearbyint(-min_data * data_scale));

  NDArray quantized_bias;
  if (!param.no_bias) {
    const float min_bias = in_data[num_inputs + 4].data().dptr<float>()[0];
    const float max_bias = in_data[num_inputs + 5].data().dptr<float>()[0];
    const float bias_int32_rescale =
        data_scale * weight_scale * MaxAbs(min_bias, max_bias) / kInt8Range;
    quantized_bias = NDArray(bias.storage_type(), bias.shape(), bias.ctx(), true, mshadow::kInt32);
    int8_t* bias_ptr            = bias.data().dptr<int8_t>();
    int32_t* quantized_bias_ptr = quantized_bias.data().dptr<int32_t>();
    const size_t bias_size      = bias.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
      quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
    }
  }

  float min_output = 0.0f;
  float max_output = 0.0f;
  if (data_is_int8) {
    QuantizationRangeForMultiplication<int8_t, int8_t, int32_t>(
        min_data, max_data, min_weight, max_weight, &min_output, &max_output, true);
  } else {
    const float data_float_for_one_quant_level   = (max_data - min_data) / kUint8Range;
    const float weight_float_for_one_quant_level = MaxAbs(min_weight, max_weight) / kInt8Range;
    const float output_range =
        data_float_for_one_quant_level * weight_float_for_one_quant_level * kInt32Range;
    min_output = -output_range;
    max_output = output_range;
  }
  AssignQuantizedRangeOutput(out_data[1].data().dptr<float>(),
                             &min_output,
                             req[1],
                             "quantized_conv");
  AssignQuantizedRangeOutput(out_data[2].data().dptr<float>(),
                             &max_output,
                             req[2],
                             "quantized_conv");

  const auto& dshape = data.shape();
  const auto& wshape = weight.shape();
  const auto& oshape = out_data[conv::kOut].shape();
  const int ndim     = param.kernel.ndim();
  CHECK(ndim == 2 || ndim == 3) << "oneDNN quantized_conv only supports 2D and 3D convolution";
  CHECK_EQ(dshape.ndim(), static_cast<uint32_t>(ndim + 2));
  CHECK_EQ(wshape.ndim(), static_cast<uint32_t>(ndim + 2));

  if (req[conv::kOut] == kNullOp) {
    return;
  }

  auto& out_arr = const_cast<NDArray&>(out_data[conv::kOut]);
  out_arr.InvalidateDNNLData();
  int32_t* out_ptr          = out_arr.data().dptr<int32_t>();
  const int8_t* data_s8     = data_is_int8 ? data.data().dptr<int8_t>() : nullptr;
  const uint8_t* data_u8    = data_is_int8 ? nullptr : data.data().dptr<uint8_t>();
  const int8_t* weight_ptr  = weight.data().dptr<int8_t>();
  const int32_t* bias_ptr   = param.no_bias ? nullptr : quantized_bias.data().dptr<int32_t>();
  [[maybe_unused]] const int omp_threads     = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const index_t batch       = dshape[0];
  const index_t in_channels = dshape[1];
  const index_t out_channels = wshape[0];
  CHECK_EQ(wshape[1], in_channels);

  const index_t out_spatial_size = oshape.ProdShape(2, oshape.ndim());
  if (req[conv::kOut] == kWriteTo || req[conv::kOut] == kWriteInplace ||
      req[conv::kOut] == kAddTo) {
#pragma omp parallel for num_threads(omp_threads)
    for (index_t linear = 0; linear < batch * out_channels * out_spatial_size; ++linear) {
      index_t tmp       = linear;
      const index_t osp2 = (ndim == 3) ? tmp % oshape[4] : 0;
      if (ndim == 3) tmp /= oshape[4];
      const index_t osp1 = tmp % oshape[3];
      tmp /= oshape[3];
      const index_t osp0 = tmp % oshape[2];
      tmp /= oshape[2];
      const index_t oc = tmp % out_channels;
      const index_t n  = tmp / out_channels;
      int32_t acc      = bias_ptr == nullptr ? 0 : bias_ptr[oc];

      for (index_t ic = 0; ic < in_channels; ++ic) {
        for (index_t k0 = 0; k0 < wshape[2]; ++k0) {
          const index_t in0 = osp0 * param.stride[0] + k0 * param.dilate[0] - param.pad[0];
          if (in0 < 0 || in0 >= dshape[2]) continue;
          for (index_t k1 = 0; k1 < wshape[3]; ++k1) {
            const index_t in1 = osp1 * param.stride[1] + k1 * param.dilate[1] - param.pad[1];
            if (in1 < 0 || in1 >= dshape[3]) continue;
            if (ndim == 2) {
              const index_t data_idx = ((n * in_channels + ic) * dshape[2] + in0) * dshape[3] + in1;
              const index_t weight_idx =
                  ((oc * in_channels + ic) * wshape[2] + k0) * wshape[3] + k1;
              const int32_t data_val =
                  data_is_int8 ? static_cast<int32_t>(data_s8[data_idx]) :
                                 static_cast<int32_t>(data_u8[data_idx]) - data_zero_point;
              acc += data_val * static_cast<int32_t>(weight_ptr[weight_idx]);
            } else {
              for (index_t k2 = 0; k2 < wshape[4]; ++k2) {
                const index_t in2 = osp2 * param.stride[2] + k2 * param.dilate[2] - param.pad[2];
                if (in2 < 0 || in2 >= dshape[4]) continue;
                const index_t data_idx =
                    (((n * in_channels + ic) * dshape[2] + in0) * dshape[3] + in1) * dshape[4] +
                    in2;
                const index_t weight_idx =
                    (((oc * in_channels + ic) * wshape[2] + k0) * wshape[3] + k1) * wshape[4] +
                    k2;
                const int32_t data_val =
                    data_is_int8 ? static_cast<int32_t>(data_s8[data_idx]) :
                                   static_cast<int32_t>(data_u8[data_idx]) - data_zero_point;
                acc += data_val * static_cast<int32_t>(weight_ptr[weight_idx]);
              }
            }
          }
        }
      }
      if (req[conv::kOut] == kAddTo) {
        out_ptr[linear] += acc;
      } else {
        out_ptr[linear] = acc;
      }
    }
  }
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedConvForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
