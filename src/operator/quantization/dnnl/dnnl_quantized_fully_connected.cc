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
 * \file dnnl_quantized_fully_connected.cc
 * \brief DNNL Quantized FullyConnected operator
 * \author Ciyong Chen
 */

#if MXNET_USE_ONEDNN == 1
#include <cmath>

#include "operator/nn/dnnl/dnnl_fully_connected-inl.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

void DNNLQuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& in_data,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  FullyConnectedParam param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const size_t num_inputs = param.no_bias ? 2 : 3;

  CHECK_EQ(in_data.size(), static_cast<size_t>(num_inputs * 3));
  CHECK_EQ(out_data.size(), 3U);

  NDArray data   = in_data[fullc::kData];
  NDArray weight = in_data[fullc::kWeight];
  bool reordered = false;
  if (data.IsDNNLData()) {
    data      = data.Reorder2Default();
    reordered = true;
  }
  if (weight.IsDNNLData()) {
    weight    = weight.Reorder2Default();
    reordered = true;
  }

  const float min_data = in_data[num_inputs + quantized_fullc::kDataMin].data().dptr<float>()[0];
  const float max_data = in_data[num_inputs + quantized_fullc::kDataMax].data().dptr<float>()[0];
  const float min_weight =
      in_data[num_inputs + quantized_fullc::kWeightMin].data().dptr<float>()[0];
  const float max_weight =
      in_data[num_inputs + quantized_fullc::kWeightMax].data().dptr<float>()[0];
  float min_output = 0.0f;
  float max_output = 0.0f;

  const bool data_is_int8 = data.dtype() == mshadow::kInt8;
  if (!data_is_int8) {
    CHECK_GT(max_data, min_data) << "uint8 quantized fully connected expects max_data > min_data";
  }
  float data_scale =
      data_is_int8 ? kInt8Range / MaxAbs(min_data, max_data) : kUint8Range / (max_data - min_data);
  float weight_scale = kInt8Range / MaxAbs(min_weight, max_weight);
  const int32_t data_zero_point =
      data_is_int8 ? 0 : static_cast<int32_t>(std::nearbyint(-min_data * data_scale));

  NDArray quantized_bias;
  if (!param.no_bias) {
    NDArray bias   = in_data[fullc::kBias];
    if (bias.IsDNNLData()) {
      bias      = bias.Reorder2Default();
      reordered = true;
    }
    if (reordered) {
      DNNLStream::Get()->Submit();
    }
    float min_bias = in_data[num_inputs + quantized_fullc::kBiasMin].data().dptr<float>()[0];
    float max_bias = in_data[num_inputs + quantized_fullc::kBiasMax].data().dptr<float>()[0];
    float bias_int32_rescale = data_scale * weight_scale * MaxAbs(min_bias, max_bias) / kInt8Range;

    quantized_bias = NDArray(bias.storage_type(), bias.shape(), bias.ctx(), true, mshadow::kInt32);
    int8_t* bias_ptr            = bias.data().dptr<int8_t>();
    int32_t* quantized_bias_ptr = quantized_bias.data().dptr<int32_t>();
    size_t bias_size            = bias.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
      quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
    }
  } else if (reordered) {
    DNNLStream::Get()->Submit();
  }

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
  AssignQuantizedRangeOutput(out_data[quantized_fullc::kOutMin].data().dptr<float>(),
                             &min_output,
                             req[quantized_fullc::kOutMin],
                             "quantized_fully_connected");
  AssignQuantizedRangeOutput(out_data[quantized_fullc::kOutMax].data().dptr<float>(),
                             &max_output,
                             req[quantized_fullc::kOutMax],
                             "quantized_fully_connected");

  const auto& data_shape   = data.shape();
  const auto& weight_shape = weight.shape();
  CHECK_EQ(weight_shape.ndim(), 2U);
  CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8)
      << "QuantizedFullyConnected expects int8 or uint8 data";
  const index_t m = param.flatten ? data_shape[0] : data_shape.ProdShape(0, data_shape.ndim() - 1);
  const index_t k =
      param.flatten ? data_shape.ProdShape(1, data_shape.ndim()) : data_shape[data_shape.ndim() - 1];
  const index_t n = weight_shape[0];
  CHECK_EQ(weight_shape[1], k);

  if (req[fullc::kOut] == kNullOp) {
    return;
  }

  auto& out_arr = const_cast<NDArray&>(out_data[fullc::kOut]);
  out_arr.InvalidateDNNLData();
  int32_t* out_ptr         = out_arr.data().dptr<int32_t>();
  const int8_t* data_s8    = data.dtype() == mshadow::kInt8 ? data.data().dptr<int8_t>() : nullptr;
  const uint8_t* data_u8   = data.dtype() == mshadow::kUint8 ? data.data().dptr<uint8_t>() : nullptr;
  const int8_t* weight_ptr = weight.data().dptr<int8_t>();
  const int32_t* bias_ptr  = param.no_bias ? nullptr : quantized_bias.data().dptr<int32_t>();
  const int omp_threads    = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  if (req[fullc::kOut] == kWriteTo || req[fullc::kOut] == kWriteInplace) {
#pragma omp parallel for num_threads(omp_threads)
    for (index_t row = 0; row < m; ++row) {
      for (index_t col = 0; col < n; ++col) {
        int32_t acc = bias_ptr == nullptr ? 0 : bias_ptr[col];
        for (index_t inner = 0; inner < k; ++inner) {
          if (data_is_int8) {
            acc += static_cast<int32_t>(data_s8[row * k + inner]) *
                   static_cast<int32_t>(weight_ptr[col * k + inner]);
          } else {
            acc += (static_cast<int32_t>(data_u8[row * k + inner]) - data_zero_point) *
                   static_cast<int32_t>(weight_ptr[col * k + inner]);
          }
        }
        out_ptr[row * n + col] = acc;
      }
    }
  } else if (req[fullc::kOut] == kAddTo) {
#pragma omp parallel for num_threads(omp_threads)
    for (index_t row = 0; row < m; ++row) {
      for (index_t col = 0; col < n; ++col) {
        int32_t acc = bias_ptr == nullptr ? 0 : bias_ptr[col];
        for (index_t inner = 0; inner < k; ++inner) {
          if (data_is_int8) {
            acc += static_cast<int32_t>(data_s8[row * k + inner]) *
                   static_cast<int32_t>(weight_ptr[col * k + inner]);
          } else {
            acc += (static_cast<int32_t>(data_u8[row * k + inner]) - data_zero_point) *
                   static_cast<int32_t>(weight_ptr[col * k + inner]);
          }
        }
        out_ptr[row * n + col] += acc;
      }
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
