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
 * \file dnnl_quantized_batch_norm.cc
 * \author Yixin Bao
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_batch_norm-inl.h"
#include "operator/mxnet_op.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

struct QuantizedBatchNormUInt8AffineKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int8_t* out,
                                  const uint8_t* data,
                                  const float* gamma,
                                  const float* beta,
                                  const float* mean,
                                  const float* var,
                                  float eps,
                                  float min_data,
                                  float max_data,
                                  float max_abs_output,
                                  index_t inner_size,
                                  int channel_count,
                                  OpReqType req) {
    const int channel = (i / inner_size) % channel_count;
    const float real =
        static_cast<float>(data[i]) * (max_data - min_data) / kUint8Range + min_data;
    const float invstd = 1.0f / sqrtf(var[channel] + eps);
    const float normalized = (real - mean[channel]) * gamma[channel] * invstd + beta[channel];
    const float scaled =
        max_abs_output > 0.0f ? floorf(std::min(std::abs(normalized) * kInt8Range / max_abs_output,
                                                kInt8Range) +
                                       0.5f) :
                                0.0f;
    const int32_t q = static_cast<int32_t>(normalized < 0.0f ? -scaled : scaled);
    KERNEL_ASSIGN(out[i], req, static_cast<int8_t>(q));
  }
};

struct QuantizedBatchNormReluUInt8AffineKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int8_t* out,
                                  const uint8_t* data,
                                  const float* gamma,
                                  const float* beta,
                                  const float* mean,
                                  const float* var,
                                  float eps,
                                  float min_data,
                                  float max_data,
                                  float max_abs_output,
                                  index_t inner_size,
                                  int channel_count,
                                  OpReqType req) {
    const int channel = (i / inner_size) % channel_count;
    const float real =
        static_cast<float>(data[i]) * (max_data - min_data) / kUint8Range + min_data;
    const float invstd = 1.0f / sqrtf(var[channel] + eps);
    const float normalized = (real - mean[channel]) * gamma[channel] * invstd + beta[channel];
    const float relu = normalized > 0.0f ? normalized : 0.0f;
    const float scaled =
        max_abs_output > 0.0f ? floorf(std::min(relu * kInt8Range / max_abs_output,
                                                kInt8Range) +
                                       0.5f) :
                                0.0f;
    KERNEL_ASSIGN(out[i], req, static_cast<int8_t>(scaled));
  }
};

template <bool fuse_relu>
static void DNNLQuantizedBatchNormForward(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& in_data,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(in_data.size(), 7U);
  CHECK_EQ(outputs.size(), 3U);

  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  const NDArray& data         = in_data[quantized_batchnorm::kData];
  auto data_mem               = data.GetDNNLData();

  const size_t channelAxis = static_cast<size_t>(
      param.axis < 0 ? static_cast<int>(data.shape().ndim()) + param.axis : param.axis);
  const int channel_count  = data.shape()[channelAxis];
  const float min_data     = in_data[quantized_batchnorm::kDataMin].data().dptr<float>()[0];
  const float max_data     = in_data[quantized_batchnorm::kDataMax].data().dptr<float>()[0];
  const float max_abs_data = std::max(std::abs(min_data), std::abs(max_data));

  float min_output = 0.0f;
  float max_output = 0.0f;
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    max_output = param.max_calib_range.value();
    min_output = param.min_calib_range.value();
  } else {
    LOG(FATAL) << "min_calib_range or max_calib_range is not available. Quantized BN currently "
                  "don't support calib_mode=None";
  }
  if (req[quantized_batchnorm::kOutMin] != kNullOp) {
    AssignQuantizedRangeOutput(outputs[quantized_batchnorm::kOutMin].data().dptr<float>(),
                               &min_output,
                               req[quantized_batchnorm::kOutMin],
                               "_contrib_quantized_batch_norm");
  }
  if (req[quantized_batchnorm::kOutMax] != kNullOp) {
    AssignQuantizedRangeOutput(outputs[quantized_batchnorm::kOutMax].data().dptr<float>(),
                               &max_output,
                               req[quantized_batchnorm::kOutMax],
                               "_contrib_quantized_batch_norm");
  }
  if (req[quantized_batchnorm::kOut] == kNullOp) {
    return;
  }
  const float max_abs_output = std::max(std::abs(min_output), std::abs(max_output));

  if (data.dtype() == mshadow::kUint8) {
    NDArray default_data = data.Reorder2Default();
    index_t inner_size = 1;
    for (size_t i = channelAxis + 1; i < data.shape().ndim(); ++i) {
      inner_size *= data.shape()[i];
    }
    if (fuse_relu) {
      mxnet_op::Kernel<QuantizedBatchNormReluUInt8AffineKernel, cpu>::Launch(
          ctx.get_stream<cpu>(),
          default_data.shape().Size(),
          outputs[quantized_batchnorm::kOut].data().dptr<int8_t>(),
          default_data.data().dptr<uint8_t>(),
          in_data[quantized_batchnorm::kGamma].data().dptr<float>(),
          in_data[quantized_batchnorm::kBeta].data().dptr<float>(),
          in_data[quantized_batchnorm::kInMovingMean].data().dptr<float>(),
          in_data[quantized_batchnorm::kInMovingVar].data().dptr<float>(),
          param.eps,
          min_data,
          max_data,
          max_abs_output,
          inner_size,
          channel_count,
          req[quantized_batchnorm::kOut]);
    } else {
      mxnet_op::Kernel<QuantizedBatchNormUInt8AffineKernel, cpu>::Launch(
          ctx.get_stream<cpu>(),
          default_data.shape().Size(),
          outputs[quantized_batchnorm::kOut].data().dptr<int8_t>(),
          default_data.data().dptr<uint8_t>(),
          in_data[quantized_batchnorm::kGamma].data().dptr<float>(),
          in_data[quantized_batchnorm::kBeta].data().dptr<float>(),
          in_data[quantized_batchnorm::kInMovingMean].data().dptr<float>(),
          in_data[quantized_batchnorm::kInMovingVar].data().dptr<float>(),
          param.eps,
          min_data,
          max_data,
          max_abs_output,
          inner_size,
          channel_count,
          req[quantized_batchnorm::kOut]);
    }
    return;
  }

  // v3: use_scale_shift split; use_scale + use_shift with separate args.
  dnnl::normalization_flags flags = dnnl::normalization_flags::use_global_stats |
                                    dnnl::normalization_flags::use_scale |
                                    dnnl::normalization_flags::use_shift;
  auto& fwd                     = DNNLBNForward::GetCached(param, ctx, data_mem, fuse_relu, flags);
  const dnnl::memory& scale_mem = fwd.GetScale();
  const dnnl::memory& shift_mem = fwd.GetShift();
  CHECK_EQ(scale_mem.get_desc().get_size(), channel_count * sizeof(float));
  CHECK_EQ(shift_mem.get_desc().get_size(), channel_count * sizeof(float));
  float* scale_buf = reinterpret_cast<float*>(scale_mem.get_data_handle());
  float* shift_buf = reinterpret_cast<float*>(shift_mem.get_data_handle());

  float* gamma_ptr = in_data[quantized_batchnorm::kGamma].data().dptr<float>();
  float* beta_ptr  = in_data[quantized_batchnorm::kBeta].data().dptr<float>();

  const NDArray& moving_mean = in_data[quantized_batchnorm::kInMovingMean];
  const NDArray& moving_var  = in_data[quantized_batchnorm::kInMovingVar];
  float* moving_mean_ptr     = moving_mean.data().dptr<float>();
  float* moving_var_ptr      = moving_var.data().dptr<float>();

  // rescale gamma and beta, to make mean=0 and var=1
  auto rescaled_mean_mem   = TmpMemMgr::Get()->Alloc(moving_mean.GetDNNLData()->get_desc());
  auto rescaled_var_mem    = TmpMemMgr::Get()->Alloc(moving_var.GetDNNLData()->get_desc());
  float* rescaled_mean_ptr = reinterpret_cast<float*>(rescaled_mean_mem->get_data_handle());
  float* rescaled_var_ptr  = reinterpret_cast<float*>(rescaled_var_mem->get_data_handle());

#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int channel = 0; channel < channel_count; ++channel) {
    float invstd       = 1.0 / std::sqrt(moving_var_ptr[channel] + param.eps);
    scale_buf[channel] = gamma_ptr[channel] * invstd * max_abs_data / max_abs_output;
    shift_buf[channel] =
        (beta_ptr[channel] - moving_mean_ptr[channel] * gamma_ptr[channel] * invstd) * kInt8Range /
        max_abs_output;
    rescaled_mean_ptr[channel] = 0.0f;
    rescaled_var_ptr[channel]  = 1.0f;
  }

  const NDArray& out = outputs[batchnorm::kOut];
  auto fwd_dst_desc  = fwd.GetPd().dst_desc();
  auto out_mem       = CreateDNNLMem(out, fwd_dst_desc, req[quantized_batchnorm::kOut]);
  dnnl_args_map_t net_args;
  net_args[DNNL_ARG_SRC]      = *data_mem;
  net_args[DNNL_ARG_SCALE]    = scale_mem;
  net_args[DNNL_ARG_SHIFT]    = shift_mem;
  net_args[DNNL_ARG_DST]      = *out_mem.second;
  net_args[DNNL_ARG_MEAN]     = *rescaled_mean_mem;
  net_args[DNNL_ARG_VARIANCE] = *rescaled_var_mem;

  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out, out_mem);
  DNNLStream::Get()->Submit();
}

inline static bool QuantizedBatchNormStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int>* in_attrs,
                                                 std::vector<int>* out_attrs) {
  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLQuantizedOps(), dispatch_mode, in_attrs, out_attrs);
}

inline static bool QuantizedBatchNormWithReLUStorageType(const nnvm::NodeAttrs& attrs,
                                                         const int dev_mask,
                                                         DispatchMode* dispatch_mode,
                                                         std::vector<int>* in_attrs,
                                                         std::vector<int>* out_attrs) {
  bool dispatched = false;
  if (!dispatched) {
    dispatched = DNNLStorageType(
        attrs, dev_mask, SupportDNNLQuantizedOps(), dispatch_mode, in_attrs, out_attrs);
  }
  return dispatched;
}

NNVM_REGISTER_OP(_contrib_quantized_batch_norm)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedBatchNormStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedBatchNormForward</*fuse_relu*/ false>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true);

NNVM_REGISTER_OP(_contrib_quantized_batch_norm_relu)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedBatchNormWithReLUStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedBatchNormForward</*fuse_relu*/ true>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
