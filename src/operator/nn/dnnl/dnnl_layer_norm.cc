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
 * \file dnnl_layer_norm.cc
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_layer_norm-inl.h"

#include <algorithm>
#include <cmath>

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v3/dev_guide_layer_normalization.html
bool SupportDNNLLayerNorm(const LayerNormParam& param, const std::vector<NDArray>& inputs) {
  if (!SupportDNNLAArch64JITPrimitives()) return false;

  const mxnet::TShape& shape = inputs[layernorm::kData].shape();

  // Native implementation (which can be found in function LayerNormCPU) is faster than oneDNN's one
  // for small tensors. Below is the heuristic based on measurements on clx machine deciding whether
  // the shape is better for oneDNN or native implementation.
  auto ShapeBetterForDNNL = [](const mxnet::TShape& shape) {
    constexpr size_t shapeLimit = 1024;
    return shape.Size() / shape[0] >= shapeLimit && shape[0] >= shapeLimit;
  };

  return (ShapeBetterForDNNL(shape) && GetRealAxis(param.axis, shape.ndim()) == shape.ndim() - 1) &&
         SupportDNNL<2, 5, DNNLTypeMode::FloatTypes>(inputs[layernorm::kData]) &&
         inputs[layernorm::kGamma].dtype() == mshadow::kFloat32 &&
         inputs[layernorm::kBeta].dtype() == mshadow::kFloat32;
}

void DNNLLayerNormForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  const auto& fwd             = DNNLLayerNormFwd::GetCached(param, ctx, inputs[layernorm::kData]);
  fwd.Execute(param, ctx, inputs, req, outputs);
}

DNNLLayerNormFwd& DNNLLayerNormFwd::GetCached(const LayerNormParam& param,
                                              const OpContext& ctx,
                                              const NDArray& data) {
  using layernorm_fwd_map = std::unordered_map<LayerNormSignature, DNNLLayerNormFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local layernorm_fwd_map layer_norm_fwds;
#else
  static MX_THREAD_LOCAL layernorm_fwd_map layer_norm_fwds;
#endif

  LayerNormSignature key(param);
  key.AddSign(data);

  auto it = layer_norm_fwds.find(key);
  if (it == layer_norm_fwds.end()) {
    DNNLLayerNormFwd fwd(param, data);
    it = AddToCache(&layer_norm_fwds, key, fwd);
  }
  return it->second;
}

DNNLLayerNormFwd::DNNLLayerNormFwd(const LayerNormParam& param, const NDArray& data) {
  const dnnl::memory::desc data_md = data.GetDNNLData()->get_desc();
  fwd_pd                           = CreatePrimitiveDesc(param, data_md);
  fwd                              = std::make_shared<layernorm_fwd_t>(*fwd_pd);
}

std::shared_ptr<layernorm_fwd_pd_t> DNNLLayerNormFwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const dnnl::memory::desc& src_md) {
  // v3: ::desc removed; use_scale_shift split into use_scale + use_shift.
  //     primitive_desc(engine, prop, src_md, dst_md, epsilon, flags, attr).
  dnnl::engine& engine = CpuEngine::Get()->get_engine();
  const auto flags     = dnnl::normalization_flags::use_scale |
                         dnnl::normalization_flags::use_shift;
  return std::make_shared<layernorm_fwd_pd_t>(engine, dnnl::prop_kind::forward_training,
                                              src_md, src_md, param.eps, flags);
}

inline dnnl::memory::desc GetMeanVarDesc(const dnnl::memory::data_type& dtype,
                                         const mxnet::TShape& _shape) {
  const auto ndim = _shape.ndim();

  dnnl::memory::dims shape(ndim, 1);
  for (int i = 0; i < ndim; ++i) {
    shape[i] = _shape[i];
  }
  switch (ndim) {
    case 1:
      return dnnl::memory::desc{shape, dtype, dnnl::memory::format_tag::a};
    case 2:
      return dnnl::memory::desc{shape, dtype, dnnl::memory::format_tag::ab};
    case 3:
      return dnnl::memory::desc{shape, dtype, dnnl::memory::format_tag::abc};
    case 4:
      return dnnl::memory::desc{shape, dtype, dnnl::memory::format_tag::abcd};
    default:
      LOG(FATAL) << "Unsupported LayerNorm stats rank for oneDNN: " << ndim;
  }
  return dnnl::memory::desc();
}

// v3: SCALE_SHIFT was split into SCALE + SHIFT, each a 1-D tensor of length C.
inline dnnl::memory GetGammaOrBetaMem(const NDArray& tensor) {
  NDArray tensor_buffer = tensor;
  if (tensor_buffer.IsDNNLData()) {
    tensor_buffer = tensor_buffer.Reorder2Default();
    DNNLStream::Get()->Submit();
  }
  const dnnl::memory::desc md(dnnl::memory::dims{tensor.shape()[0]},
                              get_dnnl_type(tensor.dtype()),
                              dnnl::memory::format_tag::a);
  auto mem = dnnl::memory(md, CpuEngine::Get()->get_engine());
  memcpy(mem.get_data_handle(), tensor_buffer.data().dptr_, md.get_size());
  return mem;
}

inline void CommitStatOutput(const NDArray& output,
                             const dnnl::memory& value,
                             const OpReqType req,
                             const char* name) {
  if (req == kNullOp) {
    return;
  }
  CHECK_EQ(output.dtype(), mshadow::kFloat32);

  const float* value_data = static_cast<const float*>(value.get_data_handle());
  float* output_data      = output.data().dptr<float>();
  const size_t size       = output.shape().Size();
  if (req == kAddTo) {
    for (size_t i = 0; i < size; ++i) {
      output_data[i] += value_data[i];
    }
  } else {
    std::copy(value_data, value_data + size, output_data);
  }
}

inline void CommitStdOutput(const NDArray& output,
                            const dnnl::memory& variance,
                            const OpReqType req,
                            float eps) {
  if (req == kNullOp) {
    return;
  }
  CHECK_EQ(output.dtype(), mshadow::kFloat32);

  const float* variance_data = static_cast<const float*>(variance.get_data_handle());
  float* output_data         = output.data().dptr<float>();
  const size_t size          = output.shape().Size();
  for (size_t i = 0; i < size; ++i) {
    const float std = std::sqrt(variance_data[i] + eps);
    if (req == kAddTo) {
      output_data[i] += std;
    } else {
      output_data[i] = std;
    }
  }
}

inline dnnl::memory CopyStatToDNNL(const NDArray& input,
                                   const dnnl::memory::desc& desc,
                                   const char* name) {
  CHECK_EQ(input.dtype(), mshadow::kFloat32);
  dnnl::memory mem(desc, CpuEngine::Get()->get_engine());
  std::copy(input.data().dptr<float>(),
            input.data().dptr<float>() + input.shape().Size(),
            static_cast<float*>(mem.get_data_handle()));
  return mem;
}

inline dnnl::memory StdToVariance(const NDArray& std, const dnnl::memory::desc& desc, float eps) {
  CHECK_EQ(std.dtype(), mshadow::kFloat32);

  dnnl::memory variance(desc, CpuEngine::Get()->get_engine());
  const float* std_data = std.data().dptr<float>();
  float* variance_data  = static_cast<float*>(variance.get_data_handle());
  const size_t size     = std.shape().Size();
  for (size_t i = 0; i < size; ++i) {
    variance_data[i] = std::max(0.0f, std_data[i] * std_data[i] - eps);
  }
  return variance;
}

inline void CommitScaleOrShiftGrad(const NDArray& output,
                                   const dnnl::memory& grad,
                                   const OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  CHECK_EQ(output.dtype(), mshadow::kFloat32);

  const float* grad_data = static_cast<const float*>(grad.get_data_handle());
  float* output_data     = output.data().dptr<float>();
  const size_t size      = output.shape().Size();
  if (req == kAddTo) {
    for (size_t i = 0; i < size; ++i) {
      output_data[i] += grad_data[i];
    }
  } else {
    std::copy(grad_data, grad_data + size, output_data);
  }
}

void DNNLLayerNormFwd::Execute(const LayerNormParam& param,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) const {
  auto mean_var_md = GetMeanVarDesc(get_dnnl_type(outputs[layernorm::kMean].dtype()),
                                    outputs[layernorm::kMean].shape());
  const size_t stat_size = outputs[layernorm::kMean].shape().Size();
  std::vector<float> mean_storage(stat_size);
  std::vector<float> variance_storage(stat_size);
  dnnl::memory mean_mem(mean_var_md, CpuEngine::Get()->get_engine(), mean_storage.data());
  dnnl::memory variance_mem(mean_var_md, CpuEngine::Get()->get_engine(), variance_storage.data());

  auto output_mem =
      CreateDNNLMem(outputs[layernorm::kOut], fwd_pd->dst_desc(), req[layernorm::kOut]);
  // v3: separate scale + shift args.
  auto scale_mem  = GetGammaOrBetaMem(inputs[layernorm::kGamma]);
  auto shift_mem  = GetGammaOrBetaMem(inputs[layernorm::kBeta]);

  dnnl_args_map_t args = {{DNNL_ARG_SRC, *inputs[layernorm::kData].GetDNNLData()},
                          {DNNL_ARG_DST, *output_mem.second},
                          {DNNL_ARG_MEAN, mean_mem},
                          {DNNL_ARG_VARIANCE, variance_mem},
                          {DNNL_ARG_SCALE, scale_mem},
                          {DNNL_ARG_SHIFT, shift_mem}};

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[layernorm::kOut], output_mem);
  DNNLStream::Get()->Submit();
  CommitStatOutput(outputs[layernorm::kMean], mean_mem, req[layernorm::kMean], "LayerNorm mean");
  CommitStdOutput(outputs[layernorm::kStd], variance_mem, req[layernorm::kStd], param.eps);
}

DNNLLayerNormBwd::DNNLLayerNormBwd(const LayerNormParam& param,
                                   const std::vector<NDArray>& inputs,
                                   const dnnl::memory::desc& data_md,
                                   const dnnl::memory::desc& diff_md)
    : fwd_pd(DNNLLayerNormFwd::CreatePrimitiveDesc(param, data_md)),
      bwd_pd(CreatePrimitiveDesc(param, data_md, diff_md, *fwd_pd)) {
  bwd = std::make_shared<layernorm_bwd_t>(*bwd_pd);
}

std::shared_ptr<layernorm_bwd_pd_t> DNNLLayerNormBwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const dnnl::memory::desc& data_md,
    const dnnl::memory::desc& diff_md,
    const layernorm_fwd_pd_t& layernorm_fwd_pd) {
  // v3: ::desc removed; primitive_desc(engine, prop_kind, diff_src_md,
  //     diff_dst_md, src_md, epsilon, flags, hint_fwd_pd, attr).
  //     use_scale_shift was split into use_scale + use_shift.
  dnnl::engine& engine = CpuEngine::Get()->get_engine();
  const auto flags     = dnnl::normalization_flags::use_scale |
                         dnnl::normalization_flags::use_shift;
  return std::make_shared<layernorm_bwd_pd_t>(engine, dnnl::prop_kind::backward,
                                              diff_md, diff_md, data_md, param.eps,
                                              flags, layernorm_fwd_pd);
}

void DNNLLayerNormBwd::Execute(const LayerNormParam& param,
                               const std::vector<NDArray>& inputs,
                               const std::vector<NDArray>& outputs,
                               const std::vector<OpReqType>& req) const {
  // v3: scale/shift handled as separate 1-D tensors.
  auto scale_mem = GetGammaOrBetaMem(inputs[layernorm::kBwdGamma]);
  auto shift_mem = GetGammaOrBetaMem(inputs[layernorm::kBwdBeta]);

  // Allocate diff_scale / diff_shift sized to the C dimension.
  // AUDIT-F1: oneDNN v3 layer_normalization_backward exposes only a single
  // diff_weights_desc() (it's shared between scale and shift; no diff_weights_desc(i)).
  // If a future v3.x splits them, switch to per-arg descriptors here.
  auto diff_weights_md = bwd_pd->diff_weights_desc();
  dnnl::memory diff_scale_mem(diff_weights_md, CpuEngine::Get()->get_engine());
  dnnl::memory diff_shift_mem(diff_weights_md, CpuEngine::Get()->get_engine());
  dnnl_output_t diff_src_mem = CreateDNNLMem(
      outputs[layernorm::kBwdDataGrad], bwd_pd->diff_src_desc(), req[layernorm::kBwdDataGrad]);
  auto mean_var_md = GetMeanVarDesc(get_dnnl_type(inputs[layernorm::kBwdMean].dtype()),
                                    inputs[layernorm::kBwdMean].shape());
  const size_t stat_size = inputs[layernorm::kBwdMean].shape().Size();
  std::vector<float> mean_storage(stat_size);
  std::vector<float> variance_storage(stat_size);
  std::copy(inputs[layernorm::kBwdMean].data().dptr<float>(),
            inputs[layernorm::kBwdMean].data().dptr<float>() + stat_size,
            mean_storage.data());
  const float* std_data = inputs[layernorm::kBwdStd].data().dptr<float>();
  for (size_t i = 0; i < stat_size; ++i) {
    variance_storage[i] = std::max(0.0f, std_data[i] * std_data[i] - param.eps);
  }
  dnnl::memory mean_mem(mean_var_md, CpuEngine::Get()->get_engine(), mean_storage.data());
  dnnl::memory variance_mem(mean_var_md, CpuEngine::Get()->get_engine(), variance_storage.data());
  dnnl_args_map_t args = {{DNNL_ARG_DIFF_DST, *inputs[layernorm::kBwdOutGrad].GetDNNLData()},
                          {DNNL_ARG_SRC, *inputs[layernorm::kBwdData].GetDNNLData()},
                          {DNNL_ARG_SCALE, scale_mem},
                          {DNNL_ARG_SHIFT, shift_mem},
                          {DNNL_ARG_MEAN, mean_mem},
                          {DNNL_ARG_VARIANCE, variance_mem},
                          {DNNL_ARG_DIFF_SRC, *diff_src_mem.second},
                          {DNNL_ARG_DIFF_SCALE, diff_scale_mem},
                          {DNNL_ARG_DIFF_SHIFT, diff_shift_mem}};
  DNNLStream::Get()->RegisterPrimArgs(*bwd, args);
  CommitOutput(outputs[layernorm::kBwdDataGrad], diff_src_mem);
  DNNLStream::Get()->Submit();
  CommitScaleOrShiftGrad(outputs[layernorm::kBwdGammaGrad],
                         diff_scale_mem,
                         req[layernorm::kBwdGammaGrad]);
  CommitScaleOrShiftGrad(outputs[layernorm::kBwdBetaGrad],
                         diff_shift_mem,
                         req[layernorm::kBwdBetaGrad]);
}

DNNLLayerNormBwd& DNNLLayerNormBwd::GetCached(const LayerNormParam& param,
                                              const std::vector<NDArray>& inputs) {
  using layernorm_bwd_map = std::unordered_map<LayerNormSignature, DNNLLayerNormBwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local layernorm_bwd_map layer_norm_bwds;
#else
  static MX_THREAD_LOCAL layernorm_bwd_map layer_norm_bwds;
#endif
  LayerNormSignature key(param);
  key.AddSign(inputs[layernorm::kBwdOutGrad]);
  key.AddSign(inputs[layernorm::kBwdData]);
  key.AddSign(inputs[layernorm::kBwdGamma]);
  key.AddSign(inputs[layernorm::kBwdMean]);
  key.AddSign(inputs[layernorm::kBwdStd]);
  key.AddSign(inputs[layernorm::kBwdBeta]);

  auto it = layer_norm_bwds.find(key);
  if (it == layer_norm_bwds.end()) {
    const dnnl::memory::desc data_md = inputs[layernorm::kBwdData].GetDNNLData()->get_desc();
    const dnnl::memory::desc diff_md = inputs[layernorm::kBwdOutGrad].GetDNNLData()->get_desc();
    DNNLLayerNormBwd bwd(param, inputs, data_md, diff_md);
    it = AddToCache(&layer_norm_bwds, key, bwd);
  }
  return it->second;
}

void DNNLLayerNormBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  DNNLLayerNormBwd& bwd       = DNNLLayerNormBwd::GetCached(param, inputs);
  bwd.Execute(param, inputs, outputs, req);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
