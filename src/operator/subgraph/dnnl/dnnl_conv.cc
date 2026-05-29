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

#if MXNET_USE_ONEDNN == 1

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "operator/elemwise_op_common.h"
#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_common.h"
#include "dnnl_conv-inl.h"

namespace mxnet {
namespace op {

using red::limits::MaxValue;
using red::limits::MinValue;

static uint32_t SgDNNLConvNumInputs(const NodeAttrs& attrs);

template <typename DType>
static void UpdateConvWeightBias(NDArray* weight,
                                 NDArray* bias,
                                 bool no_bias,
                                 const NDArray& gamma,
                                 const NDArray& beta,
                                 const NDArray& mean,
                                 const NDArray& variance,
                                 const BatchNormParam* param) {
  NDArray update_weight =
      NDArray(weight->storage_type(), weight->shape(), weight->ctx(), true, weight->dtype());
  NDArray update_bias =
      NDArray(beta.storage_type(), beta.shape(), beta.ctx(), true, weight->dtype());
  const DType* weight_ptr  = weight->data().dptr<DType>();
  const DType* bias_ptr    = no_bias ? nullptr : bias->data().dptr<DType>();
  const float* gamma_ptr   = gamma.data().dptr<float>();
  const float* beta_ptr    = beta.data().dptr<float>();
  const float* mean_ptr    = mean.data().dptr<float>();
  const float* var_ptr     = variance.data().dptr<float>();
  DType* update_weight_ptr = update_weight.data().dptr<DType>();
  DType* update_bias_ptr   = update_bias.data().dptr<DType>();
  index_t channel          = static_cast<index_t>(gamma.shape()[0]);
  const auto wshape        = weight->shape();
  size_t offset            = wshape.ProdShape(1, wshape.ndim());
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t c = 0; c < channel; ++c) {
    const DType* p1 = weight_ptr + c * offset;
    DType* p2       = update_weight_ptr + c * offset;
    float alpha     = (param->fix_gamma ? 1.0f : gamma_ptr[c]) / sqrt(var_ptr[c] + param->eps);

    if (bias_ptr)
      update_bias_ptr[c] =
          static_cast<DType>(beta_ptr[c] + alpha * (static_cast<float>(bias_ptr[c]) - mean_ptr[c]));
    else
      update_bias_ptr[c] = static_cast<DType>(beta_ptr[c] - alpha * mean_ptr[c]);

    for (size_t k = 0; k < offset; ++k) {
      p2[k] = static_cast<DType>(static_cast<float>(p1[k]) * alpha);
    }
  }
  *weight = update_weight;
  *bias   = update_bias;
}

static inline size_t GetInSumIndex(const DNNLConvFusionParam& param) {
  if (param.full_conv_param.dnnl_param.dedup_sum) {
    return 0;
  }
  return 2 + (param.full_conv_param.conv_param.no_bias ? 0 : 1) +
         (param.full_conv_param.dnnl_param.with_bn ? 4 : 0);
}

class SgDNNLConvOperator {
 public:
  explicit SgDNNLConvOperator(const nnvm::NodeAttrs& attrs)
      : subgraph_sym_(*attrs.subgraphs[0]), param_(nnvm::get<DNNLConvFusionParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  bool initialized_{false};
  bool inplace_{false};
  bool post_requantize_{false};
  nnvm::Symbol subgraph_sym_;
  DNNLConvFusionParam param_;
  std::shared_ptr<DNNLConvForward> fwd_;
  dnnl_args_map_t args_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  // B4: the conv subgraph never receives bias_min/max as a separate input the
  // way dnnl_fc.cc does — its bias-overflow guard runs inside GetWeightScales()
  // (dnnl_common.h:68-76) on the per-channel int8 bias values directly. The
  // previously-declared cached_bias_min_/max_{0.0f} members were never wired
  // and never read; removed.
  // F10: default-initialize so reuse / re-serialization can't read garbage.
  float cached_data_min_{0.0f};
  float cached_data_max_{0.0f};
  float cached_sum_min_{0.0f};
  float cached_sum_max_{0.0f};
  float cached_output_min_{0.0f};
  float cached_output_max_{0.0f};
  size_t weight_ver_{0};
  size_t bias_ver_{0};
  float data_scale_{0.0f};
  std::vector<float> weight_scales_;
};

static bool SgDNNLConvHasSupportedQATBackwardAct(const DNNLConvFusionParam& param) {
  const auto& full_param = param.full_conv_param;
  return !full_param.dnnl_param.with_act ||
         (full_param.act_param.alg == dnnl::algorithm::eltwise_relu &&
          full_param.act_param.alpha == 0.f);
}

static bool SgDNNLConvBackwardNeedsFwdOutput(const DNNLConvFusionParam& param) {
  return param.full_conv_param.dnnl_param.with_act;
}

static bool SgDNNLConvBackwardNeedsFwdOutputRange(const DNNLConvFusionParam& param) {
  const auto& dnnl_param = param.full_conv_param.dnnl_param;
  return SgDNNLConvBackwardNeedsFwdOutput(param) &&
         !dnnl_param.enabled_float_output.has_value();
}

static size_t SgDNNLConvBackwardNumAuxInputs(const DNNLConvFusionParam& param) {
  return 1 + (SgDNNLConvBackwardNeedsFwdOutput(param) ? 1 : 0) +
         (SgDNNLConvBackwardNeedsFwdOutputRange(param) ? 2 : 0);
}

static bool SgDNNLConvSupportsQATBackward(const DNNLConvFusionParam& param) {
  const auto& dnnl_param = param.full_conv_param.dnnl_param;
  const bool supported_output_type =
      !dnnl_param.enabled_float_output.has_value() ||
      dnnl_param.enabled_float_output.value() == mshadow::kFloat32;
  const bool supported_fused_output =
      !SgDNNLConvBackwardNeedsFwdOutputRange(param) ||
      (dnnl_param.min_calib_range.has_value() && dnnl_param.max_calib_range.has_value());
  return dnnl_param.quantized && supported_output_type &&
         supported_fused_output &&
         !dnnl_param.with_bn && !dnnl_param.with_sum && !dnnl_param.with_postsum_act &&
         SgDNNLConvHasSupportedQATBackwardAct(param);
}

static NDArray DequantizeQATDataCPU(const NDArray& data,
                                    const NDArray& data_min,
                                    const NDArray& data_max) {
  CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8)
      << "QAT subgraph convolution backward expects int8/uint8 quantized data, got "
      << data.dtype();
  NDArray ret(data.shape(), data.ctx(), false, mshadow::kFloat32);
  const float min_range = data_min.data().dptr<float>()[0];
  const float max_range = data_max.data().dptr<float>()[0];
  float* out            = ret.data().dptr<float>();
  const size_t size     = data.shape().Size();
  const int nthreads    = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  if (data.dtype() == mshadow::kInt8) {
    const int8_t* in = data.data().dptr<int8_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = QuantizedToFloat<int8_t>(in[i], min_range, max_range);
    }
  } else {
    const uint8_t* in    = data.data().dptr<uint8_t>();
    const float scale    = (max_range - min_range) / 255.0f;
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]) * scale + min_range;
    }
  }
  return ret;
}

static NDArray CastQATGradToFloatCPU(const NDArray& grad) {
  const NDArray default_grad = grad.IsDNNLData() ? grad.Reorder2Default() : grad;
  if (default_grad.dtype() == mshadow::kFloat32) {
    return default_grad;
  }
  CHECK(default_grad.dtype() == mshadow::kInt8 || default_grad.dtype() == mshadow::kUint8)
      << "QAT subgraph convolution backward expects float32/int8/uint8 output gradients, got "
      << default_grad.dtype();
  NDArray ret(default_grad.shape(), default_grad.ctx(), false, mshadow::kFloat32);
  float* out        = ret.data().dptr<float>();
  const size_t size = default_grad.shape().Size();
  const int nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  if (default_grad.dtype() == mshadow::kInt8) {
    const int8_t* in = default_grad.data().dptr<int8_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]);
    }
  } else {
    const uint8_t* in = default_grad.data().dptr<uint8_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]);
    }
  }
  return ret;
}

static NDArray ApplyFusedReluGradCPU(const NDArray& out_grad,
                                     const NDArray& fwd_output,
                                     const NDArray* fwd_output_min,
                                     const NDArray* fwd_output_max) {
  CHECK_EQ(out_grad.dtype(), mshadow::kFloat32);
  const NDArray default_output =
      fwd_output.IsDNNLData() ? fwd_output.Reorder2Default() : fwd_output;
  NDArray float_output;
  const NDArray* output = &default_output;
  if (default_output.dtype() == mshadow::kFloat32) {
    CHECK(fwd_output_min == nullptr && fwd_output_max == nullptr)
        << "_backward_sg_onednn_conv got quantized output ranges for a float32 forward output";
  } else {
    CHECK(fwd_output_min != nullptr && fwd_output_max != nullptr)
        << "_backward_sg_onednn_conv needs forward output min/max to backpropagate through "
           "quantized fused ReLU";
    float_output = DequantizeQATDataCPU(default_output, *fwd_output_min, *fwd_output_max);
    output       = &float_output;
  }
  CHECK_EQ(out_grad.shape(), output->shape());
  NDArray ret(out_grad.shape(), out_grad.ctx(), false, mshadow::kFloat32);
  const float* grad = out_grad.data().dptr<float>();
  const float* out  = output->data().dptr<float>();
  float* dst        = ret.data().dptr<float>();
  const size_t len  = out_grad.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < static_cast<index_t>(len); ++i) {
    dst[i] = out[i] > 0.f ? grad[i] : 0.f;
  }
  return ret;
}

template <typename DType>
static void ZeroLikeOutputCPUImpl(const NDArray& out) {
  DType* ptr       = out.data().dptr<DType>();
  const size_t len = out.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < static_cast<index_t>(len); ++i) {
    ptr[i] = DType(0);
  }
}

static void PrepareDefaultOutputCPU(const NDArray& out, OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  // This backward writes through raw TBlob/default pointers.  Executor memory
  // reuse can hand us a chunk that still carries an old oneDNN memory descriptor
  // for a different shape; refresh it to this NDArray's current default shape.
  const_cast<NDArray&>(out).InvalidateDNNLData();
  (void)out.GetDNNLData();
}

static void ZeroLikeOutputCPU(const NDArray& out, OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  if (req == kAddTo) {
    PrepareDefaultOutputCPU(out, req);
    return;
  }
  CHECK(req == kWriteTo || req == kWriteInplace)
      << "Unsupported zero-gradient request type " << req;
  PrepareDefaultOutputCPU(out, req);
  MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
    ZeroLikeOutputCPUImpl<DType>(out);
  });
}

static void SgDNNLConvQATBackward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  const auto& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  CHECK(SgDNNLConvSupportsQATBackward(param))
      << "_backward_sg_onednn_conv currently supports only simple quantized "
         "_sg_onednn_conv with no batchnorm/sum fusion and only ReLU activation fusion";

  const auto& full_conv_param = param.full_conv_param;
  const auto& conv_param      = full_conv_param.conv_param;
  const bool has_bias         = !conv_param.no_bias;
  const size_t fwd_inputs     = SgDNNLConvNumInputs(attrs);
  const bool needs_fwd_output = SgDNNLConvBackwardNeedsFwdOutput(param);
  const size_t aux_inputs     = SgDNNLConvBackwardNumAuxInputs(param);
  CHECK_EQ(inputs.size(), fwd_inputs + aux_inputs);
  CHECK_EQ(outputs.size(), fwd_inputs);
  CHECK_EQ(req.size(), fwd_inputs);

  size_t in_idx = 1;  // inputs[0] is output gradient.
  const NDArray* fwd_output     = needs_fwd_output ? &inputs[in_idx++] : nullptr;
  const NDArray* fwd_output_min = nullptr;
  const NDArray* fwd_output_max = nullptr;
  if (SgDNNLConvBackwardNeedsFwdOutputRange(param)) {
    fwd_output_min = &inputs[in_idx++];
    fwd_output_max = &inputs[in_idx++];
  }
  const NDArray& qdata  = inputs[in_idx++];
  const NDArray& weight = inputs[in_idx++];
  const NDArray* bias   = has_bias ? &inputs[in_idx++] : nullptr;
  const NDArray& data_min = inputs[in_idx++];
  const NDArray& data_max = inputs[in_idx++];
  CHECK_EQ(in_idx, inputs.size());
  CHECK_EQ(weight.dtype(), mshadow::kFloat32)
      << "_backward_sg_onednn_conv QAT path expects float32 weights";
  if (bias) {
    CHECK_EQ(bias->dtype(), mshadow::kFloat32)
        << "_backward_sg_onednn_conv QAT path expects float32 bias";
  }
  CHECK(inputs[0].dtype() == mshadow::kFloat32 || inputs[0].dtype() == mshadow::kInt8 ||
        inputs[0].dtype() == mshadow::kUint8)
      << "_backward_sg_onednn_conv QAT path expects float32/int8/uint8 output gradients";

  NDArray data = DequantizeQATDataCPU(qdata, data_min, data_max);
  NDArray float_grad = CastQATGradToFloatCPU(inputs[0]);
  NDArray conv_out_grad;
  const NDArray* out_grad = &float_grad;
  if (needs_fwd_output) {
    conv_out_grad = ApplyFusedReluGradCPU(float_grad, *fwd_output, fwd_output_min, fwd_output_max);
    out_grad      = &conv_out_grad;
  }

  nnvm::NodeAttrs conv_attrs;
  auto conv_param_copy = conv_param;
  conv_attrs.name      = attrs.name + "_qat_backward";
  conv_attrs.parsed    = conv_param_copy;
  conv_attrs.dict.clear();
  conv_param_copy.SetAttrDict(&conv_attrs.dict);

  std::vector<TBlob> conv_inputs;
  conv_inputs.reserve(has_bias ? 4 : 3);
  conv_inputs.emplace_back(out_grad->data());
  conv_inputs.emplace_back(data.data());
  conv_inputs.emplace_back(weight.data());
  if (has_bias) {
    conv_inputs.emplace_back(bias->data());
  }

  std::vector<TBlob> conv_outputs;
  conv_outputs.reserve(has_bias ? 3 : 2);
  PrepareDefaultOutputCPU(outputs[0], req[0]);
  PrepareDefaultOutputCPU(outputs[1], req[1]);
  conv_outputs.emplace_back(outputs[0].data());
  conv_outputs.emplace_back(outputs[1].data());
  if (has_bias) {
    PrepareDefaultOutputCPU(outputs[2], req[2]);
    conv_outputs.emplace_back(outputs[2].data());
  }

  std::vector<OpReqType> conv_req;
  conv_req.reserve(conv_outputs.size());
  conv_req.insert(conv_req.end(), req.begin(), req.begin() + conv_outputs.size());
  ConvolutionGradCompute<cpu>(conv_attrs, ctx, conv_inputs, conv_req, conv_outputs);

  for (size_t i = conv_outputs.size(); i < outputs.size(); ++i) {
    ZeroLikeOutputCPU(outputs[i], req[i]);
  }
}

static bool SgDNNLConvBackwardShape(const nnvm::NodeAttrs& attrs,
                                    mxnet::ShapeVector* in_shapes,
                                    mxnet::ShapeVector* out_shapes) {
  const size_t fwd_inputs = SgDNNLConvNumInputs(attrs);
  const auto& param       = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  const size_t aux_inputs = SgDNNLConvBackwardNumAuxInputs(param);
  CHECK_EQ(in_shapes->size(), fwd_inputs + aux_inputs);
  CHECK_EQ(out_shapes->size(), fwd_inputs);
  for (size_t i = 0; i < fwd_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*out_shapes, i, (*in_shapes)[i + aux_inputs]);
  }
  return true;
}

static bool SgDNNLConvBackwardType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_types,
                                   std::vector<int>* out_types) {
  const auto& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  CHECK(SgDNNLConvSupportsQATBackward(param));
  const auto& conv_param = param.full_conv_param.conv_param;
  const bool has_bias    = !conv_param.no_bias;
  const size_t fwd_inputs = SgDNNLConvNumInputs(attrs);
  const bool needs_fwd_output = SgDNNLConvBackwardNeedsFwdOutput(param);
  const bool needs_fwd_output_range = SgDNNLConvBackwardNeedsFwdOutputRange(param);
  const size_t aux_inputs     = SgDNNLConvBackwardNumAuxInputs(param);
  CHECK_EQ(in_types->size(), fwd_inputs + aux_inputs);
  CHECK_EQ(out_types->size(), fwd_inputs);

  if (needs_fwd_output_range && in_types->at(0) != -1) {
    CHECK(in_types->at(0) == mshadow::kFloat32 || in_types->at(0) == mshadow::kInt8 ||
          in_types->at(0) == mshadow::kUint8)
        << "_backward_sg_onednn_conv expects float32/int8/uint8 output gradients";
  } else {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
  }
  if (needs_fwd_output) {
    if (needs_fwd_output_range) {
      CHECK(in_types->at(1) == mshadow::kInt8 || in_types->at(1) == mshadow::kUint8)
          << "_backward_sg_onednn_conv expects int8/uint8 quantized forward output";
      TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32);
    } else {
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
    }
  }
  const size_t data_idx = aux_inputs;
  CHECK(in_types->at(data_idx) == mshadow::kInt8 || in_types->at(data_idx) == mshadow::kUint8)
      << "_backward_sg_onednn_conv expects int8/uint8 quantized data input";
  TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);

  size_t idx = data_idx + 1;
  TYPE_ASSIGN_CHECK(*in_types, idx, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_types, idx - aux_inputs, mshadow::kFloat32);
  ++idx;
  if (has_bias) {
    TYPE_ASSIGN_CHECK(*in_types, idx, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_types, idx - aux_inputs, mshadow::kFloat32);
    ++idx;
  }
  for (; idx < in_types->size(); ++idx) {
    TYPE_ASSIGN_CHECK(*in_types, idx, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_types, idx - aux_inputs, mshadow::kFloat32);
  }
  return true;
}

static bool SgDNNLConvBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  for (auto& attr : *out_attrs) {
    type_assign(&attr, mxnet::kDefaultStorage);
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

struct SgDNNLConvGrad {
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const auto& param = nnvm::get<DNNLConvFusionParam>(n->attrs.parsed);
    if (!SgDNNLConvSupportsQATBackward(param)) {
      return MakeZeroGradNodes(n, ograds);
    }
    std::vector<nnvm::NodeEntry> heads;
    heads.reserve(n->inputs.size() + SgDNNLConvBackwardNumAuxInputs(param));
    heads.emplace_back(ograds[0]);
    if (SgDNNLConvBackwardNeedsFwdOutput(param)) {
      heads.emplace_back(n, 0, 0);
      if (SgDNNLConvBackwardNeedsFwdOutputRange(param)) {
        heads.emplace_back(n, 1, 0);
        heads.emplace_back(n, 2, 0);
      }
    }
    heads.insert(heads.end(), n->inputs.begin(), n->inputs.end());
    auto p        = nnvm::Node::Create();
    p->attrs.op   = nnvm::Op::Get("_backward_sg_onednn_conv");
    p->attrs.name = n->attrs.name + "_backward";
    p->attrs.dict = n->attrs.dict;
    p->attrs.subgraphs.reserve(n->attrs.subgraphs.size());
    for (const auto& subgraph : n->attrs.subgraphs) {
      p->attrs.subgraphs.push_back(subgraph);
    }
    p->inputs = std::move(heads);
    p->control_deps.emplace_back(n);
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    CHECK_EQ(p->num_inputs(), p->inputs.size())
        << "Number of inputs to operator _backward_sg_onednn_conv (" << p->num_inputs()
        << ") does not match the actual number of inputs provided to operator "
        << p->attrs.name << " (" << p->inputs.size() << ").";
    return CreateNodeEntries(p);
  }
};

void SgDNNLConvOperator::Forward(const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  // Primary-output req contract: kNullOp means caller does not want the value,
  // so skip the entire forward. kAddTo would require an accumulation step on
  // top of the cached primitive's direct dst binding; reject it explicitly
  // rather than silently overwriting the output buffer.
  if (req[kOut] == kNullOp)
    return;
  CHECK_NE(req[kOut], kAddTo)
      << "kAddTo is not supported for the primary output of _sg_onednn_conv";
  // AMP / oneDNN v3 fallback: when the AMP pass converts a fused conv
  // subgraph to bf16 inputs+weights+output but the running CPU ISA lacks
  // native bf16 (e.g. AVX2), oneDNN v3 cannot create the convolution
  // primitive_desc.  Promote the affected NDArrays to fp32 here, run the
  // (cached) f32 conv, then reorder the f32 result back into the caller's
  // bf16 output.  When the ISA does support bf16 we fall through unchanged.
  if (!DNNLISASupportsLowpFloat(mshadow::kBfloat16)) {
    bool any_bf16 = false;
    for (const auto& nd : inputs) {
      if (nd.dtype() == mshadow::kBfloat16) { any_bf16 = true; break; }
    }
    for (const auto& nd : outputs) {
      if (nd.dtype() == mshadow::kBfloat16) { any_bf16 = true; break; }
    }
    if (any_bf16) {
      std::vector<NDArray> f32_in;
      f32_in.reserve(inputs.size());
      for (const auto& nd : inputs) {
        if (nd.dtype() == mshadow::kBfloat16) {
          f32_in.emplace_back(nd.Reorder2DefaultFloatFormat());
        } else {
          f32_in.emplace_back(nd);
        }
      }
      std::vector<NDArray> f32_out;
      std::vector<bool> out_was_bf16;
      f32_out.reserve(outputs.size());
      out_was_bf16.reserve(outputs.size());
      for (const auto& nd : outputs) {
        if (nd.dtype() == mshadow::kBfloat16) {
          // Pass the kFloat32 enum value directly. `emplace_back` perfect-
          // forwards arguments by reference; using `mshadow::DataType<T>::kFlag`
          // (a non-inline `static const int`) here would odr-use the member
          // and produce an undefined-symbol link error against libmxnet.so.
          f32_out.emplace_back(nd.shape(), nd.ctx(), /*delay_alloc=*/false,
                               static_cast<int>(mshadow::kFloat32));
          out_was_bf16.push_back(true);
        } else {
          f32_out.emplace_back(nd);
          out_was_bf16.push_back(false);
        }
      }
      // For outputs that are already non-BF16 the caller's req still applies
      // because f32_out[i] aliases outputs[i]; for BF16 outputs we own a fresh
      // f32 scratch buffer, so kWriteTo is the only sane inner req. The
      // post-reorder loop below honors the caller's intent for those.
      std::vector<OpReqType> f32_req;
      f32_req.reserve(req.size());
      for (size_t i = 0; i < req.size(); ++i) {
        f32_req.push_back((i < out_was_bf16.size() && out_was_bf16[i]) ? kWriteTo : req[i]);
      }
      this->Forward(ctx, f32_in, f32_req, f32_out);
      DNNLStream::Get()->Submit();
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!out_was_bf16[i]) continue;
        if (req[i] == kNullOp) continue;
        CHECK_NE(req[i], kAddTo)
            << "kAddTo not supported for BF16 fallback path on output " << i
            << " of _sg_onednn_conv";
        auto src_mem = f32_out[i].GetDNNLData();
        auto dst_mem = outputs[i].GetDNNLData();
        ReorderTo(src_mem, dst_mem);
      }
      return;
    }
  }

  auto& full_conv_param = param_.full_conv_param;
  auto& dnnl_param      = full_conv_param.dnnl_param;
  auto& conv_param      = full_conv_param.conv_param;
  auto bn_param         = param_.bn_param.get();

  index_t idx = 0;

  auto in_data   = idx++;
  auto in_weight = idx++;
  auto in_bias   = conv_param.no_bias ? 0 : (idx++);
  auto in_gamma  = dnnl_param.with_bn ? (idx++) : 0;
  auto in_beta   = dnnl_param.with_bn ? (idx++) : 0;
  auto in_mean   = dnnl_param.with_bn ? (idx++) : 0;
  auto in_var    = dnnl_param.with_bn ? (idx++) : 0;
  auto in_sum    = dnnl_param.with_sum ? (dnnl_param.dedup_sum ? in_data : idx++) : -1;
  float data_min = dnnl_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float data_max = dnnl_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float sum_min  = 0.0f;
  float sum_max  = 0.0f;
  if (dnnl_param.with_sum && dnnl_param.quantized) {
    if (dnnl_param.dedup_sum) {
      sum_min = data_min;
      sum_max = data_max;
    } else {
      sum_min = inputs[idx++].data().dptr<float>()[0];
      sum_max = inputs[idx++].data().dptr<float>()[0];
    }
  }
  CHECK_EQ(inputs.size(), idx);
  bool has_bias  = dnnl_param.with_bn || !conv_param.no_bias;
  NDArray data   = inputs[in_data];
  NDArray output = dnnl_param.with_sum ? inputs[in_sum] : outputs[kOut];
  // Copy inputs[in_sum] into outputs[kOut] in case inplace optimization failed.
  if (dnnl_param.with_sum) {
    if (!initialized_) {
      // TODO(zhennan): Currently, dnnl fallback mechanism will break inplace option,
      // which make check (req[kOut] == kWriteInplace) useless.
      auto in_dnnl_mem  = inputs[in_sum].GetDNNLData();
      auto out_dnnl_mem = outputs[kOut].GetDNNLData();
      if (in_dnnl_mem->get_data_handle() == out_dnnl_mem->get_data_handle()) {
        inplace_ = true;
      }
    }
    if (!inplace_) {
      auto in_dnnl_mem  = inputs[in_sum].GetDNNLData();
      auto out_dnnl_mem = outputs[kOut].GetDNNLData();
      if (outputs[kOut].dtype() == mshadow::kInt32 || outputs[kOut].dtype() == mshadow::kFloat32) {
        const auto& mem_desc  = in_dnnl_mem->get_desc();
        const auto this_dtype = get_dnnl_type(outputs[kOut].dtype());
        auto omd              = mem_desc;
        omd = CloneMemDescWithDtype(omd, this_dtype);
        dnnl_mem_ptr tmp_mem(
            new dnnl::memory(omd, CpuEngine::Get()->get_engine(), out_dnnl_mem->get_data_handle()));
        DNNLStream::Get()->RegisterMem(tmp_mem);
        DNNLStream::Get()->RegisterPrimArgs(
            dnnl::reorder(*in_dnnl_mem, *tmp_mem),
            {{DNNL_ARG_FROM, *in_dnnl_mem}, {DNNL_ARG_TO, *tmp_mem}});
        output = NDArray(tmp_mem);
      } else {
        dnnl_mem_ptr tmp_mem(new dnnl::memory(in_dnnl_mem->get_desc(),
                                              CpuEngine::Get()->get_engine(),
                                              out_dnnl_mem->get_data_handle()));
        DNNLStream::Get()->RegisterMem(tmp_mem);
        DNNLMemoryCopy(*in_dnnl_mem, tmp_mem.get());
        output = NDArray(tmp_mem);
      }
    }
  }

  // Check input change
  // TODO(zhennan): Only update cached_* changed.
  if (initialized_) {
    if (dnnl_param.with_bn) {
      if (weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initialized_ = false;
      }
    }
    if (initialized_ && dnnl_param.quantized) {
      if (cached_data_min_ != data_min || cached_data_max_ != data_max ||
          cached_sum_min_ != sum_min || cached_sum_max_ != sum_max ||
          weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initialized_ = false;
      }
    }
  }
  if (!initialized_) {
    cached_data_min_ = data_min;
    cached_data_max_ = data_max;
    cached_sum_min_  = sum_min;
    cached_sum_max_  = sum_max;
    cached_weight_   = inputs[in_weight].Reorder2Default();
    weight_ver_      = inputs[in_weight].version();
    if (!conv_param.no_bias) {
      cached_bias_ = inputs[in_bias];
      bias_ver_    = inputs[in_bias].version();
    } else {
      cached_bias_ = NDArray();
    }

    // Update weight and bias after bn fusion.
    if (dnnl_param.with_bn) {
      DNNL_REAL_TYPE_SWITCH(inputs[in_weight].dtype(), DType, {
        UpdateConvWeightBias<DType>(&cached_weight_,
                                    &cached_bias_,
                                    conv_param.no_bias,
                                    inputs[in_gamma],
                                    inputs[in_beta],
                                    inputs[in_mean],
                                    inputs[in_var],
                                    bn_param);
      });
    }
    // Quantize weight and bias.
    if (dnnl_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      auto weight_channelwise_scale = false;
      if (dnnl_param.min_calib_range.has_value() && dnnl_param.max_calib_range.has_value()) {
        cached_output_min_       = dnnl_param.min_calib_range.value();
        cached_output_max_       = dnnl_param.max_calib_range.value();
        post_requantize_         = true;
        weight_channelwise_scale = true;
      }
      if (dnnl_param.enabled_float_output.has_value()) {
        weight_channelwise_scale = true;
      }
      data_scale_ = GetQuantizeScale(data.dtype(), cached_data_min_, cached_data_max_);
      full_conv_param.src_zero_point =
          (data.dtype() == mshadow::kUint8) ?
              static_cast<int32_t>(std::nearbyint(-cached_data_min_ * data_scale_)) :
              0;
      DNNL_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
        weight_scales_ = GetWeightScales<DType>(cached_weight_,
                                                has_bias ? &cached_bias_ : nullptr,
                                                data_scale_,
                                                weight_channelwise_scale);
      });
      // Collect scale.
      size_t channel     = cached_weight_.shape()[0];
      float sum_in_scale = 1.0;
      float output_scale;
      if (dnnl_param.with_sum) {
        sum_in_scale = GetQuantizeScale(inputs[in_sum].dtype(), cached_sum_min_, cached_sum_max_);
      }
      if (post_requantize_ || dnnl_param.enabled_float_output.has_value()) {
        if (post_requantize_) {
          output_scale = GetQuantizeScale(IsOutputUInt8(param_) ? mshadow::kUint8 : mshadow::kInt8,
                                          cached_output_min_,
                                          cached_output_max_);
        } else {
          output_scale = 1.0;
        }
        if (dnnl_param.enabled_float_output.has_value()) {
          // v3 dequant: split dequant into SRC (1/data_scale) + WEIGHTS
          // (1/weight_scale[c]); bias is f32 in real units. Mirrors
          // dnnl_fc.cc:471-482 to avoid the rejected s32-bias + f32-dst combo.
          full_conv_param.src_scale = 1.0f / data_scale_;
          full_conv_param.requantize_scales.resize(weight_channelwise_scale ? channel : 1);
          for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
            full_conv_param.requantize_scales[c] = 1.0f / weight_scales_[c];
          }
        } else {
          full_conv_param.requantize_scales.resize(weight_channelwise_scale ? channel : 1);
          for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
            full_conv_param.requantize_scales[c] = 1.0 / data_scale_ / weight_scales_[c];
          }
          // oneDNN v3 removed the eltwise post-op scale argument, so the
          // output quantization scale must be folded into the DST scale even
          // when conv+activation is fused. Leaving it in act_param.scale makes
          // u8 ReLU outputs encode real values as tiny integers, which later
          // requantize to zeros.
          for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
            full_conv_param.requantize_scales[c] *= output_scale;
          }
        }
      } else {
        Stream<cpu>* s = ctx.get_stream<cpu>();
        if (data.dtype() == mshadow::kInt8) {
          mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s,
              1,
              &cached_output_min_,
              &cached_output_max_,
              &weight_scales_[1],
              &weight_scales_[2],
              &cached_data_min_,
              &cached_data_max_);
        } else {
          mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
              s,
              1,
              &cached_output_min_,
              &cached_output_max_,
              &weight_scales_[1],
              &weight_scales_[2],
              &cached_data_min_,
              &cached_data_max_);
        }
        // S7: this branch only runs when weight_channelwise_scale==false; in
        // that case GetWeightScales returns {global_scale, min, max} so [0] is
        // the per-tensor weight scale. Asserting to lock in the assumption.
        CHECK(!weight_channelwise_scale)
            << "channelwise-scale weights cannot reach the non-requantize branch";
        weight_scales_.resize(1);
        output_scale = data_scale_ * weight_scales_[0];
        full_conv_param.requantize_scales.resize(0);
      }
      if (dnnl_param.with_sum) {
        // v2:  dst = output_scales * (acc + bias) + sum_scale_v2 * dst_loaded
        // v3:  dst = DST_scale * (acc + bias + sum_scale_v3 * dst_loaded)
        // For per-tensor DST scale, pre-divide so the sum contribution matches
        // v2's. For per-OC (WEIGHTS-side scaling) there is no DST scale, so
        // sum_scale_v2 is used directly. With no requantize, no DST scale.
        full_conv_param.sum_scale = output_scale / sum_in_scale;
        if (full_conv_param.requantize_scales.size() == 1) {
          full_conv_param.sum_scale /= full_conv_param.requantize_scales[0];
        }
      }
      if (dnnl_param.with_act &&
          full_conv_param.act_param.alg == dnnl::algorithm::eltwise_clip) {
        if (dnnl_param.with_sum) {
          LOG(ERROR) << "oneDNN doesn't support conv + relu + sum fusion yet.";
          full_conv_param.act_param.alpha *= output_scale;
        }
      }
    }
    fwd_.reset(new DNNLConvForward(full_conv_param,
                                   ctx.is_train,
                                   data,
                                   cached_weight_,
                                   has_bias ? &cached_bias_ : nullptr,
                                   output));
    dnnl::memory::desc bias_md;
    if (has_bias)
      bias_md = fwd_->GetPd().bias_desc();
    ConvertWeightBias2DNNL(&cached_weight_,
                           &cached_bias_,
                           has_bias,
                           fwd_->GetPd().weights_desc(),
                           has_bias ? &bias_md : nullptr,
                           conv_param.num_group,
                           data_scale_,
                           weight_scales_);
    args_[DNNL_ARG_SRC]     = *data.GetDNNLData();
    args_[DNNL_ARG_WEIGHTS] = *cached_weight_.GetDNNLData();
    if (has_bias)
      args_[DNNL_ARG_BIAS] = *cached_bias_.GetDNNLData();
    args_[DNNL_ARG_DST] = *output.GetDNNLData();
    // v3: bind runtime output-scale tensor for quantized (requantize) conv.
    // ARG key is WEIGHTS for per-OC scales (mask=1) or DST for per-tensor
    // (mask=0) — determined inside DNNLConvForward to match the
    // set_scales_mask attr installed in GetConvFwdImpl.
    if (auto* sm = fwd_->GetOutputScaleMem()) {
      args_[DNNL_ARG_ATTR_SCALES | fwd_->GetOutputScaleArg()] = *sm;
    }
    if (auto* ssm = fwd_->GetSrcScaleMem()) {
      args_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = *ssm;
    }
    if (auto* zpm = fwd_->GetSrcZeroPointMem()) {
      args_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = *zpm;
    }
    initialized_        = true;
  }

  if (dnnl_param.with_sum) {
    const auto& output_mem   = output.GetDNNLData();
    const auto& out_mem_desc = output_mem->get_desc();
    const auto& dst_mem_desc = fwd_->GetPd().dst_desc();
    if (out_mem_desc != dst_mem_desc) {
      auto tmp_out_mem       = output.GetDNNLDataReorder(&dst_mem_desc);
      auto data_md           = dst_mem_desc;
      // v3: CloneMemDescWithDtype takes the C++ enum directly now.
      data_md = CloneMemDescWithDtype(data_md, out_mem_desc.get_data_type());
      dnnl_mem_ptr new_out_mem(
          new dnnl::memory(data_md, CpuEngine::Get()->get_engine(), output_mem->get_data_handle()));
      DNNLStream::Get()->RegisterMem(new_out_mem);
      DNNLMemoryCopy(*tmp_out_mem, new_out_mem.get());
      output = NDArray(new_out_mem);
    }
  }

  if (dnnl_param.quantized) {
    auto fwd_src_desc    = fwd_->GetPd().src_desc();
    auto data_mem        = data.GetDNNLDataReorder(&fwd_src_desc);
    auto fwd_pd_dst_desc = fwd_->GetPd().dst_desc();
    dnnl::memory* mem    = output.CreateDNNLData(&fwd_pd_dst_desc);
    args_[DNNL_ARG_SRC]  = *data_mem;
    args_[DNNL_ARG_DST]  = *mem;
    DNNLStream::Get()->RegisterPrimArgs(fwd_->GetFwd(), args_);
    DNNLStream::Get()->Submit();
  } else {
    std::vector<NDArray> new_inputs;
    if (has_bias) {
      new_inputs = {data, cached_weight_, cached_bias_};
    } else {
      new_inputs = {data, cached_weight_};
    }
    DNNLConvolutionForwardFullFeature(full_conv_param, ctx, fwd_.get(), new_inputs, req, {output});
  }

  if (dnnl_param.quantized && !dnnl_param.enabled_float_output.has_value()) {
    // Route through the shared helper so kAddTo accumulates rather than
    // silently dropping the existing scalar.
    AssignQuantizedRangeOutput(outputs[kMin].data().dptr<float>(),
                               &cached_output_min_,
                               req[kMin],
                               "_sg_onednn_conv");
    AssignQuantizedRangeOutput(outputs[kMax].data().dptr<float>(),
                               &cached_output_max_,
                               req[kMax],
                               "_sg_onednn_conv");
  }
  if (dnnl_param.with_sum) {
    auto out          = const_cast<NDArray&>(outputs[kOut]);
    auto fwd_dst_desc = fwd_->GetPd().dst_desc();
    out.UpdateDNNLMemDesc(&fwd_dst_desc);
  }
}

static void SgDNNLConvOpForward(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  SgDNNLConvOperator& op = state_ptr.get_state<SgDNNLConvOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

static uint32_t SgDNNLConvNumInputs(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  auto num_input    = DefaultSubgraphOpNumInputs(attrs);
  if (param.full_conv_param.dnnl_param.quantized)
    return num_input + 2 +
           (param.full_conv_param.dnnl_param.with_sum &&
                    !param.full_conv_param.dnnl_param.dedup_sum ?
                2 :
                0);
  else
    return num_input;
}

static void SgDNNLConvParamParser(nnvm::NodeAttrs* attrs) {
  DNNLConvFusionParam param_;

  // For back-compatible, rename
  // with_relu -> with_act
  // with_postsum_relu -> with_postsum_act

  auto old = attrs->dict.find("with_relu");
  if (old != attrs->dict.end()) {
    attrs->dict["with_act"] = old->second;
    attrs->dict.erase(old);
  }

  old = attrs->dict.find("with_postsum_relu");
  if (old != attrs->dict.end()) {
    attrs->dict["with_postsum_act"] = old->second;
    attrs->dict.erase(old);
  }

  try {
    param_.full_conv_param.dnnl_param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  CHECK_EQ(attrs->subgraphs.size(), 1);
  auto subgraph_sym = attrs->subgraphs[0];
  bool with_act     = false;
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr& node) {
    if (node->is_variable())
      return;
    auto& node_name = node->op()->name;
    if (node_name == "BatchNorm") {
      CHECK_EQ(param_.full_conv_param.dnnl_param.with_bn, true);
      CHECK(param_.bn_param.get() == nullptr);
      param_.bn_param =
          std::make_shared<BatchNormParam>(nnvm::get<BatchNormParam>(node->attrs.parsed));
    } else if (node_name == "Convolution") {
      param_.full_conv_param.conv_param = nnvm::get<ConvolutionParam>(node->attrs.parsed);
    } else if (node_name == "Activation" || node_name == "LeakyReLU" || node_name == "clip") {
      auto& post_act_param = (param_.full_conv_param.dnnl_param.with_act && !with_act) ?
                                 param_.full_conv_param.act_param :
                                 param_.full_conv_param.postsum_act_param;
      if (node_name == "Activation") {
        const auto act_param = nnvm::get<ActivationParam>(node->attrs.parsed);
        post_act_param.alg   = GetDNNLActAlgo(act_param);
        // v3 eltwise_soft_relu = log(1+exp(alpha*x))/alpha — alpha=0 is a
        // division by zero. softrelu uses alpha=1; log_sigmoid is encoded as
        // soft_relu with alpha=-1.
        if (act_param.act_type == activation::kSoftReLU) {
          post_act_param.alpha = 1.0f;
        } else if (act_param.act_type == activation::kLogSigmoid) {
          post_act_param.alpha = -1.0f;
        }
      } else if (node_name == "LeakyReLU") {
        const auto act_param = nnvm::get<LeakyReLUParam>(node->attrs.parsed);
        post_act_param.alpha = act_param.slope;
        post_act_param.alg   = GetDNNLActAlgo(act_param);
      } else {
        // v3: bounded_relu(alpha=upper) became eltwise_clip(alpha=lower, beta=upper).
        const auto clip_param = nnvm::get<ClipParam>(node->attrs.parsed);
        post_act_param.alg    = dnnl::algorithm::eltwise_clip;
        post_act_param.alpha  = 0.f;
        post_act_param.beta   = clip_param.a_max;
      }
      with_act = true;
    }
  });
  attrs->parsed = std::move(param_);
}

static std::vector<std::string> SgDNNLConvListInputNames(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  std::vector<std::string> input_names;
  input_names.emplace_back("data");
  input_names.emplace_back("weight");
  if (!param.full_conv_param.conv_param.no_bias) {
    input_names.emplace_back("bias");
  }
  if (param.full_conv_param.dnnl_param.with_bn) {
    input_names.emplace_back("gamma");
    input_names.emplace_back("beta");
    input_names.emplace_back("mean");
    input_names.emplace_back("var");
  }
  auto& dnnl_param = param.full_conv_param.dnnl_param;
  if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
    input_names.emplace_back("sum");
  }
  if (param.full_conv_param.dnnl_param.quantized) {
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  CHECK_EQ(input_names.size(), SgDNNLConvNumInputs(attrs));
  return input_names;
}

static std::vector<std::string> SgDNNLConvListOutputNames(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized &&
      !param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
    return std::vector<std::string>{"output", "output_min", "output_max"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

static OpStatePtr CreateSgDNNLConvState(const nnvm::NodeAttrs& attrs,
                                        Context ctx,
                                        const mxnet::ShapeVector& in_shapes,
                                        const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgDNNLConvOperator>(attrs);
}

template <typename DType>
static void FilterMinMaxIndice(const DNNLConvParam& dnnl_param,
                               std::vector<DType>* in_shapes,
                               std::vector<DType>* out_shapes,
                               std::vector<DType>* base_in_shapes,
                               std::vector<DType>* base_out_shapes,
                               std::unordered_set<size_t>* minmax_indice) {
  base_out_shapes->push_back(out_shapes->at(0));
  size_t last = in_shapes->size() - 1;
  if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    minmax_indice->insert(last - 2);
    minmax_indice->insert(last - 3);
    *base_in_shapes = std::vector<DType>(in_shapes->begin(), in_shapes->end() - 4);
  } else {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    *base_in_shapes = std::vector<DType>(in_shapes->begin(), in_shapes->end() - 2);
  }
}

static bool SgDNNLConvInferShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_shapes,
                                 mxnet::ShapeVector* out_shapes) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;

    FilterMinMaxIndice<mxnet::TShape>(param.full_conv_param.dnnl_param,
                                      in_shapes,
                                      out_shapes,
                                      &base_in_shapes,
                                      &base_out_shapes,
                                      &minmax_indice);
    bool result     = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (minmax_indice.count(i)) {
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
      } else {
        in_shapes->at(i) = base_in_shapes[base_idx++];
      }
    }
    out_shapes->at(0) = base_out_shapes[0];
    if (!param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return result;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgDNNLConvInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_types,
                                std::vector<int>* out_types) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    if (in_types->at(0) == mshadow::kBfloat16) {
      return false;
    }

    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_types;
    std::vector<int> base_out_types;
    FilterMinMaxIndice<int>(param.full_conv_param.dnnl_param,
                            in_types,
                            out_types,
                            &base_in_types,
                            &base_out_types,
                            &minmax_indice);
    // Override data type to fp32 for default infer type as bn doesn't support
    // uint8.
    int orig_data    = base_in_types[0];
    base_in_types[0] = mshadow::kFloat32;
    int orig_sum     = -1;
    auto& dnnl_param = param.full_conv_param.dnnl_param;
    if (param.full_conv_param.dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      auto sum_index           = GetInSumIndex(param);
      orig_sum                 = base_in_types[sum_index];
      base_in_types[sum_index] = mshadow::kFloat32;
    }
    bool result      = DefaultSubgraphOpType(attrs, &base_in_types, &base_out_types);
    base_in_types[0] = orig_data;
    if (param.full_conv_param.dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      auto sum_index           = GetInSumIndex(param);
      base_in_types[sum_index] = orig_sum;
    }
    size_t base_idx = 0;
    for (size_t i = 0; i < in_types->size(); ++i) {
      if (minmax_indice.count(i)) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      } else {
        in_types->at(i) = base_in_types[base_idx++];
      }
    }

    if (param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      TYPE_ASSIGN_CHECK(
          *out_types, 0, param.full_conv_param.dnnl_param.enabled_float_output.value());
    } else {
      if (param.full_conv_param.dnnl_param.min_calib_range.has_value() &&
          param.full_conv_param.dnnl_param.max_calib_range.has_value()) {
        if (IsOutputUInt8(param)) {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
        } else {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
        }
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }

      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return result;
  } else {
    bool result = DefaultSubgraphOpType(attrs, in_types, out_types);
    if (param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      (*out_types)[0] = param.full_conv_param.dnnl_param.enabled_float_output.value();
    }
    return result;
  }
}

static bool SgDNNLConvOpStorageType(const nnvm::NodeAttrs& attrs,
                                    const int dev_mask,
                                    DispatchMode* dispatch_mode,
                                    std::vector<int>* in_stypes,
                                    std::vector<int>* out_stypes) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_stypes;
    std::vector<int> base_out_stypes;
    FilterMinMaxIndice<int>(param.full_conv_param.dnnl_param,
                            in_stypes,
                            out_stypes,
                            &base_in_stypes,
                            &base_out_stypes,
                            &minmax_indice);
    bool result = DefaultSubgraphOpStorageType(
        attrs, dev_mask, dispatch_mode, &base_in_stypes, &base_out_stypes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_stypes->size(); ++i) {
      if (minmax_indice.count(i)) {
        type_assign(&in_stypes->at(i), mxnet::kDefaultStorage);
      } else {
        in_stypes->at(i) = base_in_stypes[base_idx++];
      }
    }
    out_stypes->at(0) = base_out_stypes[0];
    if (!param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      type_assign(&out_stypes->at(1), mxnet::kDefaultStorage);
      type_assign(&out_stypes->at(2), mxnet::kDefaultStorage);
    }
    return result;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_stypes, out_stypes);
  }
}

std::vector<std::pair<int, int>> SgDNNLConvInplaceOption(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.with_sum && !param.full_conv_param.dnnl_param.dedup_sum) {
    return std::vector<std::pair<int, int>>{{GetInSumIndex(param), 0}};
  } else {
    return std::vector<std::pair<int, int>>();
  }
}

nnvm::ObjectPtr SgDNNLConvQuantizedOp(const NodeAttrs& attrs) {
  auto const& param    = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_sg_onednn_conv");
  const int k_ndims    = param.full_conv_param.conv_param.kernel.ndim();
  CHECK(k_ndims == 2U || k_ndims == 3U)
      << "Quantized Convolution of oneDNN supports 2D/3D kernel currently."
      << "Please exclude this layer from the quantized model.";
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["quantized"] = "true";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

bool SgDNNLAvoidConvQuantizeInput(const NodeAttrs& attrs,
                                  const size_t index,
                                  const std::string quantize_granularity) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  std::unordered_set<size_t> avoid_indice;
  size_t idx = 0;
  idx++;                       // data
  avoid_indice.insert(idx++);  // weight
  if (!param.full_conv_param.conv_param.no_bias) {
    avoid_indice.insert(idx++);  // bias
  }
  if (param.full_conv_param.dnnl_param.with_bn) {
    avoid_indice.insert(idx++);  // gamma
    avoid_indice.insert(idx++);  // beta
    avoid_indice.insert(idx++);  // mean
    avoid_indice.insert(idx++);  // var
  }
  return avoid_indice.count(index);
}

NNVM_REGISTER_OP(_sg_onednn_conv)
    .add_alias("_sg_mkldnn_conv")
    .describe(R"code(_sg_onednn_conv)code" ADD_FILELINE)
    .set_num_inputs(SgDNNLConvNumInputs)
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
      if (param.full_conv_param.dnnl_param.quantized &&
          !param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
        return 3;
      }
      return 1;
    })
    .set_attr_parser(SgDNNLConvParamParser)
    .set_attr<nnvm::FListInputNames>("FListInputNames", SgDNNLConvListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", SgDNNLConvListOutputNames)
    .set_attr<FCreateOpState>("FCreateOpState", CreateSgDNNLConvState)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLConvInferShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLConvInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLConvOpStorageType)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLConvOpForward)
    .set_attr<bool>("TIsDNNL", true)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", SgDNNLConvGrad{})
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FMutateInputs>("FMutateInputs", DefaultSubgraphOpMutableInputs)
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", SgDNNLConvInplaceOption)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLConvQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgDNNLAvoidConvQuantizeInput);

NNVM_REGISTER_OP(_backward_sg_onednn_conv)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const auto& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
      return SgDNNLConvNumInputs(attrs) + SgDNNLConvBackwardNumAuxInputs(param);
    })
    .set_num_outputs([](const NodeAttrs& attrs) { return SgDNNLConvNumInputs(attrs); })
    .set_attr_parser(SgDNNLConvParamParser)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLConvBackwardShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLConvBackwardType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLConvBackwardStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FComputeEx>("FComputeEx<cpu>", SgDNNLConvQATBackward);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
