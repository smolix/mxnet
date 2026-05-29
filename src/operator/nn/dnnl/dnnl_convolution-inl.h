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
 * \file dnnl_convolution-inl.h
 * \brief
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_CONVOLUTION_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <utility>
#include <vector>

#include "operator/nn/convolution-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

struct DNNLConvParam : public dmlc::Parameter<DNNLConvParam> {
  bool with_bn;
  bool with_act;
  bool with_sum;
  bool with_postsum_act;
  bool quantized;
  bool dedup_sum;

  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  dmlc::optional<int> enabled_float_output;

  DMLC_DECLARE_PARAMETER(DNNLConvParam) {
    DMLC_DECLARE_FIELD(with_bn).set_default(false).describe("Add post batchnorm.");
    DMLC_DECLARE_FIELD(with_act).set_default(false).describe("Add post activation");
    DMLC_DECLARE_FIELD(with_sum).set_default(false).describe("Add post sum");
    DMLC_DECLARE_FIELD(with_postsum_act)
        .set_default(false)
        .describe("Add post activation after sum");
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe("enable quantization");
    DMLC_DECLARE_FIELD(dedup_sum).set_default(false).describe("deduplicated sum input");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized convolution op to calculate primitive scale");
    DNNL_DECLARE_ENABLED_FLOAT_OUTPUT_PARAMETER();
  }
};

struct DNNLConvFullParam {
  ConvolutionParam conv_param;
  DNNLConvParam dnnl_param;
  float sum_scale = 1.f;
  // For int8/u8 output, requantize_scales[c] = out_quant_scale /
  // (data_scale * weight_scale[c]). For f32 output (dequant-as-output), it is
  // 1 / weight_scale[c] and the matching SRC scale 1/data_scale is in src_scale.
  std::vector<float> requantize_scales;
  float src_scale = 0.f;
  int32_t src_zero_point = 0;
  DNNLPostEltwiseParam act_param;
  DNNLPostEltwiseParam postsum_act_param;
};

std::shared_ptr<dnnl::convolution_forward::primitive_desc> GetConvFwdImpl(
    const ConvolutionParam& param,
    const bool is_train,
    const NDArray& data,
    const NDArray& weight,
    const NDArray* bias,
    const NDArray& output);

class DNNLConvForward {
 public:
  DNNLConvForward(const DNNLConvFullParam& param,
                  const bool is_train,
                  const NDArray& data,
                  const NDArray& weight,
                  const NDArray* bias,
                  const NDArray& output);

  const dnnl::convolution_forward& GetFwd() const {
    return *fwd_;
  }

  const dnnl::convolution_forward::primitive_desc& GetPd() const {
    return *pd_;
  }

  // v3 quantized path: runtime output-scale tensor. Per-tensor scales bind on
  // DNNL_ARG_DST (mask=0); per-OC scales bind on DNNL_ARG_WEIGHTS (mask=1)
  // because v3 conv rejects per-channel DST masks. Caller queries the arg key
  // via GetOutputScaleArg(). Returns nullptr if not quantized.
  const dnnl::memory* GetOutputScaleMem() const { return out_scale_mem_.get(); }
  int GetOutputScaleArg() const { return out_scale_arg_; }
  // f32-output mode only: SRC dequant scale (1/data_scale), nullptr otherwise.
  const dnnl::memory* GetSrcScaleMem() const { return src_scale_mem_.get(); }
  const dnnl::memory* GetSrcZeroPointMem() const { return src_zero_point_mem_.get(); }

 private:
  std::shared_ptr<dnnl::convolution_forward> fwd_;
  std::shared_ptr<dnnl::convolution_forward::primitive_desc> pd_;
  std::shared_ptr<dnnl::memory> out_scale_mem_;
  std::shared_ptr<dnnl::memory> src_scale_mem_;
  std::shared_ptr<dnnl::memory> src_zero_point_mem_;
  int out_scale_arg_{DNNL_ARG_DST};
};

typedef ParamOpSign<ConvolutionParam> DNNLConvSignature;

DNNLConvForward& GetConvFwd(const DNNLConvFullParam& param,
                            const bool is_train,
                            const NDArray& data,
                            const NDArray& weight,
                            const NDArray* bias,
                            const NDArray& output);

void DNNLConvolutionForwardFullFeature(const DNNLConvFullParam& param,
                                       const OpContext& ctx,
                                       DNNLConvForward* fwd,
                                       const std::vector<NDArray>& in_data,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& out_data);

void DNNLConvolutionForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& out_data);

class DNNLConvBackward {
 public:
  DNNLConvBackward(const DNNLConvFullParam& param,
                   const NDArray& data,
                   const NDArray& weight,
                   const NDArray* bias,
                   const NDArray& output);

  const dnnl::convolution_backward_data& GetBwdData() const {
    return *bwd_data_;
  }

  const dnnl::convolution_backward_weights& GetBwdWeights() const {
    return *bwd_weight_;
  }

  const dnnl::convolution_backward_data::primitive_desc& GetDataPd() const {
    return *bwd_data_pd_;
  }

  const dnnl::convolution_backward_weights::primitive_desc& GetWeightsPd() const {
    return *bwd_weight_pd_;
  }

 private:
  std::shared_ptr<dnnl::convolution_backward_data::primitive_desc> bwd_data_pd_;
  std::shared_ptr<dnnl::convolution_backward_weights::primitive_desc> bwd_weight_pd_;
  std::shared_ptr<dnnl::convolution_backward_data> bwd_data_;
  std::shared_ptr<dnnl::convolution_backward_weights> bwd_weight_;
};

void DNNLConvolutionForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& out_data);

void DNNLConvolutionBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_CONVOLUTION_INL_H_
