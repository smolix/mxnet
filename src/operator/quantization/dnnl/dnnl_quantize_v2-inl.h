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
 * \file dnnl_quantize_v2-inl.h
 * \brief
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_V2_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_V2_INL_H_
#if MXNET_USE_ONEDNN == 1
#include <algorithm>
#include <string>
#include <vector>

#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantize_v2-inl.h"
#include "operator/quantization/quantized_range_utils.h"

namespace mxnet {
namespace op {

class SgDNNLQuantizeOperator {
 public:
  explicit SgDNNLQuantizeOperator(const nnvm::NodeAttrs& attrs)
      : attrs_(attrs), param_(nnvm::get<QuantizeV2Param>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  void RunNativeCPUFallback(const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs);

  nnvm::NodeAttrs attrs_;
  bool initalized_{false};
  QuantizeV2Param param_;
  float cached_data_min_{0.f};
  float cached_data_max_{0.f};
  dnnl::memory::desc o_desc_;
  dnnl_args_map_t args_;
  std::shared_ptr<dnnl::reorder> fwd_pd_;
  // v3: runtime scale tensor for set_scales_mask reorder attr.
  dnnl::memory scale_mem_;
};

void SgDNNLQuantizeOperator::RunNativeCPUFallback(const OpContext& ctx,
                                                 const std::vector<NDArray>& inputs,
                                                 const std::vector<OpReqType>& req,
                                                 const std::vector<NDArray>& outputs) {
  NDArray data = inputs[0];
  if (data.IsDNNLData()) {
    data = data.Reorder2Default();
    DNNLStream::Get()->Submit();
  }
  if (req[0] != kNullOp) {
    const_cast<NDArray&>(outputs[0]).InvalidateDNNLData();
  }
  std::vector<TBlob> input_blobs{data.data()};
  std::vector<TBlob> output_blobs{outputs[0].data(), outputs[1].data(), outputs[2].data()};
  QuantizeV2Operator<cpu> native_op(attrs_);
  native_op.Forward(ctx, input_blobs, req, output_blobs);
}

void SgDNNLQuantizeOperator::Forward(const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  float quantized_range = 0.0;
  NDArray in_buffer     = inputs[0];
  float data_min        = mshadow::red::limits::MaxValue<float>();
  float data_max        = mshadow::red::limits::MinValue<float>();

  // Pass through quantized data
  if (inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8) {
    CHECK_NE(req[0], kAddTo)
        << "_contrib_quantize_v2 oneDNN quantized pass-through does not support kAddTo";
    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      const float out_min = param_.min_calib_range.value();
      const float out_max = param_.max_calib_range.value();
      AssignQuantizedRangeOutput(
          outputs[1].data().dptr<float>(), &out_min, req[1], "_contrib_quantize_v2");
      AssignQuantizedRangeOutput(
          outputs[2].data().dptr<float>(), &out_max, req[2], "_contrib_quantize_v2");
    } else {
      if (inputs[0].dtype() == mshadow::kUint8) {
        const float out_min = 0;
        const float out_max = kUint8Range;
        AssignQuantizedRangeOutput(
            outputs[1].data().dptr<float>(), &out_min, req[1], "_contrib_quantize_v2");
        AssignQuantizedRangeOutput(
            outputs[2].data().dptr<float>(), &out_max, req[2], "_contrib_quantize_v2");
      } else {
        const float out_min = -kInt8Range;
        const float out_max = kInt8Range;
        AssignQuantizedRangeOutput(
            outputs[1].data().dptr<float>(), &out_min, req[1], "_contrib_quantize_v2");
        AssignQuantizedRangeOutput(
            outputs[2].data().dptr<float>(), &out_max, req[2], "_contrib_quantize_v2");
      }
    }
    if (req[0] != kNullOp && req[0] != kWriteInplace) {
      const_cast<NDArray&>(outputs[0]).CopyFrom(*inputs[0].GetDNNLData());
      DNNLStream::Get()->Submit();
    }
  } else {
    if (in_buffer.IsView() && in_buffer.IsDNNLData())
      in_buffer = inputs[0].Reorder2Default();
    auto i_mem = in_buffer.GetDNNLData();

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      data_min = param_.min_calib_range.value();
      data_max = param_.max_calib_range.value();
    } else {
      // no calib info
      in_buffer     = inputs[0].Reorder2Default();
      auto in_ptr   = in_buffer.data().dptr<float>();
      auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
      std::vector<float> data_maxs(nthreads, data_max);
      std::vector<float> data_mins(nthreads, data_min);
#pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(in_buffer.shape().Size()); i++) {
        int tid = omp_get_thread_num();
        if (in_ptr[i] > data_maxs[tid])
          data_maxs[tid] = in_ptr[i];
        if (in_ptr[i] < data_mins[tid])
          data_mins[tid] = in_ptr[i];
      }
      for (index_t i = 0; i < nthreads; i++) {
        if (data_maxs[i] > data_max)
          data_max = data_maxs[i];
        if (data_mins[i] < data_min)
          data_min = data_mins[i];
      }

      if (initalized_ && (cached_data_min_ != data_min || cached_data_max_ != data_max))
        initalized_ = false;
    }

    // Write output min/max
    auto out_type = GetQuantizeOutputType(param_);
    if (out_type == mshadow::kUint8) {
      RunNativeCPUFallback(ctx, inputs, req, outputs);
      return;
    } else if (out_type == mshadow::kInt8) {
      float real_range = MaxAbs(data_min, data_max);
      quantized_range  = kInt8Range;
      const float out_min = -real_range;
      const float out_max = real_range;
      AssignQuantizedRangeOutput(
          outputs[1].data().dptr<float>(), &out_min, req[1], "_contrib_quantize_v2");
      AssignQuantizedRangeOutput(
          outputs[2].data().dptr<float>(), &out_max, req[2], "_contrib_quantize_v2");
    } else {
      LOG(FATAL) << "oneDNN quantize op only supports int8 and uint8 as output type";
    }
    if (req[0] == kNullOp)
      return;

    if (!initalized_) {
      cached_data_min_ = data_min;
      cached_data_max_ = data_max;
      float real_range = MaxAbs(data_min, data_max);
      float scale      = quantized_range / real_range;
      // v3: set_output_scales removed; use set_scales_mask + runtime arg.
      dnnl::primitive_attr attr;
      const int mask = 0;
      attr.set_scales_mask(DNNL_ARG_SRC, mask);
      dnnl::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
      auto i_desc             = i_mem->get_desc();
      size_t i_ndim           = in_buffer.shape().ndim();
      if (i_ndim == 4) {
        dnnl::memory::format_tag o_fmt = dnnl::memory::format_tag::nhwc;
        // v3: get_dims() returns a vector by value; store once before iterating.
        const auto src_dims = i_desc.get_dims();
        dnnl::memory::dims o_dims(src_dims.begin(), src_dims.end());
        o_desc_ = dnnl::memory::desc(o_dims, get_dnnl_type(out_type), o_fmt);
      } else {
        o_desc_                = i_desc;
        o_desc_ = CloneMemDescWithDtype(o_desc_, get_dnnl_type_t(out_type));
      }
      auto reorder_pd =
          dnnl::reorder::primitive_desc(cpu_engine, i_desc, cpu_engine, o_desc_, attr);
      fwd_pd_     = std::make_shared<dnnl::reorder>(reorder_pd);
      // v3: bind runtime scale tensor for the reorder attr.
      dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::x);
      scale_mem_ = dnnl::memory(scale_md, cpu_engine);
      *reinterpret_cast<float*>(scale_mem_.get_data_handle()) = scale;
      initalized_ = true;
    }
    auto o_mem           = CreateDNNLMem(outputs[0], o_desc_, req[0]);
    args_[DNNL_ARG_FROM] = *i_mem;
    args_[DNNL_ARG_TO]   = *o_mem.second;
    args_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = scale_mem_;
    DNNLStream::Get()->RegisterPrimArgs(*fwd_pd_, args_);
    CommitOutput(outputs[0], o_mem);
    DNNLStream::Get()->Submit();
  }
}

static void SgDNNLQuantizeForward(const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  SgDNNLQuantizeOperator& op = state_ptr.get_state<SgDNNLQuantizeOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_V2_INL_H_
