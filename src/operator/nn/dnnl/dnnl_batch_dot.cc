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
 * \file dnnl_batch_dot.cc
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#if MXNET_USE_ONEDNN == 1

#include <cmath>
#include <cstring>
#include <limits>

#include "dnnl_batch_dot-inl.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLDotParam);

// Support for https://oneapi-src.github.io/oneDNN/v3/dev_guide_matmul.html
bool SupportDNNLBatchDot(const std::vector<NDArray>& inputs) {
#if defined(__aarch64__) || defined(_M_ARM64)
  // The oneDNN AArch64 matmul/batch-dot implementation currently routes small
  // CPU batch-dot workloads through a Xbyak_aarch64 JIT path that fails on
  // Apple Silicon. Use MXNet's native CPU batch-dot fallback on ARM64 until
  // that oneDNN path is reliable.
  return false;
#else
  return SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::lhs]) &&
         SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::rhs]);
#endif
}

DNNLBatchDotFwd& DNNLBatchDotFwd::GetCached(const DNNLDotParam& param,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<NDArray>& outputs) {
  using batch_dot_fwd_map = std::unordered_map<BatchDotSignature, DNNLBatchDotFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local batch_dot_fwd_map fwds;
#else
  static MX_THREAD_LOCAL batch_dot_fwd_map fwds;
#endif

  BatchDotSignature key(param);
  key.AddSign(inputs[DotIn::lhs]);
  key.AddSign(inputs[DotIn::rhs]);
  key.AddSign(outputs[DotOut::out]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLBatchDotFwd fwd(param, inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

// Returns the v3 primitive_attr for a quantized batch_dot and, via *out_scale,
// the scalar output scale that must be bound at execute time as the runtime
// DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST memory. *out_scale is left unchanged
// (caller initializes to NaN) when no requantize / float-output path is taken.
dnnl::primitive_attr GetQuantizationAttributes(const DNNLDotParam& param,
                                               const std::vector<NDArray>& inputs,
                                               const std::vector<NDArray>& outputs,
                                               float* out_scale) {
  dnnl::primitive_attr attr;
  float lhs_scale_ = GetQuantizeScale(inputs[DotIn::lhs].dtype(),
                                      inputs[DotIn::lhs_min].data().dptr<float>()[0],
                                      inputs[DotIn::lhs_max].data().dptr<float>()[0]);
  float rhs_scale_ = GetQuantizeScale(inputs[DotIn::rhs].dtype(),
                                      inputs[DotIn::rhs_min].data().dptr<float>()[0],
                                      inputs[DotIn::rhs_max].data().dptr<float>()[0]);
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    *out_scale = lhs_scale_ * rhs_scale_ /
                 GetQuantizeScale(outputs[DotOut::out].dtype(),
                                  param.min_calib_range.value(),
                                  param.max_calib_range.value());
    attr.set_scales_mask(DNNL_ARG_DST, 0);
  } else if (param.enabled_float_output.has_value()) {
    *out_scale = lhs_scale_ * rhs_scale_;
    attr.set_scales_mask(DNNL_ARG_DST, 0);
  }
  return attr;
}

namespace {

dnnl::memory::dim GetBatchDotBatchDim(const mxnet::TShape& shape) {
  dnnl::memory::dim big_dim = shape[0];
  for (size_t i = 1; i < shape.ndim() - 2; ++i) {
    big_dim *= shape[i];
  }
  return big_dim;
}

dnnl::memory::desc GetBatchDotMemoryDesc(const NDArray& tensor,
                                         const bool transpose,
                                         const bool allow_any) {
  const auto shape  = tensor.shape();
  const auto ndim   = shape.ndim();
  const auto bigDim = GetBatchDotBatchDim(shape);
  if (transpose) {
    return dnnl::memory::desc(dnnl::memory::dims{bigDim, shape[ndim - 1], shape[ndim - 2]},
                              get_dnnl_type(tensor.dtype()),
                              dnnl::memory::format_tag::acb);
  }
  return dnnl::memory::desc(dnnl::memory::dims{bigDim, shape[ndim - 2], shape[ndim - 1]},
                            get_dnnl_type(tensor.dtype()),
                            allow_any ? dnnl::memory::format_tag::any
                                      : dnnl::memory::format_tag::abc);
}

dnnl::memory GetBatchDotInputMemory(const NDArray& tensor,
                                    const bool transpose,
                                    const dnnl::memory::desc& target_desc) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto mem    = dnnl::memory(GetBatchDotMemoryDesc(tensor, transpose, false),
                             engine,
                             reinterpret_cast<void*>(tensor.data().dptr_));
  if (mem.get_desc() == target_desc) {
    return mem;
  }

  auto target_mem = TmpMemMgr::Get()->Alloc(target_desc);
  DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(mem, *target_mem),
                                      {{DNNL_ARG_FROM, mem}, {DNNL_ARG_TO, *target_mem}});
  return *target_mem;
}

}  // namespace

DNNLBatchDotFwd::DNNLBatchDotFwd(const DNNLDotParam& param,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<NDArray>& outputs) {
  auto lhs_shape = inputs[DotIn::lhs].shape();
  auto bigDim    = GetBatchDotBatchDim(lhs_shape);

  dnnl::memory::desc data_md    = GetBatchDotMemoryDesc(
      inputs[DotIn::lhs], param.transpose_a, !param.transpose_a);
  dnnl::memory::desc weights_md = GetBatchDotMemoryDesc(
      inputs[DotIn::rhs], param.transpose_b, !param.transpose_b);
  dnnl::memory::desc out_md(
      dnnl::memory::dims{bigDim, data_md.get_dims()[1], weights_md.get_dims()[2]},
      get_dnnl_type(outputs[DotOut::out].dtype()),
      dnnl::memory::format_tag::any);
  // v3: matmul ::desc removed; primitive_desc takes args directly.
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  if (param.quantized) {
    float out_scale = std::numeric_limits<float>::quiet_NaN();
    auto attrs = GetQuantizationAttributes(param, inputs, outputs, &out_scale);
    fwd_pd     = std::make_shared<batch_dot_fwd_pd_t>(
        engine, data_md, weights_md, out_md, attrs);
    if (!std::isnan(out_scale)) {
      // v3 runtime DNNL_ARG_DST scale memory matching attr.set_scales_mask
      // above. Built once at PD-create time and bound at Execute.
      dnnl::memory::desc scale_md(
          dnnl::memory::dims{1}, dnnl::memory::data_type::f32,
          dnnl::memory::format_tag::x);
      out_scale_mem = std::make_shared<dnnl::memory>(scale_md, engine);
      std::memcpy(out_scale_mem->get_data_handle(), &out_scale, sizeof(float));
    }
  } else {
    fwd_pd = std::make_shared<batch_dot_fwd_pd_t>(
        engine, data_md, weights_md, out_md);
  }

  fwd = std::make_shared<batch_dot_fwd_t>(*fwd_pd);
}

void DNNLBatchDotFwd::Execute(const OpContext& ctx,
                              const DNNLDotParam& param,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  auto lhs = inputs[DotIn::lhs];
  auto rhs = inputs[DotIn::rhs];
  // Created primitive descriptor assumes that both inputs are in default format
  if (lhs.IsDNNLData())
    lhs = lhs.Reorder2Default();
  if (rhs.IsDNNLData())
    rhs = rhs.Reorder2Default();

  TmpMemMgr::Get()->Init(ctx.requested[0]);

  auto lhs_mem = GetBatchDotInputMemory(lhs, param.transpose_a, fwd_pd->src_desc());
  auto rhs_mem = GetBatchDotInputMemory(rhs, param.transpose_b, fwd_pd->weights_desc());
  dnnl_output_t out_mem = CreateDNNLMem(
      outputs[DotOut::out], fwd_pd->dst_desc(), req[DotOut::out], &inputs[DotIn::lhs]);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, lhs_mem},
      {DNNL_ARG_WEIGHTS, rhs_mem},
      {DNNL_ARG_DST, *out_mem.second},
  };
  // v3 bind runtime DNNL_ARG_DST output scale for quantized batch_dot.
  if (out_scale_mem) {
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = *out_scale_mem;
  }

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[0], out_mem);
  DNNLStream::Get()->Submit();

  if (param.quantized && !param.enabled_float_output.has_value()) {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    float min_output;
    float max_output;
    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      min_output = param.min_calib_range.value();
      max_output = param.max_calib_range.value();
    } else {
      if (inputs[DotIn::lhs].dtype() == mshadow::kInt8) {
        mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
            s,
            1,
            &min_output,
            &max_output,
            inputs[DotIn::rhs_min].data().dptr<float>(),
            inputs[DotIn::rhs_max].data().dptr<float>(),
            inputs[DotIn::lhs_min].data().dptr<float>(),
            inputs[DotIn::lhs_max].data().dptr<float>());
      } else {
        mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
            s,
            1,
            &min_output,
            &max_output,
            inputs[DotIn::rhs_min].data().dptr<float>(),
            inputs[DotIn::rhs_max].data().dptr<float>(),
            inputs[DotIn::lhs_min].data().dptr<float>(),
            inputs[DotIn::lhs_max].data().dptr<float>());
      }
    }

    if (req[DotOut::out_min] != kNullOp) {
      AssignQuantizedRangeOutput(outputs[DotOut::out_min].data().dptr<float>(),
                                 &min_output,
                                 req[DotOut::out_min],
                                 "_sg_onednn_batch_dot");
    }
    if (req[DotOut::out_max] != kNullOp) {
      AssignQuantizedRangeOutput(outputs[DotOut::out_max].data().dptr<float>(),
                                 &max_output,
                                 req[DotOut::out_max],
                                 "_sg_onednn_batch_dot");
    }
  }
}

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::DNNLDotParam> {
  size_t operator()(const mxnet::op::DNNLDotParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.transpose_a);
    ret        = dmlc::HashCombine(ret, val.transpose_b);
    ret        = dmlc::HashCombine(ret, val.quantized);
    ret        = dmlc::HashCombine(ret, val.min_calib_range);
    ret        = dmlc::HashCombine(ret, val.max_calib_range);
    ret        = dmlc::HashCombine(ret, val.enabled_float_output);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_USE_ONEDNN == 1
