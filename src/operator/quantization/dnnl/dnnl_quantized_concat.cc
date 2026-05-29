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
 * \file quantized_concat.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1
#include <limits>

#include "operator/nn/dnnl/dnnl_concat-inl.h"
#include "operator/mxnet_op.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

namespace quantized_concat_enum {
enum QuantizedConcatOutputs { kOut, kMin, kMax };
}

static float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

static uint8_t QuantizeAffineUInt8(float real, float min, float max) {
  if (max <= min)
    return 0;
  const float scaled = floorf((real - min) * kUint8Range / (max - min) + 0.5f);
  const int32_t q =
      static_cast<int32_t>(std::min(kUint8Range, std::max(0.0f, scaled)));
  return static_cast<uint8_t>(q);
}

static bool DNNLQuantizedConcatAffineUInt8Fallback(const ConcatParam& param,
                                                  const std::vector<NDArray>& in_data,
                                                  const std::vector<OpReqType>& req,
                                                  const std::vector<NDArray>& out_data,
                                                  const std::vector<float>& data_min,
                                                  const std::vector<float>& data_max,
                                                  float output_min,
                                                  float output_max) {
  const auto out_dtype = out_data[quantized_concat_enum::kOut].dtype();
  if (out_dtype != mshadow::kUint8 && out_dtype != mshadow::kInt8) {
    return false;
  }

  bool needs_rescale = false;
  for (int i = 0; i < param.num_args; ++i) {
    if (in_data[i].dtype() != mshadow::kUint8 && in_data[i].dtype() != mshadow::kInt8) {
      return false;
    }
    needs_rescale = needs_rescale || in_data[i].dtype() != out_dtype ||
                    data_min[i] != output_min || data_max[i] != output_max;
  }
  if (!needs_rescale) {
    return false;
  }

  AssignQuantizedRangeOutput(
      out_data[quantized_concat_enum::kMin].data().dptr<float>(),
      &output_min,
      req[quantized_concat_enum::kMin],
      "quantized_concat");
  AssignQuantizedRangeOutput(
      out_data[quantized_concat_enum::kMax].data().dptr<float>(),
      &output_max,
      req[quantized_concat_enum::kMax],
      "quantized_concat");
  if (req[quantized_concat_enum::kOut] == kNullOp) {
    return true;
  }

  auto& out_arr = const_cast<NDArray&>(out_data[quantized_concat_enum::kOut]);
  out_arr.InvalidateDNNLData();

  std::vector<NDArray> inputs;
  inputs.reserve(param.num_args);
  for (int i = 0; i < param.num_args; ++i) {
    inputs.emplace_back(in_data[i].Reorder2Default());
  }

  const int param_dim = CheckAxis(param.dim.has_value() ? param.dim.value() : 0,
                                  inputs[0].shape().ndim());
  size_t before_axis = 1;
  for (int i = 0; i < param_dim; ++i) {
    before_axis *= inputs[0].shape()[i];
  }
  size_t after_axis = 1;
  for (int i = param_dim + 1; i < inputs[0].shape().ndim(); ++i) {
    after_axis *= inputs[0].shape()[i];
  }

  uint8_t* out_u8 = out_dtype == mshadow::kUint8 ?
                        out_arr.data().dptr<uint8_t>() :
                        nullptr;
  int8_t* out_s8 = out_dtype == mshadow::kInt8 ?
                       out_arr.data().dptr<int8_t>() :
                       nullptr;
  const float output_absmax = MaxAbs(output_min, output_max);
  size_t out_offset = 0;
  for (size_t prefix = 0; prefix < before_axis; ++prefix) {
    for (int input_idx = 0; input_idx < param.num_args; ++input_idx) {
      const size_t axis_size = inputs[input_idx].shape()[param_dim];
      const size_t block_size = axis_size * after_axis;
      const size_t in_offset = prefix * block_size;
      if (inputs[input_idx].dtype() == mshadow::kUint8) {
        const uint8_t* in = inputs[input_idx].data().dptr<uint8_t>();
        const float scale = (data_max[input_idx] - data_min[input_idx]) / kUint8Range;
        for (size_t j = 0; j < block_size; ++j) {
          const float real = static_cast<float>(in[in_offset + j]) * scale + data_min[input_idx];
          if (out_dtype == mshadow::kUint8) {
            const uint8_t q = QuantizeAffineUInt8(real, output_min, output_max);
            KERNEL_ASSIGN(out_u8[out_offset + j], req[quantized_concat_enum::kOut], q);
          } else {
            const float scaled = output_absmax > 0.0f ?
                                     floorf(std::min(std::abs(real) * kInt8Range / output_absmax,
                                                     kInt8Range) +
                                            0.5f) :
                                     0.0f;
            const int32_t q = static_cast<int32_t>(real < 0.0f ? -scaled : scaled);
            KERNEL_ASSIGN(out_s8[out_offset + j],
                          req[quantized_concat_enum::kOut],
                          static_cast<int8_t>(q));
          }
        }
      } else {
        const int8_t* in = inputs[input_idx].data().dptr<int8_t>();
        const float scale = MaxAbs(data_min[input_idx], data_max[input_idx]) / kInt8Range;
        for (size_t j = 0; j < block_size; ++j) {
          const float real = static_cast<float>(in[in_offset + j]) * scale;
          const float scaled = output_absmax > 0.0f ?
                                   floorf(std::min(std::abs(real) * kInt8Range / output_absmax,
                                                   kInt8Range) +
                                          0.5f) :
                                   0.0f;
          const int32_t q = static_cast<int32_t>(real < 0.0f ? -scaled : scaled);
          KERNEL_ASSIGN(out_s8[out_offset + j],
                        req[quantized_concat_enum::kOut],
                        static_cast<int8_t>(q));
        }
      }
      out_offset += block_size;
    }
  }
  return true;
}

static void DNNLQuantizedConcatForward(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& in_data,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& out_data) {
  // The op declares FResourceRequest::kTempSpace; bind it so we can
  // allocate the rescale destination and per-input scale buffers from
  // TmpMemMgr. Without this Init, the bare `dnnl::memory(desc, engine)`
  // ctor uses oneDNN's internal allocator, which we observed handing
  // back overlapping storage for back-to-back small allocations in this
  // op (symptom: int8 concat output with one input region zeroed —
  // tests/python/dnnl/subgraphs/test_conv_subgraph.py::
  // test_pos_single_concat_pos_neg[int8-data_shape1] failed with channels
  // 3-6 = 0 because the relu_out reorder's destination was clobbered
  // before the concat read it). Mirrors dnnl_quantized_batch_norm.cc.
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_data.size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_data.size(), 3U);
  // Collect data min/max and output_neg_min, output_pos_max
  std::vector<float> data_min(param_.num_args);
  std::vector<float> data_max(param_.num_args);
  float output_neg_min = std::numeric_limits<float>::max();
  float output_pos_max = std::numeric_limits<float>::lowest();
  for (int i = 0; i < param_.num_args; ++i) {
    data_min[i] = in_data[param_.num_args + 2 * i].data().dptr<float>()[0];
    if (data_min[i] < output_neg_min)
      output_neg_min = data_min[i];
    data_max[i] = in_data[param_.num_args + 2 * i + 1].data().dptr<float>()[0];
    if (data_max[i] > output_pos_max)
      output_pos_max = data_max[i];
  }
  const auto out_dtype = out_data[quantized_concat_enum::kOut].dtype();
  if (out_dtype != mshadow::kUint8) {
    output_neg_min = std::min(output_neg_min, 0.0f);
    output_pos_max = std::max(output_pos_max, 0.0f);
  }
  if (DNNLQuantizedConcatAffineUInt8Fallback(
          param_, in_data, req, out_data, data_min, data_max, output_neg_min, output_pos_max)) {
    return;
  }
  AssignQuantizedRangeOutput(out_data[quantized_concat_enum::kMin].data().dptr<float>(),
                             &output_neg_min,
                             req[quantized_concat_enum::kMin],
                             "quantized_concat");
  AssignQuantizedRangeOutput(out_data[quantized_concat_enum::kMax].data().dptr<float>(),
                             &output_pos_max,
                             req[quantized_concat_enum::kMax],
                             "quantized_concat");
  auto out_scale = GetScale(out_data[quantized_concat_enum::kOut], output_neg_min, output_pos_max);
  std::vector<dnnl::memory::desc> data_md;
  std::vector<const dnnl::memory*> data_mem;
  // Hold per-input f32 scale buffers so their backing memory outlives
  // DNNLStream::Submit(). Without this, the local `dnnl::memory` ctor
  // path would tie scale storage to oneDNN's internal allocator (see
  // note inside the loop below).
  std::vector<std::shared_ptr<float>> scale_bufs;
  scale_bufs.reserve(param_.num_args);
  for (int i = 0; i < param_.num_args; ++i) {
    auto i_scale = GetScale(in_data[i], data_min[i], data_max[i]);
    if (i_scale == out_scale) {
      CHECK(in_data[i].dtype() == out_dtype);
      auto mem = in_data[i].GetDNNLData();
      data_mem.push_back(mem);
      data_md.push_back(mem->get_desc());
    } else {
      auto mem      = in_data[i].GetDNNLData();
      auto mem_desc = mem->get_desc();
      if (in_data[i].dtype() != out_dtype) {
        // v3: CloneMemDescWithDtype takes the dnnl C++ enum directly.
        mem_desc = CloneMemDescWithDtype(mem_desc, get_dnnl_type(out_dtype));
      }
      auto cpu_engine = CpuEngine::Get()->get_engine();
      // Allocate rescale destination from TmpMemMgr (mirrors the working
      // dnnl_quantized_batch_norm.cc pattern). Engine-internal allocations
      // via `dnnl::memory(desc, engine)` gave overlapping storage across
      // back-to-back concat-input reorders, corrupting the second input's
      // data with zeros before the concat consumed it.
      auto rescaled_mem = TmpMemMgr::Get()->Alloc(mem_desc);
      if (in_data[i].dtype() != out_dtype) {
        // oneDNN v3 JIT reorder appears to skip the scaling stage for the
        // mixed-dtype case (u8 -> s8 with attr-scales): the destination
        // ends up with values consistent with an unscaled copy, leaving
        // one input's channels effectively zeroed in the concat output.
        // Both set_scales_mask(SRC,0) and set_scales_mask(DST,0) reproduce
        // the same failure on data_shape=(4,3,24,24). The same-dtype
        // s8->s8 sibling reorder works in the same loop with the same
        // attr layout. Sidestep by routing through f32: a u8->f32
        // dequantize then f32->s8 quantize. Both reorders use single-
        // tensor (DST mask=0) scales, which are exercised correctly by
        // other v3 paths in this codebase.
        //
        // v3 scale convention with only DST scale present:
        //   dst = src / dst_scale
        // Step 1 (u8 -> f32, dequantize): want dst_f32 = src_u8 / i_scale
        //   => dst_scale = i_scale.
        // Step 2 (f32 -> s8, quantize): want dst_s8 = src_f32 * out_scale
        //   => dst_scale = 1 / out_scale.
        // Composed:
        //   dst_s8 = (src_u8 / i_scale) * out_scale
        //          = src_u8 * (out_scale / i_scale)
        // which matches the math of the original single-reorder path.
        auto f32_desc =
            CloneMemDescWithDtype(mem->get_desc(),
                                  dnnl::memory::data_type::f32);
        auto f32_mem = TmpMemMgr::Get()->Alloc(f32_desc);

        // Step 1: u8 -> f32 (dequantize) with DST scale = i_scale.
        dnnl::primitive_attr deq_attr;
        deq_attr.set_scales_mask(DNNL_ARG_DST, 0);
        const auto deq_pd = dnnl::reorder::primitive_desc(
            cpu_engine, mem->get_desc(), cpu_engine, f32_desc, deq_attr);
        auto deq_scale_buf = std::make_shared<float>(i_scale);
        scale_bufs.push_back(deq_scale_buf);
        dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::x);
        auto deq_scale_mem =
            dnnl::memory(scale_md, cpu_engine, deq_scale_buf.get());
        dnnl_args_map_t deq_args;
        deq_args[DNNL_ARG_SRC] = *mem;
        deq_args[DNNL_ARG_DST] = *f32_mem;
        deq_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = deq_scale_mem;
        DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(deq_pd), deq_args);

        // Step 2: f32 -> s8 (quantize) with DST scale = 1 / out_scale.
        dnnl::primitive_attr q_attr;
        q_attr.set_scales_mask(DNNL_ARG_DST, 0);
        const auto q_pd = dnnl::reorder::primitive_desc(
            cpu_engine, f32_desc, cpu_engine, mem_desc, q_attr);
        auto q_scale_buf = std::make_shared<float>(1.0f / out_scale);
        scale_bufs.push_back(q_scale_buf);
        auto q_scale_mem =
            dnnl::memory(scale_md, cpu_engine, q_scale_buf.get());
        dnnl_args_map_t q_args;
        q_args[DNNL_ARG_SRC] = *f32_mem;
        q_args[DNNL_ARG_DST] = *rescaled_mem;
        q_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = q_scale_mem;
        DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(q_pd), q_args);
      } else {
        // Same-dtype rescale (e.g. s8 -> s8). v3: set_output_scales removed;
        // use set_scales_mask + runtime arg. Direct single reorder works
        // for the same-dtype case.
        dnnl::primitive_attr reorder_attr;
        reorder_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        const auto reorder_pd = dnnl::reorder::primitive_desc(
            cpu_engine, mem->get_desc(), cpu_engine, mem_desc, reorder_attr);
        // Scale memory: keep the f32 storage user-managed and held alive by
        // `scale_bufs` until DNNLStream::Submit() runs.
        auto scale_buf = std::make_shared<float>(out_scale / i_scale);
        scale_bufs.push_back(scale_buf);
        dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::x);
        auto scale_mem = dnnl::memory(scale_md, cpu_engine, scale_buf.get());
        dnnl_args_map_t reorder_args;
        reorder_args[DNNL_ARG_SRC] = *mem;
        reorder_args[DNNL_ARG_DST] = *rescaled_mem;
        reorder_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = scale_mem;
        DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), reorder_args);
      }
      data_mem.push_back(rescaled_mem);
      data_md.push_back(mem_desc);
    }
  }
  int param_dim                = param_.dim.has_value() ? param_.dim.value() : 0;
  param_dim                    = CheckAxis(param_dim, in_data[concat_enum::kData0].shape().ndim());
  DNNLConcatFwd& fwd           = DNNLConcatFwd::GetCached(param_dim, in_data, data_md);
  mxnet::dnnl_output_t out_mem = CreateDNNLMem(
      out_data[quantized_concat_enum::kOut], fwd.fwd_pd.dst_desc(), req[concat_enum::kOut]);
  dnnl_args_map_t net_args;
  net_args[DNNL_ARG_DST] = *out_mem.second;
  for (int i = 0; i < param_.num_args; i++) {
    net_args[DNNL_ARG_MULTIPLE_SRC + i] = *data_mem[i];
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

inline static bool ConcatStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_attrs->size(), 3U);

  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLQuantizedOps(), dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_concat)
    .set_attr<FInferStorageType>("FInferStorageType", ConcatStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedConcatForward)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
