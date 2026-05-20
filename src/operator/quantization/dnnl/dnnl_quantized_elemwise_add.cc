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
 * \file dnnl_quantized_elemwise_add.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/quantized_elemwise_add-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuantizeElemwiseAddParam);

class DNNLQuantizedSumFwd {
 public:
  dnnl::sum::primitive_desc fwd_pd;

  DNNLQuantizedSumFwd(const dnnl::memory::desc& output_md,
                      const std::vector<float>& scales,
                      const std::vector<dnnl::memory::desc>& inputs_md)
      // v3: sum::primitive_desc(engine, dst_md, scales, src_mds, attr={}).
      : fwd_pd(CpuEngine::Get()->get_engine(), output_md, scales, inputs_md) {
    fwd_ = std::make_shared<dnnl::sum>(fwd_pd);
  }

  const dnnl::sum& GetFwd() const {
    return *fwd_;
  }

  static DNNLQuantizedSumFwd& GetCached(const dnnl::memory::desc& output_md,
                                        const std::vector<float>& scales,
                                        const std::vector<dnnl::memory::desc>& inputs_md);

 private:
  std::shared_ptr<dnnl::sum> fwd_;
  std::shared_ptr<dnnl::memory> out_;
};

DNNLQuantizedSumFwd& DNNLQuantizedSumFwd::GetCached(
    const dnnl::memory::desc& output_md,
    const std::vector<float>& scales,
    const std::vector<dnnl::memory::desc>& inputs_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, DNNLQuantizedSumFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, DNNLQuantizedSumFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(output_md);
  key.AddSign(scales);
  key.AddSign(inputs_md);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLQuantizedSumFwd fwd(output_md, scales, inputs_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

class DNNLQuantizedBinAddFwd {
 public:
  dnnl::binary::primitive_desc fwd_pd;

  DNNLQuantizedBinAddFwd(const dnnl::memory::desc& output_md,
                         const std::vector<float>& scales,
                         const std::vector<dnnl::memory::desc>& inputs_md) {
    // v3: ::desc removed; binary primitive_desc takes args directly.
    //     set_scales(arg, mask, values) became set_scales_mask(arg, mask)
    //     with the value bound at execute time as a runtime memory arg.
    dnnl::primitive_attr input_scales;
    input_scales.set_scales_mask(DNNL_ARG_SRC_0, 0);
    input_scales.set_scales_mask(DNNL_ARG_SRC_1, 0);
    fwd_pd = dnnl::binary::primitive_desc(
        CpuEngine::Get()->get_engine(), dnnl::algorithm::binary_add,
        inputs_md[0], inputs_md[1], output_md, input_scales);
    fwd_   = std::make_shared<dnnl::binary>(fwd_pd);
    // Pre-allocate scale memories so callers can bind them at execute time.
    dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::x);
    src0_scale_ = dnnl::memory(scale_md, CpuEngine::Get()->get_engine());
    src1_scale_ = dnnl::memory(scale_md, CpuEngine::Get()->get_engine());
    *reinterpret_cast<float*>(src0_scale_.get_data_handle()) = scales[0];
    *reinterpret_cast<float*>(src1_scale_.get_data_handle()) = scales[1];
  }

  const dnnl::memory& GetSrc0Scale() const { return src0_scale_; }
  const dnnl::memory& GetSrc1Scale() const { return src1_scale_; }

  const dnnl::binary& GetFwd() const {
    return *fwd_;
  }

  static DNNLQuantizedBinAddFwd& GetCached(const dnnl::memory::desc& output_md,
                                           const std::vector<float>& scales,
                                           const std::vector<dnnl::memory::desc>& inputs_md);

 private:
  std::shared_ptr<dnnl::binary> fwd_;
  std::shared_ptr<dnnl::memory> out_;
  // v3: per-input scale runtime memories.
  dnnl::memory src0_scale_;
  dnnl::memory src1_scale_;
};

DNNLQuantizedBinAddFwd& DNNLQuantizedBinAddFwd::GetCached(
    const dnnl::memory::desc& output_md,
    const std::vector<float>& scales,
    const std::vector<dnnl::memory::desc>& inputs_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, DNNLQuantizedBinAddFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, DNNLQuantizedBinAddFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(output_md);
  key.AddSign(scales);
  key.AddSign(inputs_md);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLQuantizedBinAddFwd fwd(output_md, scales, inputs_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

static void DNNLQuantizedElemwiseAddForward(const nnvm::NodeAttrs& attrs,
                                            const OpContext& ctx,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<NDArray>& outputs) {
  const QuantizeElemwiseAddParam& params = nnvm::get<QuantizeElemwiseAddParam>(attrs.parsed);
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(inputs.size(), 6U) << "should be A, B, A_min, A_max, B_min, B_max";
  // C, C_min, C_max
  CHECK_EQ(outputs.size(), 3U) << "should be C, C_min, C_max";
  // Collect data min,max,absmax
  const float A_min        = inputs[q_elemwise_add::kAMin].data().dptr<float>()[0];
  const float B_min        = inputs[q_elemwise_add::kBMin].data().dptr<float>()[0];
  const float A_max        = inputs[q_elemwise_add::kAMax].data().dptr<float>()[0];
  const float B_max        = inputs[q_elemwise_add::kBMax].data().dptr<float>()[0];
  const bool is_A_int8     = (inputs[q_elemwise_add::kDataA].dtype() == mshadow::kInt8);
  const bool is_B_int8     = (inputs[q_elemwise_add::kDataB].dtype() == mshadow::kInt8);
  const float A_type_range = is_A_int8 ? kInt8Range : kUint8Range;
  const float B_type_range = is_B_int8 ? kInt8Range : kUint8Range;
  const float A_absmax     = MaxAbs(A_min, A_max);
  const float B_absmax     = MaxAbs(B_min, B_max);
  const float A_scale      = A_type_range / A_absmax;
  const float B_scale      = B_type_range / B_absmax;
  auto A_mem               = inputs[q_elemwise_add::kDataA].GetDNNLData();
  auto B_mem               = inputs[q_elemwise_add::kDataB].GetDNNLData();
  bool diff_in_types       = (is_A_int8 != is_B_int8);
  assert(diff_in_types ==
         (inputs[q_elemwise_add::kDataA].dtype() != inputs[q_elemwise_add::kDataB].dtype()));
  dnnl::memory* rescaled_mem;              // rescaled_mem is for reorder dnnl memory
  double output_data_range = kInt32Range;  // output default set as int32

  if (outputs[q_elemwise_add::kOut].dtype() == mshadow::kInt8) {
    output_data_range = kInt8Range;
  } else if (outputs[q_elemwise_add::kOut].dtype() == mshadow::kUint8) {
    output_data_range = kUint8Range;
  }

  float output_min   = 0;
  float output_max   = 0;
  float output_scale = 0;
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    output_min   = params.min_calib_range.value();
    output_max   = params.max_calib_range.value();
    output_scale = output_data_range / MaxAbs(output_min, output_max);
  } else {
    output_max   = A_absmax + B_absmax;
    output_min   = -output_max;
    output_scale = output_data_range / output_max;
  }

  std::vector<float> scales(2);  // 2: scale 0 for input A, scale 1 for input B
  scales[0] = output_scale / A_scale;
  scales[1] = output_scale / B_scale;

  // We can use more efficient sum kernel when there is no broadcast - when shapes are the same
  const bool sum_kernel =
      (inputs[q_elemwise_add::kDataA].shape() == inputs[q_elemwise_add::kDataB].shape());

  if (diff_in_types) {
    if (sum_kernel) {
      // rescale uint8 to int8 by reorder to temporary memory
      auto s8_desc               = is_A_int8 ? A_mem->get_desc() : B_mem->get_desc();
      rescaled_mem               = TmpMemMgr::Get()->Alloc(s8_desc);
      // S4: u8∈[0,255] (range 255) → s8∈[-128,127] (range 255), shift+scale
      // simplifies under symmetric-around-0 quantization to a pure 0.5 scale;
      // the corresponding range fixup happens at the dequant scale below.
      // Mirrors dnnl_fc.cc's u8_to_s8_scale constant.
      const float u8_to_s8_scale = 0.5;
      auto engine                = CpuEngine::Get()->get_engine();
      // v3: set_output_scales removed; bind runtime scale tensor.
      dnnl::primitive_attr reorder_attr;
      // v3 reorder: DNNL_ARG_DST divides; use DNNL_ARG_SRC to multiply by u8_to_s8_scale.
      reorder_attr.set_scales_mask(DNNL_ARG_SRC, 0);
      auto u8_mem = (is_A_int8 == true) ? B_mem : A_mem;
      const auto reorder_pd =
          dnnl::reorder::primitive_desc(engine, u8_mem->get_desc(), engine, s8_desc, reorder_attr);
      dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                                  dnnl::memory::format_tag::x);
      auto scale_mem = dnnl::memory(scale_md, engine);
      *reinterpret_cast<float*>(scale_mem.get_data_handle()) = u8_to_s8_scale;
      dnnl_args_map_t args({{DNNL_ARG_FROM, *u8_mem},
                            {DNNL_ARG_TO, *rescaled_mem},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem}});
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), args);
      // Modify scale to restore original uint8 values:
      if (is_A_int8) {
        B_mem = rescaled_mem;
        scales[1] *= 1.0 / u8_to_s8_scale;
      } else {
        A_mem = rescaled_mem;
        scales[0] *= 1.0 / u8_to_s8_scale;
      }
    }
  }

  std::vector<dnnl::memory::desc> in_desc;
  in_desc.push_back(A_mem->get_desc());
  in_desc.push_back(B_mem->get_desc());

  dnnl_output_t out_mem;
  auto output_md     = outputs[q_elemwise_add::kOut].GetDNNLData()->get_desc();
  DNNLStream* stream = DNNLStream::Get();

  if (sum_kernel) {
    const auto& fwd = DNNLQuantizedSumFwd::GetCached(output_md, scales, in_desc);
    out_mem         = CreateDNNLMem(outputs[q_elemwise_add::kOut],
                            fwd.fwd_pd.dst_desc(),
                            req[q_elemwise_add::kOut],
                            &inputs[q_elemwise_add::kDataA]);
    const dnnl_args_map_t args({{DNNL_ARG_MULTIPLE_SRC, *A_mem},
                                {DNNL_ARG_MULTIPLE_SRC + 1, *B_mem},
                                {DNNL_ARG_DST, *out_mem.second}});
    stream->RegisterPrimArgs(fwd.GetFwd(), args);
  } else {
    const auto& fwd = DNNLQuantizedBinAddFwd::GetCached(output_md, scales, in_desc);
    const auto potentially_inplace_input =
        (outputs[q_elemwise_add::kOut].GetDNNLData()->get_data_handle() ==
         inputs[q_elemwise_add::kDataB].GetDNNLData()->get_data_handle()) ?
            q_elemwise_add::kDataB :
            q_elemwise_add::kDataA;
    out_mem = CreateDNNLMem(outputs[q_elemwise_add::kOut],
                            fwd.fwd_pd.dst_desc(),
                            req[q_elemwise_add::kOut],
                            &inputs[potentially_inplace_input]);

    // v3: per-input scales were declared on the primitive via set_scales_mask;
    // their values MUST be bound at execute time as runtime memory args, otherwise
    // oneDNN dereferences an uninitialized scale pointer and segfaults inside
    // ref_binary_t::execute_ref.
    const dnnl_args_map_t args(
        {{DNNL_ARG_SRC_0, *A_mem},
         {DNNL_ARG_SRC_1, *B_mem},
         {DNNL_ARG_DST, *out_mem.second},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, fwd.GetSrc0Scale()},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, fwd.GetSrc1Scale()}});
    stream->RegisterPrimArgs(fwd.GetFwd(), args);
  }
  CommitOutput(outputs[q_elemwise_add::kOut], out_mem);
  stream->Submit();

  outputs[q_elemwise_add::kMin].data().dptr<float>()[0] = output_min;
  outputs[q_elemwise_add::kMax].data().dptr<float>()[0] = output_max;
}

inline static bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  // Check num of inputs: A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_attrs->size(), 6U);
  // Check num of outputs: C, C_min, C_max
  CHECK_EQ(out_attrs->size(), 3U);

  return DNNLStorageType(
      attrs, dev_mask, SupportDNNLQuantizedOps(), dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_elemwise_add)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedElemwiseAddForward)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr_parser(ParamParser<QuantizeElemwiseAddParam>)
    .add_arguments(QuantizeElemwiseAddParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_quantized_npi_add)
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedElemwiseAddForward)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr_parser(ParamParser<QuantizeElemwiseAddParam>)
    .add_arguments(QuantizeElemwiseAddParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
