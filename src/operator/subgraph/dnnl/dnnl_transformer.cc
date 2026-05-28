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

#include <string>
#include <utility>
#include <vector>

#include "operator/contrib/transformer-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/tensor/elemwise_unary_op.h"
#include "operator/subgraph/common.h"
#include "dnnl_transformer-inl.h"

// 3 tensors within one (queries keys values)
#define QKV_NUM 3

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLSelfAttParam);

static bool SgDNNLSelfAttSupportsBackward(const DNNLSelfAttParam& param) {
  if (!param.quantized) {
    return !param.enabled_float_output.has_value() ||
           param.enabled_float_output.value() == mshadow::kFloat32;
  }
  return param.enabled_float_output.has_value() &&
         param.enabled_float_output.value() == mshadow::kFloat32;
}

static NDArray SelfAttToDefault(const NDArray& data) {
  return data.IsDNNLData() ? data.Reorder2Default() : data;
}

static NDArray DequantizeSelfAttTensorCPU(const NDArray& data,
                                          const NDArray& data_min,
                                          const NDArray& data_max,
                                          const char* op_name) {
  const NDArray default_data = SelfAttToDefault(data);
  CHECK(default_data.dtype() == mshadow::kInt8 || default_data.dtype() == mshadow::kUint8)
      << op_name << " expects int8/uint8 quantized input, got " << default_data.dtype();
  NDArray ret(default_data.shape(), default_data.ctx(), false, mshadow::kFloat32);
  const float min_range = data_min.data().dptr<float>()[0];
  const float max_range = data_max.data().dptr<float>()[0];
  float* out            = ret.data().dptr<float>();
  const size_t size     = default_data.shape().Size();
  const int nthreads    = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  if (default_data.dtype() == mshadow::kInt8) {
    const int8_t* in = default_data.data().dptr<int8_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = QuantizedToFloat<int8_t>(in[i], min_range, max_range);
    }
  } else {
    const uint8_t* in = default_data.data().dptr<uint8_t>();
    const float scale = (max_range - min_range) / 255.0f;
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]) * scale + min_range;
    }
  }
  return ret;
}

static NDArray FloatSelfAttTensorCPU(const NDArray& data,
                                     const NDArray* data_min,
                                     const NDArray* data_max,
                                     const char* op_name) {
  const NDArray default_data = SelfAttToDefault(data);
  if (default_data.dtype() == mshadow::kFloat32) {
    return default_data;
  }
  CHECK(data_min != nullptr && data_max != nullptr)
      << op_name << " received quantized input without min/max ranges";
  return DequantizeSelfAttTensorCPU(default_data, *data_min, *data_max, op_name);
}

static NDArray CastSelfAttGradToFloatCPU(const NDArray& grad, const char* op_name) {
  const NDArray default_grad = SelfAttToDefault(grad);
  if (default_grad.dtype() == mshadow::kFloat32) {
    return default_grad;
  }
  CHECK(default_grad.dtype() == mshadow::kInt8 || default_grad.dtype() == mshadow::kUint8 ||
        default_grad.dtype() == mshadow::kInt32)
      << op_name << " expects float32/int8/uint8/int32 output gradients, got "
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
  } else if (default_grad.dtype() == mshadow::kUint8) {
    const uint8_t* in = default_grad.data().dptr<uint8_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]);
    }
  } else {
    const int32_t* in = default_grad.data().dptr<int32_t>();
#pragma omp parallel for num_threads(nthreads)
    for (index_t i = 0; i < static_cast<index_t>(size); ++i) {
      out[i] = static_cast<float>(in[i]);
    }
  }
  return ret;
}

template <typename DType>
static void ZeroSelfAttOutputCPUImpl(const NDArray& out) {
  DType* ptr       = out.data().dptr<DType>();
  const size_t len = out.shape().Size();
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < static_cast<index_t>(len); ++i) {
    ptr[i] = DType(0);
  }
}

static void PrepareDefaultSelfAttOutputCPU(const NDArray& out, OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  const_cast<NDArray&>(out).InvalidateDNNLData();
  (void)out.GetDNNLData();
}

static void ZeroLikeSelfAttOutputCPU(const NDArray& out, OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  if (req == kAddTo) {
    PrepareDefaultSelfAttOutputCPU(out, req);
    return;
  }
  CHECK(req == kWriteTo || req == kWriteInplace)
      << "Unsupported zero-gradient request type " << req;
  PrepareDefaultSelfAttOutputCPU(out, req);
  MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
    ZeroSelfAttOutputCPUImpl<DType>(out);
  });
}

static void SelfAttRowMajorSgemm(bool trans_a,
                                 bool trans_b,
                                 index_t m,
                                 index_t n,
                                 index_t k,
                                 float alpha,
                                 const float* a,
                                 index_t lda,
                                 const float* b,
                                 index_t ldb,
                                 float beta,
                                 float* c,
                                 index_t ldc) {
#if (MSHADOW_USE_CBLAS == 1 || MSHADOW_USE_MKL == 1)
  cblas_sgemm(CblasRowMajor,
              trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans,
              m,
              n,
              k,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
#else
  for (index_t row = 0; row < m; ++row) {
    for (index_t col = 0; col < n; ++col) {
      float acc = 0.f;
      for (index_t inner = 0; inner < k; ++inner) {
        const float av = trans_a ? a[inner * lda + row] : a[row * lda + inner];
        const float bv = trans_b ? b[col * ldb + inner] : b[inner * ldb + col];
        acc += av * bv;
      }
      c[row * ldc + col] = alpha * acc + beta * c[row * ldc + col];
    }
  }
#endif
}

template <bool with_split>
static bool SgDNNLSelfAttShape(const NodeAttrs& attrs,
                               mxnet::ShapeVector* in_shape,
                               mxnet::ShapeVector* out_shape) {
  const auto& params        = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  unsigned int in_shape_num = 1;
  auto in_shape_0           = in_shape->at(0);
  auto in_shape_1           = in_shape_0;  // with split there is only one input
  CHECK_EQ(in_shape_0.ndim(), 3U)
      << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
      << "but the given tensor is " << in_shape_0.ndim() << "D";

  if constexpr (!with_split) {
    in_shape_1 = in_shape->at(1);  // without split we need to consider 2nd input
    CHECK_EQ(in_shape_1.ndim(), 3U)
        << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
        << "but the given tensor is " << in_shape_1.ndim() << "D";
    CHECK_EQ(in_shape_0[0], in_shape_1[0]);
    CHECK_EQ(in_shape_0[2], in_shape_1[2]);
    in_shape_num = 2;
  }

  if (params.quantized) {
    CHECK_EQ(in_shape->size(), 3 * in_shape_num)
        << "Input: [queries_keys_values, min_qkv, max_qkv] "
        << "- currently have " << in_shape->size() << " inputs";
    if constexpr (with_split) {
      SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape({1}));
    } else {
      SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 3, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 4, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 5, mxnet::TShape({1}));
    }

    if (!params.enabled_float_output.has_value()) {
      out_shape->resize(3);
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));  // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));  // max output
    } else {
      out_shape->resize(1);
    }
  } else {
    CHECK_EQ(in_shape->size(), in_shape_num)
        << "Input:[queries_keys_values] - currently have " << in_shape->size() << " inputs";
    out_shape->resize(1);
  }

  SHAPE_ASSIGN_CHECK(
      *out_shape, 0, mxnet::TShape({in_shape_0[0], params.heads, in_shape_0[1], in_shape_1[1]}));
  return true;
}

template <bool with_split>
static bool SgDNNLSelfAttQKInferType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int>* in_types,
                                     std::vector<int>* out_types) {
  const auto& params        = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  unsigned int in_shape_num = 1;
  if constexpr (!with_split) {
    CHECK_EQ(in_types->at(0), in_types->at(1));
    in_shape_num = 2;
  }
  if (params.quantized) {
    CHECK_EQ(in_types->size(), 3 * in_shape_num);

    if (in_types->at(0) == mshadow::kBfloat16) {
      return false;
    }

    CHECK(in_types->at(0) == mshadow::kInt8)
        << "QuantizedSelfAttentionQK only supports int8 input, while " << in_types->at(0)
        << " is given.";

    if constexpr (with_split) {
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
    } else {
      TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 4, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 5, mshadow::kFloat32);
    }

    if (params.enabled_float_output.has_value()) {
      CHECK_EQ(out_types->size(), 1U);
      TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
    } else {
      CHECK_EQ(out_types->size(), 3U);
      if (params.min_calib_range.has_value() && params.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
  } else {
    CHECK_EQ(in_types->size(), in_shape_num);
    CHECK_EQ(out_types->size(), 1U);
    if (in_types->at(0) == mshadow::kFloat32) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
      if constexpr (!with_split) {
        TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else if (in_types->at(0) == mshadow::kBfloat16) {
      if constexpr (!with_split) {
        TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kBfloat16);
      }
      if (params.enabled_float_output.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kBfloat16);
      }
    } else {
      CHECK_EQ(in_types->at(0), -1);
      return false;
    }
  }

  return true;
}

class SgDNNLSelfAttQKOp {
 public:
  explicit SgDNNLSelfAttQKOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<DNNLSelfAttParam>(attrs.parsed)) {}

  template <bool with_split>
  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               bool already_prepared);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph oneDNN self attention qk only supports "
                  "inference computation.";
  }

  template <bool with_split>
  void Initialize(const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  DNNLSelfAttParam param_;
  dnnl_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_query_mem_;
  std::shared_ptr<dnnl::memory> cached_key_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  // v3: runtime scale tensor for set_scales_mask matmul attr.
  std::shared_ptr<dnnl::memory> cached_scale_mem_;
  // F4/F11: default-init so non-quantized paths don't observe garbage if a
  // future caller drops the param_.quantized guard around the read sites.
  float min_data_0_{0.0f};
  float max_data_0_{0.0f};
  float min_data_1_{0.0f};
  float max_data_1_{0.0f};
  float min_output_{0.0f};
  float max_output_{0.0f};
  float data_scale_0_{0.0f};
  float data_scale_1_{0.0f};
};

static OpStatePtr CreateSgDNNLSelfAttQKState(const nnvm::NodeAttrs& attrs,
                                             Context ctx,
                                             const mxnet::ShapeVector& in_shapes,
                                             const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgDNNLSelfAttQKOp>(attrs);
}

template <bool with_split>
void SgDNNLSelfAttQKForward(const OpStatePtr& state_pointer,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  // XOP19: gate before any output write so the caller's sentinel buffer is
  // observable on kNullOp / kAddTo.  The op binds the primary output via
  // `set_data_handle(outputs[0])` and the QK matmul writes there in place,
  // so neither kNullOp nor kAddTo can be honored without a tmp accumulation
  // buffer.  Reject loudly.
  if (req[0] == kNullOp)
    return;
  CHECK_NE(req[0], kAddTo)
      << "kAddTo is not supported for the primary output of _sg_onednn_selfatt_qk";
  SgDNNLSelfAttQKOp& op = state_pointer.get_state<SgDNNLSelfAttQKOp>();
  // AMP / oneDNN v3 fallback: on CPU ISAs without native bf16 (e.g. AVX2),
  // oneDNN v3 has no bf16 matmul kernel, so the QK matmul primitive_desc
  // creation fails inside Initialize().  Promote bf16 inputs/outputs to fp32
  // for the duration of this call and reorder the fp32 result back into the
  // caller's bf16 output.  The cached state on `op` is then keyed against
  // f32 buffers so subsequent calls (also bf16-input on AVX2) reuse it.
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
          f32_out.emplace_back(nd.shape(), nd.ctx(), /*delay_alloc=*/false,
                               static_cast<int>(mshadow::kFloat32));
          out_was_bf16.push_back(true);
        } else {
          f32_out.emplace_back(nd);
          out_was_bf16.push_back(false);
        }
      }
      std::vector<OpReqType> f32_req;
      f32_req.reserve(req.size());
      for (size_t i = 0; i < req.size(); ++i) {
        f32_req.push_back((i < out_was_bf16.size() && out_was_bf16[i]) ? kWriteTo : req[i]);
      }
      SgDNNLSelfAttQKForward<with_split>(state_pointer, ctx, f32_in, f32_req, f32_out);
      DNNLStream::Get()->Submit();
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!out_was_bf16[i]) continue;
        if (req[i] == kNullOp) continue;
        CHECK_NE(req[i], kAddTo)
            << "kAddTo not supported for BF16 fallback path on output " << i
            << " of _sg_onednn_selfatt_qk";
        auto src_mem = f32_out[i].GetDNNLData();
        auto dst_mem = outputs[i].GetDNNLData();
        ReorderTo(src_mem, dst_mem);
      }
      return;
    }
  }
  bool already_prepared = false;
  if (!op.IsInitialized()) {
    op.Initialize<with_split>(ctx, inputs, req, outputs);
    already_prepared = true;
  }
  op.Forward<with_split>(ctx, inputs, req, outputs, already_prepared);
}

static bool SgDNNLSelfAttStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  const DNNLSelfAttParam& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  const bool support = !param.quantized || SupportDNNLQuantizedOps();
  return DNNLStorageType(attrs, dev_mask, support, dispatch_mode, in_attrs, out_attrs);
}

template <bool with_split>
void SgDNNLSelfAttQKOp::Initialize(const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  using namespace dnnl;

  const auto in_tensor_0 = inputs[0];
  auto in_tensor_1       = in_tensor_0;  // with split there is only one input
  const auto out_tensor  = outputs[0];

  const auto in_dtype = get_dnnl_type(in_tensor_0.dtype());

  const memory::dim heads          = param_.heads;
  const memory::dim sequences      = in_tensor_0.shape()[0];
  const memory::dim qkv_seq_len_0  = in_tensor_0.shape()[1];
  const memory::dim output_lin_dim = in_tensor_0.shape()[2];
  memory::dim embed_dim            = output_lin_dim;
  if constexpr (with_split) {
    embed_dim /= QKV_NUM;
  } else {
    in_tensor_1 = inputs[1];  // without split we need to consider 2nd input
  }
  const memory::dim qkv_seq_len_1  = in_tensor_1.shape()[1];
  const memory::dim head_dim       = embed_dim / heads;
  const memory::dim batch_stride_0 = output_lin_dim * qkv_seq_len_0;
  const memory::dim batch_stride_1 = output_lin_dim * qkv_seq_len_1;

  float min_data = 0.0f;
  float max_data = 0.0f;

  const auto engine = CpuEngine::Get()->get_engine();

  memory::dims query_dims    = {sequences, heads, qkv_seq_len_0, head_dim};
  memory::dims key_dims      = {sequences, heads, head_dim, qkv_seq_len_1};
  memory::dims query_strides = {batch_stride_0, head_dim, output_lin_dim, 1};
  memory::dims key_strides   = {batch_stride_1, head_dim, 1, output_lin_dim};

  auto query_md = memory::desc(query_dims, in_dtype, query_strides);
  auto key_md   = memory::desc(key_dims, in_dtype, key_strides);

  float oscale = 1.0f;
  if (param_.quantized) {
    if constexpr (with_split) {
      min_data_0_   = inputs[1].data().dptr<float>()[0];
      max_data_0_   = inputs[2].data().dptr<float>()[0];
      data_scale_0_ = data_scale_1_ =
          GetQuantizeScale(in_tensor_0.dtype(), min_data_0_, max_data_0_);
    } else {
      min_data_0_   = inputs[2].data().dptr<float>()[0];
      max_data_0_   = inputs[3].data().dptr<float>()[0];
      min_data_1_   = inputs[4].data().dptr<float>()[0];
      max_data_1_   = inputs[5].data().dptr<float>()[0];
      data_scale_0_ = GetQuantizeScale(in_tensor_0.dtype(), min_data_0_, max_data_0_);
      data_scale_1_ = GetQuantizeScale(in_tensor_1.dtype(), min_data_1_, max_data_1_);
    }

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();
      oscale      = (data_scale_0_ * data_scale_1_) /
               GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_);
    } else if (param_.enabled_float_output.has_value()) {
      oscale = data_scale_0_ * data_scale_1_;
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_data, &max_data, &min_data, &max_data);
    }
  }

  // v3: set_output_scales / matmul::desc removed; bind scale at execute time.
  dnnl::primitive_attr attr;
  attr.set_scales_mask(DNNL_ARG_DST, 0);
  auto matmul_pd = matmul::primitive_desc(
      engine, query_md, key_md, GetMemDesc(out_tensor), attr);
  fwd_           = std::make_shared<matmul>(matmul_pd);

  // Pre-build the runtime scale memory.
  dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                              dnnl::memory::format_tag::x);
  cached_scale_mem_ = std::make_shared<memory>(scale_md, engine);
  *reinterpret_cast<float*>(cached_scale_mem_->get_data_handle()) = oscale;

  MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
    DType* query_mem_ptr = inputs[0].data().dptr<DType>();
    DType* key_mem_ptr;
    if constexpr (with_split) {
      key_mem_ptr = query_mem_ptr + embed_dim;
    } else {
      key_mem_ptr = inputs[1].data().dptr<DType>();
    }
    cached_query_mem_ = std::make_shared<memory>(query_md, engine, query_mem_ptr);
    cached_key_mem_   = std::make_shared<memory>(key_md, engine, key_mem_ptr);
  });

  MSHADOW_TYPE_SWITCH(out_tensor.dtype(), DType, {
    cached_out_mem_ =
        std::make_shared<memory>(matmul_pd.dst_desc(), engine, out_tensor.data().dptr<DType>());
  });

  args_[DNNL_ARG_SRC]                        = *cached_query_mem_;
  args_[DNNL_ARG_WEIGHTS]                    = *cached_key_mem_;
  args_[DNNL_ARG_DST]                        = *cached_out_mem_;
  args_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = *cached_scale_mem_;
  initialized_                               = true;
}

template <bool with_split>
void SgDNNLSelfAttQKOp::Forward(const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs,
                                bool already_prepared) {
  if (!already_prepared) {
    const size_t output_lin_dim = inputs[0].shape()[2];
    const size_t embed_dim      = output_lin_dim / QKV_NUM;

    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      DType* query_mem_ptr = inputs[0].data().dptr<DType>();
      DType* key_mem_ptr;
      if constexpr (with_split) {
        key_mem_ptr = query_mem_ptr + embed_dim;
      } else {
        key_mem_ptr = inputs[1].data().dptr<DType>();
      }
      cached_query_mem_->set_data_handle(query_mem_ptr);
      cached_key_mem_->set_data_handle(key_mem_ptr);
    });

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_->set_data_handle(outputs[0].data().dptr<DType>());
    });
  }
  DNNLStream::Get()->RegisterPrimArgs(*fwd_, args_);
  DNNLStream::Get()->Submit();

  if (param_.quantized && !param_.enabled_float_output.has_value()) {
    AssignQuantizedRangeOutput(outputs[1].data().dptr<float>(), &min_output_,
                               req[1], "_sg_onednn_selfatt_qk");
    AssignQuantizedRangeOutput(outputs[2].data().dptr<float>(), &max_output_,
                               req[2], "_sg_onednn_selfatt_qk");
  }
}

template <bool with_split>
nnvm::ObjectPtr SgDNNLSelfAttQKQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const& param    = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  if constexpr (with_split) {
    node->attrs.op = Op::Get("_sg_onednn_selfatt_qk_split");
  } else {
    node->attrs.op = Op::Get("_sg_onednn_selfatt_qk");
  }
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  node->attrs.subgraphs = attrs.subgraphs;
  node->op()->attr_parser(&(node->attrs));
  return node;
}

template <bool with_split>
static size_t SgDNNLSelfAttQKNumForwardInputs(const DNNLSelfAttParam& param) {
  if (param.quantized) {
    return with_split ? 3U : 6U;
  }
  return with_split ? 1U : 2U;
}

template <bool with_split>
static void SgDNNLSelfAttQKBackward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  CHECK(SgDNNLSelfAttSupportsBackward(param))
      << "_backward_sg_onednn_selfatt_qk currently supports float32 oneDNN self-attention "
         "backward and quantized QAT backward with enabled_float_output=float32";
  const size_t fwd_inputs = SgDNNLSelfAttQKNumForwardInputs<with_split>(param);
  CHECK_EQ(inputs.size(), fwd_inputs + 1);
  CHECK_EQ(outputs.size(), fwd_inputs);
  CHECK_EQ(req.size(), fwd_inputs);

  const NDArray output_grad =
      CastSelfAttGradToFloatCPU(inputs[0], "_backward_sg_onednn_selfatt_qk");
  const float* grad_ptr = output_grad.data().dptr<float>();
  const size_t base     = 1;

  if constexpr (with_split) {
    const NDArray qkv =
        FloatSelfAttTensorCPU(inputs[base],
                              param.quantized ? &inputs[base + 1] : nullptr,
                              param.quantized ? &inputs[base + 2] : nullptr,
                              "_backward_sg_onednn_selfatt_qk_split qkv");
    const auto qkv_shape = qkv.shape();
    CHECK_EQ(qkv_shape.ndim(), 3U);
    CHECK_EQ(qkv_shape[2] % QKV_NUM, 0U);
    const index_t batch_size = qkv_shape[0];
    const index_t seq_len    = qkv_shape[1];
    const index_t embed_dim  = qkv_shape[2] / QKV_NUM;
    CHECK_EQ(embed_dim % param.heads, 0);
    const index_t head_dim = embed_dim / param.heads;
    CHECK_EQ(output_grad.shape()[0], batch_size);
    CHECK_EQ(output_grad.shape()[1], param.heads);
    CHECK_EQ(output_grad.shape()[2], seq_len);
    CHECK_EQ(output_grad.shape()[3], seq_len);

    if (req[0] != kNullOp) {
      PrepareDefaultSelfAttOutputCPU(outputs[0], req[0]);
      if (req[0] == kWriteTo || req[0] == kWriteInplace) {
        MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
          ZeroSelfAttOutputCPUImpl<DType>(outputs[0]);
        });
      }
      CHECK_EQ(outputs[0].dtype(), mshadow::kFloat32);
      const float* qkv_ptr = qkv.data().dptr<float>();
      float* qkv_grad      = outputs[0].data().dptr<float>();
      const float beta     = req[0] == kAddTo ? 1.f : 0.f;
      const index_t qkv_row_stride = qkv_shape[2];
      for (index_t b = 0; b < batch_size; ++b) {
        for (int h = 0; h < param.heads; ++h) {
          const size_t q_offset = (b * seq_len * qkv_row_stride) + h * head_dim;
          const size_t k_offset = (b * seq_len * qkv_row_stride) + embed_dim + h * head_dim;
          const float* query    = qkv_ptr + q_offset;
          const float* key      = qkv_ptr + k_offset;
          const float* grad     = grad_ptr + ((b * param.heads + h) * seq_len * seq_len);
          float* query_grad     = qkv_grad + q_offset;
          float* key_grad       = qkv_grad + k_offset;
          SelfAttRowMajorSgemm(false,
                               false,
                               seq_len,
                               head_dim,
                               seq_len,
                               1.f,
                               grad,
                               seq_len,
                               key,
                               qkv_row_stride,
                               beta,
                               query_grad,
                               qkv_row_stride);
          SelfAttRowMajorSgemm(true,
                               false,
                               seq_len,
                               head_dim,
                               seq_len,
                               1.f,
                               grad,
                               seq_len,
                               query,
                               qkv_row_stride,
                               beta,
                               key_grad,
                               qkv_row_stride);
        }
      }
    }
    for (size_t i = 1; i < fwd_inputs; ++i) {
      ZeroLikeSelfAttOutputCPU(outputs[i], req[i]);
    }
  } else {
    const NDArray queries =
        FloatSelfAttTensorCPU(inputs[base],
                              param.quantized ? &inputs[base + 2] : nullptr,
                              param.quantized ? &inputs[base + 3] : nullptr,
                              "_backward_sg_onednn_selfatt_qk queries");
    const NDArray keys =
        FloatSelfAttTensorCPU(inputs[base + 1],
                              param.quantized ? &inputs[base + 4] : nullptr,
                              param.quantized ? &inputs[base + 5] : nullptr,
                              "_backward_sg_onednn_selfatt_qk keys");
    const auto query_shape = queries.shape();
    const auto key_shape   = keys.shape();
    CHECK_EQ(query_shape.ndim(), 3U);
    CHECK_EQ(key_shape.ndim(), 3U);
    CHECK_EQ(query_shape[0], key_shape[0]);
    CHECK_EQ(query_shape[2], key_shape[2]);
    CHECK_EQ(query_shape[2] % param.heads, 0);
    const index_t batch_size = query_shape[0];
    const index_t query_len  = query_shape[1];
    const index_t key_len    = key_shape[1];
    const index_t embed_dim  = query_shape[2];
    const index_t head_dim   = embed_dim / param.heads;
    CHECK_EQ(output_grad.shape()[0], batch_size);
    CHECK_EQ(output_grad.shape()[1], param.heads);
    CHECK_EQ(output_grad.shape()[2], query_len);
    CHECK_EQ(output_grad.shape()[3], key_len);

    const float* query_ptr = queries.data().dptr<float>();
    const float* key_ptr   = keys.data().dptr<float>();
    float* query_grad      = nullptr;
    float* key_grad        = nullptr;
    if (req[0] != kNullOp) {
      CHECK_EQ(outputs[0].dtype(), mshadow::kFloat32);
      PrepareDefaultSelfAttOutputCPU(outputs[0], req[0]);
      query_grad = outputs[0].data().dptr<float>();
    }
    if (req[1] != kNullOp) {
      CHECK_EQ(outputs[1].dtype(), mshadow::kFloat32);
      PrepareDefaultSelfAttOutputCPU(outputs[1], req[1]);
      key_grad = outputs[1].data().dptr<float>();
    }
    const float query_beta = req[0] == kAddTo ? 1.f : 0.f;
    const float key_beta   = req[1] == kAddTo ? 1.f : 0.f;
    for (index_t b = 0; b < batch_size; ++b) {
      for (int h = 0; h < param.heads; ++h) {
        const size_t q_offset = (b * query_len * embed_dim) + h * head_dim;
        const size_t k_offset = (b * key_len * embed_dim) + h * head_dim;
        const float* query    = query_ptr + q_offset;
        const float* key      = key_ptr + k_offset;
        const float* grad     = grad_ptr + ((b * param.heads + h) * query_len * key_len);
        if (query_grad != nullptr) {
          SelfAttRowMajorSgemm(false,
                               false,
                               query_len,
                               head_dim,
                               key_len,
                               1.f,
                               grad,
                               key_len,
                               key,
                               embed_dim,
                               query_beta,
                               query_grad + q_offset,
                               embed_dim);
        }
        if (key_grad != nullptr) {
          SelfAttRowMajorSgemm(true,
                               false,
                               key_len,
                               head_dim,
                               query_len,
                               1.f,
                               grad,
                               key_len,
                               query,
                               embed_dim,
                               key_beta,
                               key_grad + k_offset,
                               embed_dim);
        }
      }
    }
    for (size_t i = 2; i < fwd_inputs; ++i) {
      ZeroLikeSelfAttOutputCPU(outputs[i], req[i]);
    }
  }
}

template <bool with_split>
static bool SgDNNLSelfAttQKBackwardShape(const nnvm::NodeAttrs& attrs,
                                         mxnet::ShapeVector* in_shapes,
                                         mxnet::ShapeVector* out_shapes) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  const size_t fwd_inputs = SgDNNLSelfAttQKNumForwardInputs<with_split>(param);
  CHECK_EQ(in_shapes->size(), fwd_inputs + 1);
  CHECK_EQ(out_shapes->size(), fwd_inputs);
  for (size_t i = 0; i < fwd_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*out_shapes, i, (*in_shapes)[i + 1]);
  }
  return true;
}

static bool SgDNNLSelfAttValidBackwardGradType(int type_flag) {
  return type_flag == mshadow::kFloat32 || type_flag == mshadow::kInt8 ||
         type_flag == mshadow::kUint8 || type_flag == mshadow::kInt32;
}

template <bool with_split>
static bool SgDNNLSelfAttQKBackwardType(const nnvm::NodeAttrs& attrs,
                                        std::vector<int>* in_types,
                                        std::vector<int>* out_types) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  CHECK(SgDNNLSelfAttSupportsBackward(param));
  const size_t fwd_inputs = SgDNNLSelfAttQKNumForwardInputs<with_split>(param);
  CHECK_EQ(in_types->size(), fwd_inputs + 1);
  CHECK_EQ(out_types->size(), fwd_inputs);
  if (in_types->at(0) == -1) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
  } else {
    CHECK(SgDNNLSelfAttValidBackwardGradType(in_types->at(0)))
        << "_backward_sg_onednn_selfatt_qk expects float32/int8/uint8/int32 output gradients";
  }

  const size_t base = 1;
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, base, mshadow::kInt8);
    if constexpr (with_split) {
      TYPE_ASSIGN_CHECK(*in_types, base + 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, base + 2, mshadow::kFloat32);
    } else {
      TYPE_ASSIGN_CHECK(*in_types, base + 1, mshadow::kInt8);
      for (size_t i = base + 2; i < base + fwd_inputs; ++i) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      }
    }
  } else {
    TYPE_ASSIGN_CHECK(*in_types, base, mshadow::kFloat32);
    if constexpr (!with_split) {
      TYPE_ASSIGN_CHECK(*in_types, base + 1, mshadow::kFloat32);
    }
  }

  for (size_t i = 0; i < fwd_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
  }
  return true;
}

static bool SgDNNLSelfAttBackwardStorageType(const nnvm::NodeAttrs& attrs,
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

template <bool with_split>
struct SgDNNLSelfAttQKGrad {
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const auto& param = nnvm::get<DNNLSelfAttParam>(n->attrs.parsed);
    if (!SgDNNLSelfAttSupportsBackward(param)) {
      return MakeZeroGradNodes(n, ograds);
    }
    std::vector<nnvm::NodeEntry> heads;
    heads.reserve(n->inputs.size() + 1);
    heads.emplace_back(ograds[0]);
    heads.insert(heads.end(), n->inputs.begin(), n->inputs.end());
    auto p        = nnvm::Node::Create();
    p->attrs.op   = nnvm::Op::Get(with_split ? "_backward_sg_onednn_selfatt_qk_split" :
                                               "_backward_sg_onednn_selfatt_qk");
    p->attrs.name = n->attrs.name + "_backward";
    p->attrs.dict = n->attrs.dict;
    p->inputs     = std::move(heads);
    p->control_deps.emplace_back(n);
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    CHECK_EQ(p->num_inputs(), p->inputs.size())
        << "Number of inputs to operator " << p->op()->name << " (" << p->num_inputs()
        << ") does not match the actual number of inputs provided to operator "
        << p->attrs.name << " (" << p->inputs.size() << ").";
    return CreateNodeEntries(p);
  }
};

#define MXNET_OPERATOR_REGISTER_SELFATT_QK(name)                                                 \
  NNVM_REGISTER_OP(name)                                                                         \
      .set_num_outputs([](const NodeAttrs& attrs) {                                              \
        auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);                           \
        if (param.quantized && !param.enabled_float_output.has_value()) {                        \
          return 3;                                                                              \
        } else {                                                                                 \
          return 1;                                                                              \
        }                                                                                        \
      })                                                                                         \
      .set_attr<nnvm::FListOutputNames>(                                                         \
          "FListOutputNames",                                                                    \
          [](const NodeAttrs& attrs) {                                                           \
            auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);                       \
            std::vector<std::string> output_names{"output"};                                     \
            if (param.quantized && !param.enabled_float_output.has_value()) {                    \
              output_names.emplace_back("min_output");                                           \
              output_names.emplace_back("max_output");                                           \
            }                                                                                    \
            return output_names;                                                                 \
          })                                                                                     \
      .set_attr_parser(ParamParser<DNNLSelfAttParam>)                                            \
      .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttStorageType)                \
      .set_attr<FCreateOpState>("FCreateOpState", CreateSgDNNLSelfAttQKState)                    \
      .set_attr<bool>("TIsDNNL", true)                                                           \
      .set_attr<FQuantizable>("FQuantizable",                                                    \
                              [](const NodeAttrs& attrs) { return QuantizeType::kMust; })        \
      .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; }) \
      .add_arguments(DNNLSelfAttParam::__FIELDS__())

MXNET_OPERATOR_REGISTER_SELFATT_QK(_sg_onednn_selfatt_qk)
    .describe(R"code(_sg_onednn_selfatt_qk)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 6;
      } else {
        return 2;
      }
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       auto const& param =
                                           nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                       std::vector<std::string> input_names{"queries"};
                                       input_names.emplace_back("keys");
                                       if (param.quantized) {
                                         input_names.emplace_back("min_q");
                                         input_names.emplace_back("max_q");
                                         input_names.emplace_back("min_k");
                                         input_names.emplace_back("max_k");
                                       }
                                       return input_names;
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttShape<false>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKInferType<false>)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLSelfAttQKForward<false>)
    .set_attr<nnvm::FGradient>("FGradient", SgDNNLSelfAttQKGrad<false>{})
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttQKQuantizedOp<false>)
    .add_argument("queries", "NDArray-or-Symbol", "Interleaved queries, keys and values")
    .add_argument("keys", "NDArray-or-Symbol", "Interleaved queries, keys and values")
    .add_argument("min_q", "NDArray-or-Symbol", "Minimum value of queries.")
    .add_argument("max_q", "NDArray-or-Symbol", "Maximum value of queries.")
    .add_argument("min_k", "NDArray-or-Symbol", "Minimum value of keys.")
    .add_argument("max_k", "NDArray-or-Symbol", "Maximum value of keys.");

MXNET_OPERATOR_REGISTER_SELFATT_QK(_sg_onednn_selfatt_qk_split)
    .add_alias("_sg_mkldnn_selfatt_qk")
    .describe(R"code(_sg_onednn_selfatt_qk_split)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       auto const& param =
                                           nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                       std::vector<std::string> input_names{"queries_keys_values"};
                                       if (param.quantized) {
                                         input_names.emplace_back("min_qkv");
                                         input_names.emplace_back("max_qkv");
                                       }
                                       return input_names;
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttShape<true>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKInferType<true>)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLSelfAttQKForward<true>)
    .set_attr<nnvm::FGradient>("FGradient", SgDNNLSelfAttQKGrad<true>{})
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttQKQuantizedOp<true>)
    .add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
    .add_argument("min_qkv", "NDArray-or-Symbol", "Minimum value of queries, keys and values.")
    .add_argument("max_qkv", "NDArray-or-Symbol", "Maximum value of queries, keys and values.");

NNVM_REGISTER_OP(_backward_sg_onednn_selfatt_qk)
    .set_num_inputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttQKNumForwardInputs<false>(
                 nnvm::get<DNNLSelfAttParam>(attrs.parsed)) +
             1;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttQKNumForwardInputs<false>(
          nnvm::get<DNNLSelfAttParam>(attrs.parsed));
    })
    .set_attr_parser(ParamParser<DNNLSelfAttParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttQKBackwardShape<false>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKBackwardType<false>)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttBackwardStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", SgDNNLSelfAttQKBackward<false>);

NNVM_REGISTER_OP(_backward_sg_onednn_selfatt_qk_split)
    .set_num_inputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttQKNumForwardInputs<true>(
                 nnvm::get<DNNLSelfAttParam>(attrs.parsed)) +
             1;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttQKNumForwardInputs<true>(
          nnvm::get<DNNLSelfAttParam>(attrs.parsed));
    })
    .set_attr_parser(ParamParser<DNNLSelfAttParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttQKBackwardShape<true>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKBackwardType<true>)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttBackwardStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", SgDNNLSelfAttQKBackward<true>);

/**********************************_sg_onednn_selfatt_valatt**********************************/

static bool SgDNNLSelfAttValShape(const NodeAttrs& attrs,
                                  mxnet::ShapeVector* in_shape,
                                  mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  auto att_shape     = in_shape->at(0);
  auto qkv_shape     = in_shape->at(1);

  CHECK_EQ(att_shape.ndim(), 4U)
      << "Attention maps should be 4D in batch-heads-seq_length-seq_length, "
      << "but the given tensor is " << att_shape.ndim() << "D";

  CHECK_EQ(qkv_shape.ndim(), 3U)
      << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
      << "but the given tensor is " << qkv_shape.ndim() << "D";

  if (params.quantized) {
    CHECK_EQ(in_shape->size(), 6U) << "Input:[attention, queries_keys_values, "
                                   << "attn_min, attn_max, qkv_min, qkv_max] - currently have "
                                   << in_shape->size() << " inputs";
    for (int i = 2; i < 6; i++) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape({1}));
    }

    out_shape->resize(params.enabled_float_output.has_value() ? 1 : 3);
    SHAPE_ASSIGN_CHECK(
        *out_shape,
        0,
        mxnet::TShape(
            {att_shape[0], att_shape[2], att_shape[1] * qkv_shape[2] / params.heads / QKV_NUM}));
    if (!params.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));  // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));  // max output
    }
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Inputs: [queries_keys_values, attention] - currently have "
                                   << in_shape->size() << " inputs";
    auto qkv_shape = in_shape->at(1);
    auto att_shape = in_shape->at(0);
    CHECK_EQ(qkv_shape.ndim(), 3U)
        << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
        << "but the given tensor is " << qkv_shape.ndim() << "D";
    out_shape->resize(1);
    SHAPE_ASSIGN_CHECK(
        *out_shape,
        0,
        mxnet::TShape(
            {att_shape[0], att_shape[2], att_shape[1] * qkv_shape[2] / params.heads / QKV_NUM}));
    return true;
  }

  return true;
}

static bool SgDNNLSelfAttValInferType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int>* in_types,
                                      std::vector<int>* out_types) {
  const auto& params = nnvm::get<DNNLSelfAttParam>(attrs.parsed);

  if (params.quantized) {
    if (in_types->at(0) == mshadow::kBfloat16 || in_types->at(1) == mshadow::kBfloat16) {
      return false;
    }

    CHECK_EQ(in_types->size(), 6U) << "Input:[attention, queries_keys_values, min_att, max_att, "
                                      "min_qkv, max_qkv] - currently have "
                                   << in_types->size() << " inputs";

    CHECK(in_types->at(0) == mshadow::kUint8)
        << "QuantizedSelfAttentionQK only supports int8/uint8 input, while " << in_types->at(0)
        << " is given.";
    CHECK(in_types->at(1) == mshadow::kInt8 || in_types->at(1) == mshadow::kUint8)
        << "QuantizedSelfAttentionQK only supports int8/uint8 input, while " << in_types->at(1)
        << " is given.";
    for (int i = 2; i < 6; i++) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }

    if (params.enabled_float_output.has_value()) {
      CHECK_EQ(out_types->size(), 1U);
      TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
    } else {
      CHECK_EQ(out_types->size(), 3U);
      if (params.min_calib_range.has_value() && params.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
  } else {
    CHECK_EQ(in_types->size(), 2U);
    CHECK_EQ(out_types->size(), 1U);
    if (in_types->at(0) == mshadow::kFloat32 || in_types->at(1) == mshadow::kFloat32) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else if (in_types->at(0) == mshadow::kBfloat16 || in_types->at(1) == mshadow::kBfloat16) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kBfloat16);
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kBfloat16);
      if (params.enabled_float_output.has_value()) {
        CHECK_EQ(params.enabled_float_output.value(), mshadow::kFloat32);
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kBfloat16);
      }
    } else {
      return false;
    }
  }

  return true;
}

nnvm::ObjectPtr SgDNNLSelfAttValAttQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node          = nnvm::Node::Create();
  auto const& param             = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  node->attrs.op                = Op::Get("_sg_onednn_selfatt_valatt");
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  node->attrs.subgraphs = attrs.subgraphs;
  node->op()->attr_parser(&(node->attrs));
  return node;
}

class DNNLSelfAttValAttOp {
 public:
  explicit DNNLSelfAttValAttOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<DNNLSelfAttParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               bool already_prepared);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph oneDNN self attention val only supports "
                  "inference computation.";
  }

  void Initialize(const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  DNNLSelfAttParam param_;
  dnnl_args_map_t args_;
  dnnl_args_map_t reorder_args;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::reorder> reorder_;
  std::shared_ptr<dnnl::memory> cached_att_mem_;
  std::shared_ptr<dnnl::memory> cached_value_mem_;
  std::shared_ptr<dnnl::memory> cached_result_mem_;
  std::shared_ptr<dnnl::memory> cached_tmp_mem_;
  std::shared_ptr<dnnl::memory> cached_transposed_mem_;  // op output
  // v3: runtime scale tensor for set_scales_mask matmul attr.
  std::shared_ptr<dnnl::memory> cached_scale_mem_;
  // F4/F11: default-init for same reason as the QK op above.
  float min_qkv_{0.0f};
  float max_qkv_{0.0f};
  float min_att_{0.0f};
  float max_att_{0.0f};
  float min_output_{0.0f};
  float max_output_{0.0f};
  float qkv_scale_{0.0f};
  float att_scale_{0.0f};
};

static OpStatePtr CreateDNNLSelfAttValAttState(const nnvm::NodeAttrs& attrs,
                                               Context ctx,
                                               const mxnet::ShapeVector& in_shapes,
                                               const std::vector<int>& in_types) {
  return OpStatePtr::Create<DNNLSelfAttValAttOp>(attrs);
}

static void DNNLSelfAttValAttForward(const OpStatePtr& state_pointer,
                                     const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  // XOP19: same primary-output req gate as SgDNNLSelfAttQKForward above.
  // The ValAtt matmul also writes in place into outputs[0] without a tmp
  // accumulation buffer, so kAddTo cannot be honored.
  if (req[0] == kNullOp)
    return;
  CHECK_NE(req[0], kAddTo)
      << "kAddTo is not supported for the primary output of _sg_onednn_selfatt_valatt";
  DNNLSelfAttValAttOp& op = state_pointer.get_state<DNNLSelfAttValAttOp>();
  // AMP / oneDNN v3 fallback: matmul has no bf16 kernel on AVX2 either, so
  // mirror the upcast that SgDNNLSelfAttQKForward applies for the QK matmul.
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
          f32_out.emplace_back(nd.shape(), nd.ctx(), /*delay_alloc=*/false,
                               static_cast<int>(mshadow::kFloat32));
          out_was_bf16.push_back(true);
        } else {
          f32_out.emplace_back(nd);
          out_was_bf16.push_back(false);
        }
      }
      std::vector<OpReqType> f32_req;
      f32_req.reserve(req.size());
      for (size_t i = 0; i < req.size(); ++i) {
        f32_req.push_back((i < out_was_bf16.size() && out_was_bf16[i]) ? kWriteTo : req[i]);
      }
      DNNLSelfAttValAttForward(state_pointer, ctx, f32_in, f32_req, f32_out);
      DNNLStream::Get()->Submit();
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!out_was_bf16[i]) continue;
        if (req[i] == kNullOp) continue;
        CHECK_NE(req[i], kAddTo)
            << "kAddTo not supported for BF16 fallback path on output " << i
            << " of _sg_onednn_selfatt_valatt";
        auto src_mem = f32_out[i].GetDNNLData();
        auto dst_mem = outputs[i].GetDNNLData();
        ReorderTo(src_mem, dst_mem);
      }
      return;
    }
  }
  bool already_prepared   = false;
  if (!op.IsInitialized()) {
    op.Initialize(ctx, inputs, req, outputs);
    already_prepared = true;
  }
  op.Forward(ctx, inputs, req, outputs, already_prepared);
}

void DNNLSelfAttValAttOp::Initialize(const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  using namespace dnnl;

  const auto attn_tensor = inputs[0].Reorder2Default();
  const auto qkv_tensor  = inputs[1].Reorder2Default();
  const auto out_tensor  = outputs[0];

  const auto qkv_dtype  = get_dnnl_type(qkv_tensor.dtype());
  const auto attn_dtype = get_dnnl_type(attn_tensor.dtype());

  const memory::dim heads          = param_.heads;
  const memory::dim sequences      = qkv_tensor.shape()[0];
  const memory::dim qkv_seq_len    = qkv_tensor.shape()[1];
  const memory::dim output_lin_dim = qkv_tensor.shape()[2];
  const memory::dim embed_dim      = output_lin_dim / QKV_NUM;
  const memory::dim head_dim       = embed_dim / heads;
  const memory::dim batch_stride   = output_lin_dim * qkv_seq_len;

  const auto engine = CpuEngine::Get()->get_engine();

  memory::dims attn_dims  = {sequences, heads, qkv_seq_len, qkv_seq_len};
  memory::dims value_dims = {sequences, heads, qkv_seq_len, head_dim};
  memory::dims out_dims   = {sequences, heads, qkv_seq_len, head_dim};

  // needed to make transpose on 2nd and 3rd axis with oneDNN
  memory::dims transpose_dims = {sequences, heads, qkv_seq_len, head_dim, 1};

  memory::dims value_strides = {batch_stride, head_dim, output_lin_dim, 1};

  // for attention tensor just use normal data layout,
  // for value tensor we need to use strides as input tensor consists of queries, keys and values
  const auto attn_md  = memory::desc(attn_dims, attn_dtype, memory::format_tag::abcd);
  const auto value_md = memory::desc(value_dims, qkv_dtype, value_strides);

  // result = attn * value
  // tmp = result + artificial dimension (1) - same memory ptr as result
  // transpose = transposed tmp - output
  memory::desc result_md, tmp_md, transpose_md;

  float oscale = 1.0f;
  if (param_.quantized) {
    min_att_ = inputs[2].data().dptr<float>()[0];
    max_att_ = inputs[3].data().dptr<float>()[0];
    min_qkv_ = inputs[4].data().dptr<float>()[0];
    max_qkv_ = inputs[5].data().dptr<float>()[0];

    att_scale_ = GetQuantizeScale(mshadow::kUint8, min_att_, max_att_);
    qkv_scale_ = GetQuantizeScale(mshadow::kInt8, min_qkv_, max_qkv_);

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();
      oscale      = (att_scale_ * qkv_scale_) /
               GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_);
    } else if (param_.enabled_float_output.has_value()) {
      oscale = att_scale_ * qkv_scale_;
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_att_, &max_att_, &min_qkv_, &max_qkv_);
    }
  }
  memory::data_type result_dnnl_dtype = get_dnnl_type(out_tensor.dtype());

  result_md    = memory::desc(out_dims, result_dnnl_dtype, memory::format_tag::abcd);
  tmp_md       = memory::desc(transpose_dims, result_dnnl_dtype, memory::format_tag::abcde);
  transpose_md = memory::desc(transpose_dims, result_dnnl_dtype, memory::format_tag::acbde);

  // multiply by 2 as we need to skip query and key
  const size_t value_offset = inputs[1].shape()[2] / QKV_NUM * 2;
  auto att_buffer           = inputs[0];
  if (att_buffer.IsDNNLData())
    att_buffer = att_buffer.Reorder2Default();

  MSHADOW_TYPE_SWITCH(att_buffer.dtype(), DType, {
    DType* attention_ptr = att_buffer.data().dptr<DType>();
    cached_att_mem_      = std::make_shared<memory>(attn_md, engine, attention_ptr);
  });

  MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
    DType* value_mem_ptr = inputs[1].data().dptr<DType>() + value_offset;
    cached_value_mem_    = std::make_shared<memory>(value_md, engine, value_mem_ptr);
  });

  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    cached_result_mem_ = std::make_shared<memory>(result_md, engine);
    DType* orig_buf    = reinterpret_cast<DType*>(cached_result_mem_->get_data_handle());
    cached_tmp_mem_    = std::make_shared<dnnl::memory>(tmp_md, engine, orig_buf);
    cached_transposed_mem_ =
        std::make_shared<dnnl::memory>(transpose_md, engine, outputs[0].data().dptr<DType>());
  });

  // v3: set_output_scales / matmul::desc removed.
  dnnl::primitive_attr attr;
  attr.set_scales_mask(DNNL_ARG_DST, 0);
  auto matmul_pd = matmul::primitive_desc(engine, attn_md, value_md, result_md, attr);
  fwd_           = std::make_shared<matmul>(matmul_pd);
  dnnl::memory::desc scale_md({1}, dnnl::memory::data_type::f32,
                              dnnl::memory::format_tag::x);
  cached_scale_mem_ = std::make_shared<memory>(scale_md, engine);
  *reinterpret_cast<float*>(cached_scale_mem_->get_data_handle()) = oscale;
  args_[DNNL_ARG_SRC]                        = *cached_att_mem_;
  args_[DNNL_ARG_WEIGHTS]                    = *cached_value_mem_;
  args_[DNNL_ARG_DST]                        = *cached_result_mem_;
  args_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = *cached_scale_mem_;

  auto reorder_pd            = dnnl::reorder::primitive_desc(engine, tmp_md, engine, transpose_md);
  reorder_                   = std::make_shared<dnnl::reorder>(reorder_pd);
  reorder_args[DNNL_ARG_SRC] = *cached_tmp_mem_;
  reorder_args[DNNL_ARG_DST] = *cached_transposed_mem_;

  initialized_ = true;
}

void DNNLSelfAttValAttOp::Forward(const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs,
                                  bool already_prepared) {
  if (!already_prepared) {
    // multiply by 2 as we need to skip queries and keys
    const size_t value_offset = inputs[1].shape()[2] / QKV_NUM * 2;

    auto att_buffer = inputs[0];
    if (att_buffer.IsDNNLData())
      att_buffer = att_buffer.Reorder2Default();

    MSHADOW_TYPE_SWITCH(att_buffer.dtype(), DType, {
      DType* attention_ptr = att_buffer.data().dptr<DType>();
      cached_att_mem_->set_data_handle(attention_ptr);
    });

    MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
      DType* qkv_ptr       = inputs[1].data().dptr<DType>();
      DType* value_mem_ptr = qkv_ptr + value_offset;
      cached_value_mem_->set_data_handle(value_mem_ptr);
    });

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_transposed_mem_->set_data_handle(outputs[0].data().dptr<DType>());
    });
  }
  DNNLStream::Get()->RegisterPrimArgs(*fwd_, args_);
  DNNLStream::Get()->RegisterPrimArgs(*reorder_, reorder_args);
  DNNLStream::Get()->Submit();

  if (param_.quantized && !param_.enabled_float_output.has_value()) {
    AssignQuantizedRangeOutput(outputs[1].data().dptr<float>(), &min_output_,
                               req[1], "_sg_onednn_selfatt_valatt");
    AssignQuantizedRangeOutput(outputs[2].data().dptr<float>(), &max_output_,
                               req[2], "_sg_onednn_selfatt_valatt");
  }
}

static size_t SgDNNLSelfAttValAttNumForwardInputs(const DNNLSelfAttParam& param) {
  return param.quantized ? 6U : 2U;
}

static void SgDNNLSelfAttValAttBackward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  CHECK(SgDNNLSelfAttSupportsBackward(param))
      << "_backward_sg_onednn_selfatt_valatt currently supports float32 oneDNN "
         "self-attention backward and quantized QAT backward with enabled_float_output=float32";
  const size_t fwd_inputs = SgDNNLSelfAttValAttNumForwardInputs(param);
  CHECK_EQ(inputs.size(), fwd_inputs + 1);
  CHECK_EQ(outputs.size(), fwd_inputs);
  CHECK_EQ(req.size(), fwd_inputs);

  const NDArray output_grad =
      CastSelfAttGradToFloatCPU(inputs[0], "_backward_sg_onednn_selfatt_valatt");
  const size_t base = 1;
  const NDArray attention =
      FloatSelfAttTensorCPU(inputs[base],
                            param.quantized ? &inputs[base + 2] : nullptr,
                            param.quantized ? &inputs[base + 3] : nullptr,
                            "_backward_sg_onednn_selfatt_valatt attention");
  const NDArray qkv =
      FloatSelfAttTensorCPU(inputs[base + 1],
                            param.quantized ? &inputs[base + 4] : nullptr,
                            param.quantized ? &inputs[base + 5] : nullptr,
                            "_backward_sg_onednn_selfatt_valatt qkv");
  const auto att_shape = attention.shape();
  const auto qkv_shape = qkv.shape();
  CHECK_EQ(att_shape.ndim(), 4U);
  CHECK_EQ(qkv_shape.ndim(), 3U);
  CHECK_EQ(qkv_shape[2] % QKV_NUM, 0U);
  const index_t batch_size = att_shape[0];
  const index_t heads      = att_shape[1];
  const index_t query_len  = att_shape[2];
  const index_t key_len    = att_shape[3];
  const index_t embed_dim  = qkv_shape[2] / QKV_NUM;
  CHECK_EQ(heads, param.heads);
  CHECK_EQ(qkv_shape[0], batch_size);
  CHECK_EQ(qkv_shape[1], key_len);
  CHECK_EQ(query_len, key_len)
      << "_sg_onednn_selfatt_valatt backward currently supports self-attention square maps";
  CHECK_EQ(embed_dim % heads, 0);
  const index_t head_dim = embed_dim / heads;
  CHECK_EQ(output_grad.shape()[0], batch_size);
  CHECK_EQ(output_grad.shape()[1], query_len);
  CHECK_EQ(output_grad.shape()[2], embed_dim);

  float* attention_grad = nullptr;
  float* qkv_grad       = nullptr;
  if (req[0] != kNullOp) {
    CHECK_EQ(outputs[0].dtype(), mshadow::kFloat32);
    PrepareDefaultSelfAttOutputCPU(outputs[0], req[0]);
    attention_grad = outputs[0].data().dptr<float>();
  }
  if (req[1] != kNullOp) {
    CHECK_EQ(outputs[1].dtype(), mshadow::kFloat32);
    PrepareDefaultSelfAttOutputCPU(outputs[1], req[1]);
    if (req[1] == kWriteTo || req[1] == kWriteInplace) {
      MSHADOW_TYPE_SWITCH(outputs[1].dtype(), DType, {
        ZeroSelfAttOutputCPUImpl<DType>(outputs[1]);
      });
    }
    qkv_grad = outputs[1].data().dptr<float>();
  }

  const float* grad_ptr = output_grad.data().dptr<float>();
  const float* att_ptr  = attention.data().dptr<float>();
  const float* qkv_ptr  = qkv.data().dptr<float>();
  const float att_beta  = req[0] == kAddTo ? 1.f : 0.f;
  const float qkv_beta  = req[1] == kAddTo ? 1.f : 0.f;
  const index_t qkv_row_stride = qkv_shape[2];
  for (index_t b = 0; b < batch_size; ++b) {
    for (index_t h = 0; h < heads; ++h) {
      const float* att = att_ptr + ((b * heads + h) * query_len * key_len);
      const float* out_grad =
          grad_ptr + (b * query_len * embed_dim) + h * head_dim;
      const float* value =
          qkv_ptr + (b * key_len * qkv_row_stride) + 2 * embed_dim + h * head_dim;
      if (qkv_grad != nullptr) {
        float* value_grad =
            qkv_grad + (b * key_len * qkv_row_stride) + 2 * embed_dim + h * head_dim;
        SelfAttRowMajorSgemm(true,
                             false,
                             key_len,
                             head_dim,
                             query_len,
                             1.f,
                             att,
                             key_len,
                             out_grad,
                             embed_dim,
                             qkv_beta,
                             value_grad,
                             qkv_row_stride);
      }
      if (attention_grad != nullptr) {
        float* att_grad = attention_grad + ((b * heads + h) * query_len * key_len);
        SelfAttRowMajorSgemm(false,
                             true,
                             query_len,
                             key_len,
                             head_dim,
                             1.f,
                             out_grad,
                             embed_dim,
                             value,
                             qkv_row_stride,
                             att_beta,
                             att_grad,
                             key_len);
      }
    }
  }
  for (size_t i = 2; i < fwd_inputs; ++i) {
    ZeroLikeSelfAttOutputCPU(outputs[i], req[i]);
  }
}

static bool SgDNNLSelfAttValAttBackwardShape(const nnvm::NodeAttrs& attrs,
                                             mxnet::ShapeVector* in_shapes,
                                             mxnet::ShapeVector* out_shapes) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  const size_t fwd_inputs = SgDNNLSelfAttValAttNumForwardInputs(param);
  CHECK_EQ(in_shapes->size(), fwd_inputs + 1);
  CHECK_EQ(out_shapes->size(), fwd_inputs);
  for (size_t i = 0; i < fwd_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*out_shapes, i, (*in_shapes)[i + 1]);
  }
  return true;
}

static bool SgDNNLSelfAttValAttBackwardType(const nnvm::NodeAttrs& attrs,
                                            std::vector<int>* in_types,
                                            std::vector<int>* out_types) {
  const auto& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  CHECK(SgDNNLSelfAttSupportsBackward(param));
  const size_t fwd_inputs = SgDNNLSelfAttValAttNumForwardInputs(param);
  CHECK_EQ(in_types->size(), fwd_inputs + 1);
  CHECK_EQ(out_types->size(), fwd_inputs);
  if (in_types->at(0) == -1) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
  } else {
    CHECK(SgDNNLSelfAttValidBackwardGradType(in_types->at(0)))
        << "_backward_sg_onednn_selfatt_valatt expects float32/int8/uint8/int32 "
           "output gradients";
  }

  const size_t base = 1;
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, base, mshadow::kUint8);
    if (in_types->at(base + 1) == -1) {
      TYPE_ASSIGN_CHECK(*in_types, base + 1, mshadow::kInt8);
    } else {
      CHECK(in_types->at(base + 1) == mshadow::kInt8 ||
            in_types->at(base + 1) == mshadow::kUint8)
          << "_backward_sg_onednn_selfatt_valatt expects int8/uint8 quantized qkv input";
    }
    for (size_t i = base + 2; i < base + fwd_inputs; ++i) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }
  } else {
    TYPE_ASSIGN_CHECK(*in_types, base, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_types, base + 1, mshadow::kFloat32);
  }

  for (size_t i = 0; i < fwd_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*out_types, i, mshadow::kFloat32);
  }
  return true;
}

struct SgDNNLSelfAttValAttGrad {
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const auto& param = nnvm::get<DNNLSelfAttParam>(n->attrs.parsed);
    if (!SgDNNLSelfAttSupportsBackward(param)) {
      return MakeZeroGradNodes(n, ograds);
    }
    std::vector<nnvm::NodeEntry> heads;
    heads.reserve(n->inputs.size() + 1);
    heads.emplace_back(ograds[0]);
    heads.insert(heads.end(), n->inputs.begin(), n->inputs.end());
    auto p        = nnvm::Node::Create();
    p->attrs.op   = nnvm::Op::Get("_backward_sg_onednn_selfatt_valatt");
    p->attrs.name = n->attrs.name + "_backward";
    p->attrs.dict = n->attrs.dict;
    p->inputs     = std::move(heads);
    p->control_deps.emplace_back(n);
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    CHECK_EQ(p->num_inputs(), p->inputs.size())
        << "Number of inputs to operator " << p->op()->name << " (" << p->num_inputs()
        << ") does not match the actual number of inputs provided to operator "
        << p->attrs.name << " (" << p->inputs.size() << ").";
    return CreateNodeEntries(p);
  }
};

NNVM_REGISTER_OP(_sg_onednn_selfatt_valatt)
    .add_alias("_sg_mkldnn_selfatt_valatt")
    .describe(R"code(_sg_onednn_selfatt_valatt)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 6;
      } else {
        return 2;
      }
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized && !param.enabled_float_output.has_value()) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr_parser(ParamParser<DNNLSelfAttParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
          std::vector<std::string> input_names{"attention", "queries_keys_values"};
          if (param.quantized) {
            input_names.emplace_back("min_attention");
            input_names.emplace_back("max_attention");

            input_names.emplace_back("min_qkv");
            input_names.emplace_back("max_qkv");
          }
          return input_names;
        })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        auto const& param =
                                            nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                        std::vector<std::string> output_names{"output"};
                                        if (param.quantized &&
                                            !param.enabled_float_output.has_value()) {
                                          output_names.emplace_back("min_output");
                                          output_names.emplace_back("max_output");
                                        }
                                        return output_names;
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttValShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttValInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttStorageType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateDNNLSelfAttValAttState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", DNNLSelfAttValAttForward)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<nnvm::FGradient>("FGradient", SgDNNLSelfAttValAttGrad{})
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttValAttQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("attention", "NDArray-or-Symbol", "Attention maps")
    .add_argument("queries_keys_values",
                  "NDArray-or-Symbol",
                  "Queries, keys and values interleaved")
    .add_argument("min_attention", "NDArray-or-Symbol", "Minimum value of attention maps.")
    .add_argument("max_attention", "NDArray-or-Symbol", "Maximum value of attention maps.")
    .add_argument("min_qkv", "NDArray-or-Symbol", "Minimum value of queries, keys and values.")
    .add_argument("max_qkv", "NDArray-or-Symbol", "Maximum value of queries, keys and values.")
    .add_arguments(DNNLSelfAttParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_sg_onednn_selfatt_valatt)
    .set_num_inputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttValAttNumForwardInputs(
                 nnvm::get<DNNLSelfAttParam>(attrs.parsed)) +
             1;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      return SgDNNLSelfAttValAttNumForwardInputs(
          nnvm::get<DNNLSelfAttParam>(attrs.parsed));
    })
    .set_attr_parser(ParamParser<DNNLSelfAttParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttValAttBackwardShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttValAttBackwardType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttBackwardStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", SgDNNLSelfAttValAttBackward);

}  // namespace op
}  // namespace mxnet

#endif
