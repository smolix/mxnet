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
 * \file dnnl_convolution.cc
 * \brief
 * \author Da Zheng
 */

#if MXNET_USE_ONEDNN == 1

#include <cstring>
#include <string>

#include "operator/nn/convolution-inl.h"
#include "dnnl_base-inl.h"
#include "dnnl_convolution-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLConvParam);

// Support for https://oneapi-src.github.io/oneDNN/v3/dev_guide_convolution.html
bool SupportDNNLConv(const ConvolutionParam& params, const NDArray& input) {
  if (params.kernel.ndim() > 3 || params.kernel.ndim() == 0)
    return false;
  return SupportDNNL<3, 5, DNNLTypeMode::AllTypes>(input);
}

std::shared_ptr<dnnl::convolution_forward::primitive_desc> GetConvFwdImpl(
    const DNNLConvFullParam& param,
    const bool is_train,
    const NDArray& data,
    const NDArray& weights,
    const NDArray* bias,
    const NDArray& output) {
  auto prop      = is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_inference;
  auto data_md   = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.conv_param.num_group, param.dnnl_param.quantized);
  auto out_md    = GetMemDesc(output);
  // v3 dequant: when dst is f32 use f32 bias_md (matches the dequant path used
  // by FC). The s32-bias + f32-dst + DST-scale combo is rejected by v3 conv.
  const bool float_output =
      param.dnnl_param.quantized && param.dnnl_param.enabled_float_output.has_value();
  auto bias_md =
      bias ? (param.dnnl_param.quantized
                  ? GetMemDesc(*bias, float_output ? mshadow::kFloat32 : mshadow::kInt32)
                  : GetMemDesc(*bias)) :
             dnnl::memory::desc{{}, dnnl::memory::data_type::undef, dnnl::memory::format_tag::any};
  auto bias_md_ptr = bias ? &bias_md : nullptr;

  dnnl::memory::dims strides(param.conv_param.kernel.ndim());
  dnnl::memory::dims padding(param.conv_param.kernel.ndim());
  if (param.conv_param.kernel.ndim() == 1) {
    CHECK_GE(param.conv_param.stride.ndim(), 1);
    CHECK_GE(param.conv_param.pad.ndim(), 1);
    CHECK_GE(param.conv_param.dilate.ndim(), 1);
    strides[0] = param.conv_param.stride[0];
    padding[0] = param.conv_param.pad[0];
  } else if (param.conv_param.kernel.ndim() == 2) {
    CHECK_GE(param.conv_param.stride.ndim(), 2);
    CHECK_GE(param.conv_param.pad.ndim(), 2);
    CHECK_GE(param.conv_param.dilate.ndim(), 2);
    strides[0] = param.conv_param.stride[0];
    strides[1] = param.conv_param.stride[1];
    padding[0] = param.conv_param.pad[0];
    padding[1] = param.conv_param.pad[1];
  } else if (param.conv_param.kernel.ndim() == 3) {
    CHECK_GE(param.conv_param.stride.ndim(), 3);
    CHECK_GE(param.conv_param.pad.ndim(), 3);
    CHECK_GE(param.conv_param.dilate.ndim(), 3);
    strides[0] = param.conv_param.stride[0];
    strides[1] = param.conv_param.stride[1];
    strides[2] = param.conv_param.stride[2];
    padding[0] = param.conv_param.pad[0];
    padding[1] = param.conv_param.pad[1];
    padding[2] = param.conv_param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected oneDNN Conv kernel size " << param.conv_param.kernel.ndim()
               << ", supporting only 1 or 2 or 3.";
  }
  dnnl::primitive_attr attr;
  dnnl::post_ops ops;
  if (param.dnnl_param.with_act) {
    // v3: post_ops::append_eltwise(algo, alpha, beta); scale parameter
    //     dropped. The per-post-op scale is folded into the global output
    //     scales when needed.
    const auto& act_param = param.act_param;
    ops.append_eltwise(act_param.alg, act_param.alpha, act_param.beta);
  }
  if (param.dnnl_param.with_sum) {
    ops.append_sum(param.sum_scale);
  }
  if (param.dnnl_param.with_postsum_act) {
    const auto& act_param = param.postsum_act_param;
    ops.append_eltwise(act_param.alg, act_param.alpha, act_param.beta);
  }
  attr.set_post_ops(ops);

  if (param.dnnl_param.quantized && param.requantize_scales.size()) {
    if (float_output) {
      // v3 dequant: bind dequant scales to SRC + WEIGHTS (no DST scale). v3
      // conv rejects s32 bias + f32 dst + DST scale; splitting the scale and
      // using f32 bias avoids the unsupported combo. Mirrors dnnl_fc.cc.
      attr.set_scales_mask(DNNL_ARG_SRC, 0);
      attr.set_scales_mask(DNNL_ARG_WEIGHTS, param.requantize_scales.size() > 1 ? 1 : 0);
    } else if (param.requantize_scales.size() > 1) {
      // v3: per-OC output scales must go on DNNL_ARG_WEIGHTS (mask=1<<0=1 along
      // the OC axis of weights) because v3 conv rejects per-channel DNNL_ARG_DST
      // scales (only per-tensor mask=0 supported on DST). Mathematically
      // equivalent to v2 set_output_scales(mask=2): each OC's output is scaled
      // by scales[oc]. Per-tensor case (size==1) keeps DNNL_ARG_DST mask=0.
      attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1);
    } else {
      attr.set_scales_mask(DNNL_ARG_DST, 0);
    }
  }

  // v3 dilation is always required; build it once (zeros if no dilation).
  const int kndim = param.conv_param.kernel.ndim();
  dnnl::memory::dims dilates(kndim, 0);
  if (param.conv_param.dilate.ndim() != 0) {
    CHECK_EQ(param.conv_param.dilate.ndim(), kndim);
    for (int i = 0; i < kndim; ++i) {
      dilates[i] = param.conv_param.dilate[i] - 1;
    }
  }

  auto engine = CpuEngine::Get()->get_engine();

  // FU-3 / apache#19218: oneDNN v3's brg_conv:avx2 path pads IC to a
  // multiple of simd_w (=8 on AVX2). For low-IC inputs (e.g. RGB IC=3) this
  // wastes ~63-81% of every AVX2 vector op on padding zeros, and the
  // thread-scaling on bs=1 collapses on many-NUMA hosts (8x EPYC Zen 2: 64
  // threads is ~10x slower than 1 thread). On AVX-512 hosts the heuristic
  // is fine -- brg_conv is the intended fast path. So we only skip brg when
  // the host has no AVX-512 *and* the shape is in the pathological region
  // (small IC + small batch). We do this by walking next_impl() and picking
  // the first non-brg impl. If none exists we fall through to the original
  // selection.
  static const bool host_has_avx512 = []() {
    // dnnl::get_effective_cpu_isa() reflects oneDNN's actual dispatch ISA,
    // which honours ONEDNN_MAX_CPU_ISA. On AMD EPYC Zen 2 this is avx2.
    // The ISA enum is a bitmask: AVX-512 variants share bit 0x20 of the
    // underlying integer; AVX-class ISAs (sse41/avx/avx2/avx2_vnni*) don't.
    const auto isa = static_cast<unsigned>(dnnl::get_effective_cpu_isa());
    constexpr unsigned avx512_core_bits =
        static_cast<unsigned>(dnnl::cpu_isa::avx512_core);
    return (isa & avx512_core_bits) == avx512_core_bits;
  }();
  // bs and IC come from the *logical* (NCHW/NCDHW/NCW) data shape. data_md
  // dim[0] is N, dim[1] is C, regardless of the eventual blocked layout.
  const auto src_dims         = data_md.get_dims();
  const dnnl_dim_t batch_size = src_dims.empty() ? 1 : src_dims[0];
  const dnnl_dim_t ic         = src_dims.size() < 2 ? 0 : src_dims[1];
  // simd_w on AVX2 is 8 f32 lanes; brg_conv pads IC up to this and beyond.
  const bool avoid_brg_heuristic =
      !host_has_avx512 && batch_size <= 1 && ic > 0 && ic < 8 &&
      !param.dnnl_param.quantized;

  auto GetConvFwdPd = [&param, &data, &weights, &output, avoid_brg_heuristic](
                          std::shared_ptr<dnnl::convolution_forward::primitive_desc> conv_pd) {
    try {
      while (conv_pd->dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->src_desc().get_size() != GetArraySize(data) ||
             (!param.dnnl_param.quantized &&
              conv_pd->weights_desc().get_size() != GetArraySize(weights))) {
        CHECK(conv_pd->next_impl()) << "No convolution implementation for this request.";
      }
      // FU-3: now that we have a *memory-compatible* impl, optionally skip
      // brg_conv:avx2 in favor of the next non-brg impl. We save the
      // current pd as a fallback in case every remaining impl is brg.
      if (avoid_brg_heuristic) {
        const char* info = conv_pd->impl_info_str();
        if (info && std::strstr(info, "brg") != nullptr) {
          auto saved = conv_pd;
          // Advance through brg candidates until we find a non-brg one that
          // is also memory-compatible. Reuse the same memory-size guards.
          while (true) {
            if (!conv_pd->next_impl()) {
              // No non-brg impl exists: restore the original brg pd.
              conv_pd = saved;
              break;
            }
            const bool sizes_ok =
                conv_pd->dst_desc().get_size() == GetArraySize(output) &&
                conv_pd->src_desc().get_size() == GetArraySize(data) &&
                (param.dnnl_param.quantized ||
                 conv_pd->weights_desc().get_size() == GetArraySize(weights));
            if (!sizes_ok) continue;
            const char* info2 = conv_pd->impl_info_str();
            if (!info2 || std::strstr(info2, "brg") == nullptr) {
              break;  // accept this non-brg, size-compatible impl
            }
          }
        }
      }
      return conv_pd;
    } catch (dnnl::error& e) {
      if (e.status == dnnl_unimplemented && param.dnnl_param.quantized) {
        LOG(ERROR) << "AVX512-BW support or Intel(R) MKL dependency is "
                      "required for int8 convolution";
      } else {
        LOG(ERROR) << e.message;
      }
      throw;
    }
  };

  // v3: ::desc removed; primitive_desc takes the args directly.
  if (bias_md_ptr == nullptr) {
    return GetConvFwdPd(std::make_shared<dnnl::convolution_forward::primitive_desc>(
        engine, prop, dnnl::algorithm::convolution_direct,
        data_md, weight_md, out_md,
        strides, dilates, padding, padding, attr));
  } else {
    return GetConvFwdPd(std::make_shared<dnnl::convolution_forward::primitive_desc>(
        engine, prop, dnnl::algorithm::convolution_direct,
        data_md, weight_md, *bias_md_ptr, out_md,
        strides, dilates, padding, padding, attr));
  }
}

static std::shared_ptr<dnnl::convolution_backward_data::primitive_desc> GetConvBwdData(
    const ConvolutionParam& param,
    const NDArray& data,
    const NDArray& weight,
    const NDArray& output,
    const dnnl::convolution_forward::primitive_desc& fwd_pd) {
  auto data_md   = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight, param.num_group);
  auto out_md    = GetMemDesc(output);
  auto engine    = CpuEngine::Get()->get_engine();
  dnnl::memory::dims strides(param.kernel.ndim());
  dnnl::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1);
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.dilate.ndim(), 1);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2);
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.dilate.ndim(), 2);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.kernel.ndim() == 3) {
    CHECK_GE(param.stride.ndim(), 3);
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.dilate.ndim(), 3);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
    padding[2] = param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected oneDNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2 or 3.";
  }

  auto GetConvBwdDataPd = [&data, &weight, &output, &fwd_pd](
                              std::shared_ptr<dnnl::convolution_backward_data::primitive_desc>
                                  conv_pd) {
    try {
      // DNNL introduced padded formats since 0.15 which require more memory
      // compared to the actual size of the tensor. Currently, DNNL operators
      // still reuse memory from memory planning, so here we need to select a
      // suboptimal kernel for computation that has the expected memory size requirements
      while (conv_pd->diff_dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->diff_src_desc().get_size() != GetArraySize(data) ||
             conv_pd->weights_desc().get_size() != GetArraySize(weight)) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd->next_impl()) << "No convolution backward implementation for this request.";
      }
      return conv_pd;
    } catch (dnnl::error& e) {
      LOG(ERROR) << e.message;
      throw;
    }
  };

  // v3: dilation is required; build it once (zeros if no dilation).
  const int kndim = param.kernel.ndim();
  dnnl::memory::dims dilates(kndim, 0);
  if (param.dilate.ndim() != 0) {
    CHECK_EQ(param.dilate.ndim(), kndim);
    for (int i = 0; i < kndim; ++i) {
      dilates[i] = param.dilate[i] - 1;
    }
  }

  // v3: ::desc removed; primitive_desc takes args directly.
  return GetConvBwdDataPd(std::make_shared<dnnl::convolution_backward_data::primitive_desc>(
      engine, dnnl::algorithm::convolution_direct,
      data_md, weight_md, out_md,
      strides, dilates, padding, padding, fwd_pd));
}

static std::shared_ptr<dnnl::convolution_backward_weights::primitive_desc> GetConvBwdWeights(
    const ConvolutionParam& param,
    const NDArray& data,
    const NDArray& weight,
    const NDArray* bias,
    const NDArray& output,
    const dnnl::convolution_forward::primitive_desc& fwd_pd) {
  auto data_md   = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight, param.num_group);
  auto out_md    = GetMemDesc(output);
  auto engine    = CpuEngine::Get()->get_engine();
  dnnl::memory::dims strides(param.kernel.ndim());
  dnnl::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1);
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.dilate.ndim(), 1);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2);
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.dilate.ndim(), 2);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.kernel.ndim() == 3) {
    CHECK_GE(param.stride.ndim(), 3);
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.dilate.ndim(), 3);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
    padding[2] = param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected oneDNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2 or 3.";
  }

  auto GetConvBwdWeightsPd = [&data, &weight, &output, &fwd_pd](
                                 std::shared_ptr<
                                     dnnl::convolution_backward_weights::primitive_desc> conv_pd) {
    try {
      // DNNL introduced padded formats since 0.15 which require more memory
      // compared to the actual size of the tensor. Currently, DNNL operators
      // still reuse memory from memory planning, so here we need to select a
      // suboptimal kernel for computation that has the expected memory size requirements
      while (conv_pd->diff_dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->src_desc().get_size() != GetArraySize(data) ||
             conv_pd->diff_weights_desc().get_size() != GetArraySize(weight)) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd->next_impl()) << "No convolution backward implementation for this request.";
      }
      return conv_pd;
    } catch (dnnl::error& e) {
      LOG(ERROR) << e.message;
      throw;
    }
  };

  // v3: dilation is required; build it once (zeros if no dilation).
  const int kndim = param.kernel.ndim();
  dnnl::memory::dims dilates(kndim, 0);
  if (param.dilate.ndim() != 0) {
    CHECK_EQ(param.dilate.ndim(), kndim);
    for (int i = 0; i < kndim; ++i) {
      dilates[i] = param.dilate[i] - 1;
    }
  }

  // v3: ::desc removed; primitive_desc takes args directly.
  if (bias == nullptr) {
    return GetConvBwdWeightsPd(
        std::make_shared<dnnl::convolution_backward_weights::primitive_desc>(
            engine, dnnl::algorithm::convolution_direct,
            data_md, weight_md, out_md,
            strides, dilates, padding, padding, fwd_pd));
  } else {
    auto bias_md = GetMemDesc(*bias);
    return GetConvBwdWeightsPd(
        std::make_shared<dnnl::convolution_backward_weights::primitive_desc>(
            engine, dnnl::algorithm::convolution_direct,
            data_md, weight_md, bias_md, out_md,
            strides, dilates, padding, padding, fwd_pd));
  }
}

DNNLConvForward::DNNLConvForward(const DNNLConvFullParam& param,
                                 const bool is_train,
                                 const NDArray& data,
                                 const NDArray& weight,
                                 const NDArray* bias,
                                 const NDArray& output)
    : pd_(GetConvFwdImpl(param, is_train, data, weight, bias, output)) {
  fwd_ = std::make_shared<dnnl::convolution_forward>(GetPd());
  // v3 quantized: runtime scale tensors. For int8/u8 dst, a single scale binds
  // on DNNL_ARG_DST (per-tensor) or DNNL_ARG_WEIGHTS (per-OC). For f32 dst
  // (dequant) the v3 kernel requires the scale be split across SRC and
  // WEIGHTS rather than DST; SRC gets 1/data_scale and WEIGHTS gets
  // 1/weight_scale[c]. Mirrors dnnl_fully_connected.cc.
  if (param.dnnl_param.quantized && param.requantize_scales.size()) {
    auto engine             = CpuEngine::Get()->get_engine();
    const bool float_output = param.dnnl_param.enabled_float_output.has_value();
    dnnl::memory::desc scale_md(
        dnnl::memory::dims{static_cast<dnnl::memory::dim>(param.requantize_scales.size())},
        dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    out_scale_mem_ = std::make_shared<dnnl::memory>(scale_md, engine);
    std::memcpy(out_scale_mem_->get_data_handle(),
                param.requantize_scales.data(),
                param.requantize_scales.size() * sizeof(float));
    if (float_output) {
      out_scale_arg_ = DNNL_ARG_WEIGHTS;
      dnnl::memory::desc src_scale_md(
          dnnl::memory::dims{1},
          dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
      src_scale_mem_ = std::make_shared<dnnl::memory>(src_scale_md, engine);
      *reinterpret_cast<float*>(src_scale_mem_->get_data_handle()) = param.src_scale;
    } else {
      out_scale_arg_ = (param.requantize_scales.size() > 1) ? DNNL_ARG_WEIGHTS : DNNL_ARG_DST;
    }
  }
}

DNNLConvForward& GetConvFwd(const DNNLConvFullParam& param,
                            const bool is_train,
                            const NDArray& data,
                            const NDArray& weight,
                            const NDArray* bias,
                            const NDArray& output) {
  using conv_fwd_map = std::unordered_map<DNNLConvSignature, DNNLConvForward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local conv_fwd_map fwds;
#else
  static MX_THREAD_LOCAL conv_fwd_map fwds;
#endif
  // TODO(zhennan): Hash conv_param for now, need to hash full param if we want to enable cache for
  // fused conv
  DNNLConvSignature key(param.conv_param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will decide the right layout
  // for the, so we only need to get the shape and the data type of the arrays.
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd = DNNLConvForward(param, is_train, data, weight, bias, output);
    it       = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLConvolutionForwardFullFeature(const DNNLConvFullParam& param,
                                       const OpContext& ctx,
                                       DNNLConvForward* fwd,
                                       const std::vector<NDArray>& in_data,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);

  auto& data   = in_data[conv::kData];
  auto& weight = in_data[conv::kWeight];
  bool no_bias = param.conv_param.no_bias && !param.dnnl_param.with_bn;

  auto fwd_src_desc = fwd->GetPd().src_desc();
  auto data_mem     = data.GetDNNLDataReorder(&fwd_src_desc);
  const dnnl::memory* weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle DNNL correctly. Let's reorder it to the default format
    // for now.
    if (weight.IsDNNLData())
      // This asks the engine to change the layout of the weight array after it's used.
      weight.Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd->GetPd().weights_desc(), param.conv_param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to reorder data every
    // time.
    if (weight.IsDefaultData()) {
      // We also need to modify the layout on the original weight array. The data conversion happens
      // after the weight array is used.
      auto fwd_weight_desc = fwd->GetPd().weights_desc();
      weight.DNNLDataReorderAsync(&fwd_weight_desc);
      weight_mem = GetWeights(weight, fwd->GetPd().weights_desc(), param.conv_param.num_group);
    } else {
      auto fwd_weight_desc = fwd->GetPd().weights_desc();
      weight_mem           = weight.GetDNNLDataReorder(&fwd_weight_desc);
    }
  }
  dnnl_output_t out_mem;
  if (param.dnnl_param.with_sum) {
    out_mem = dnnl_output_t(OutDataOp::Noop,
                            const_cast<dnnl::memory*>(out_data[conv::kOut].GetDNNLData()));
  } else {
    out_mem = CreateDNNLMem(out_data[conv::kOut], fwd->GetPd().dst_desc(), req[conv::kOut]);
  }

  dnnl_args_map_t net_args;
  if (!no_bias) {
    const dnnl::memory* bias_mem = in_data[conv::kBias].GetDNNLData();
    net_args.insert({DNNL_ARG_BIAS, *bias_mem});
  }

  net_args.insert({DNNL_ARG_SRC, *data_mem});
  net_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  // v3: bind the runtime DNNL_ARG_DST scale tensor declared via
  // attr.set_scales_mask in GetConvFwdImpl for quantized requantize output.
  if (auto* sm = fwd->GetOutputScaleMem()) {
    net_args.insert({DNNL_ARG_ATTR_SCALES | fwd->GetOutputScaleArg(), *sm});
  }
  if (auto* ssm = fwd->GetSrcScaleMem()) {
    net_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *ssm});
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd->GetFwd(), net_args);
  CommitOutput(out_data[conv::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

void DNNLConvolutionForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& out_data) {
  DNNLConvFullParam param;
  param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  param.dnnl_param.Init(std::unordered_map<std::string, std::string>());
  auto& fwd = GetConvFwd(param,
                         ctx.is_train,
                         in_data[conv::kData],
                         in_data[conv::kWeight],
                         param.conv_param.no_bias ? nullptr : &in_data[conv::kBias],
                         out_data[conv::kOut]);
  DNNLConvolutionForwardFullFeature(param, ctx, &fwd, in_data, req, out_data);
}

DNNLConvBackward::DNNLConvBackward(const DNNLConvFullParam& param,
                                   const NDArray& data,
                                   const NDArray& weight,
                                   const NDArray* bias,
                                   const NDArray& output) {
  const auto fwd_pd = GetConvFwdImpl(param, true, data, weight, bias, output);
  bwd_data_pd_      = GetConvBwdData(param.conv_param, data, weight, output, *fwd_pd);
  bwd_weight_pd_    = GetConvBwdWeights(param.conv_param, data, weight, bias, output, *fwd_pd);
  bwd_data_         = std::make_shared<dnnl::convolution_backward_data>(GetDataPd());
  bwd_weight_       = std::make_shared<dnnl::convolution_backward_weights>(GetWeightsPd());
}

static inline DNNLConvBackward& GetConvBwd(const DNNLConvFullParam& param,
                                           const NDArray& data,
                                           const NDArray& weight,
                                           const NDArray* bias,
                                           const NDArray& output) {
  using dnnl_conv_bwd_map = std::unordered_map<DNNLConvSignature, DNNLConvBackward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local dnnl_conv_bwd_map bwds;
#else
  static MX_THREAD_LOCAL dnnl_conv_bwd_map bwds;
#endif
  // TODO(zhennan): Hash conv_param for now, need to hash full param if we want to enable cache for
  // fused conv
  DNNLConvSignature key(param.conv_param);
  // Here we can sign the conv op with NDArray because conv primitive will decide the right layout
  // for the, so we only need to get the shape and the data type of the arrays.
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd = DNNLConvBackward(param, data, weight, bias, output);
    it       = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void DNNLConvolutionBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray>& in_grad = outputs;
  DNNLConvFullParam full_param;
  full_param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  full_param.dnnl_param.Init(std::unordered_map<std::string, std::string>());

  auto& data       = inputs[conv::kData + 1];
  auto& weight     = inputs[conv::kWeight + 1];
  const auto* bias = full_param.conv_param.no_bias ? nullptr : &inputs[conv::kBias + 1];
  auto& out_grad   = inputs[conv::kOut];

  const ConvolutionParam& param = full_param.conv_param;

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  DNNLConvBackward& convBwd   = GetConvBwd(full_param, data, weight, bias, out_grad);
  auto convBwd_data_diff_desc = convBwd.GetDataPd().diff_dst_desc();
  auto out_grad_mem           = out_grad.GetDNNLDataReorder(&convBwd_data_diff_desc);
  if (req[conv::kData]) {
    auto weight_mem = GetWeights(weight, convBwd.GetDataPd().weights_desc(), param.num_group);
    auto in_grad_mem =
        CreateDNNLMem(in_grad[conv::kData], convBwd.GetDataPd().diff_src_desc(), req[conv::kData]);
    DNNLStream::Get()->RegisterPrimArgs(convBwd.GetBwdData(),
                                        {{DNNL_ARG_DIFF_DST, *out_grad_mem},
                                         {DNNL_ARG_WEIGHTS, *weight_mem},
                                         {DNNL_ARG_DIFF_SRC, *in_grad_mem.second}});
    CommitOutput(in_grad[conv::kData], in_grad_mem);
  }

  auto req_weight = req.size() > conv::kWeight ? req.at(conv::kWeight) : kNullOp;
  auto req_bias   = req.size() > conv::kBias ? req.at(conv::kBias) : kNullOp;
  if (req_weight || req_bias) {
    if (convBwd.GetDataPd().diff_dst_desc() != convBwd.GetWeightsPd().diff_dst_desc()) {
      auto convBwd_weight_diff_desc = convBwd.GetWeightsPd().diff_dst_desc();
      out_grad_mem                  = out_grad.GetDNNLDataReorder(&convBwd_weight_diff_desc);
    }
    auto convBwd_weight_src_desc = convBwd.GetWeightsPd().src_desc();
    auto data_mem                = data.GetDNNLDataReorder(&convBwd_weight_src_desc);
    auto in_grad_weight          = CreateDNNLWeightGrad(
        in_grad[conv::kWeight], convBwd.GetWeightsPd().diff_weights_desc(), req[conv::kWeight]);

    dnnl_args_map_t net_args = {{DNNL_ARG_DIFF_DST, *out_grad_mem},
                                {DNNL_ARG_SRC, *data_mem},
                                {DNNL_ARG_DIFF_WEIGHTS, *in_grad_weight.second}};
    dnnl_output_t in_grad_bias;
    if (!param.no_bias) {
      in_grad_bias = CreateDNNLMem(
          in_grad[conv::kBias], convBwd.GetWeightsPd().diff_bias_desc(), req[conv::kBias]);
      net_args.insert({DNNL_ARG_DIFF_BIAS, *in_grad_bias.second});
    }
    DNNLStream::Get()->RegisterPrimArgs(convBwd.GetBwdWeights(), net_args);
    CommitOutput(in_grad[conv::kWeight], in_grad_weight);
    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    if (!param.no_bias) {
      CommitOutput(in_grad[conv::kBias], in_grad_bias);
    }
  }
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
