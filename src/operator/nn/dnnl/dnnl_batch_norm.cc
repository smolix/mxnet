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
 * \file dnnl_batch_norm.cc
 */

#if MXNET_USE_ONEDNN == 1
#include <dnnl.hpp>

#include <utility>
#include <vector>

#include "dnnl_base-inl.h"
#include "dnnl_batch_norm-inl.h"
#include "operator/nn/batch_norm-inl.h"

namespace mxnet {
namespace op {

typedef dnnl::batch_normalization_forward::primitive_desc t_bn_f_pdesc;

typedef dnnl::batch_normalization_backward::primitive_desc t_bn_b_pdesc;


DNNLBNForward::DNNLBNForward(const t_bn_f_pdesc& _pd, bool is_train_and_not_global_stats)
    : pd(_pd) {
  // v3 split SCALE_SHIFT into separate scale + shift; each is a 1-D f32
  // tensor of length C. weights_desc() in v3 may return an invalid /
  // empty desc when use_scale|use_shift flags are used, so build the
  // 1-D length-C desc explicitly.
  auto engine = CpuEngine::Get()->get_engine();
  const auto channels = pd.src_desc().get_dims()[1];
  const dnnl::memory::desc scale_shift_md(
      {channels}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
  scale_m.reset(new dnnl::memory(scale_shift_md, engine));
  shift_m.reset(new dnnl::memory(scale_shift_md, engine));
  fwd.reset(new dnnl::batch_normalization_forward(pd));
  this->is_train_and_not_global_stats = is_train_and_not_global_stats;
}

const dnnl::memory& DNNLBNForward::GetScale() const {
  return *scale_m;
}

const dnnl::memory& DNNLBNForward::GetShift() const {
  return *shift_m;
}

const t_bn_f_pdesc& DNNLBNForward::GetPd() const {
  return pd;
}

const dnnl::batch_normalization_forward& DNNLBNForward::GetFwd() const {
  return *fwd;
}

DNNLBNForward& DNNLBNForward::GetCached(const BatchNormParam& param,
                                        const OpContext& ctx,
                                        const dnnl::memory* data_mem,
                                        bool fuse_relu,
                                        dnnl::normalization_flags flags) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLBNSignature, DNNLBNForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLBNSignature, DNNLBNForward, OpHash> fwds;
#endif

  DNNLBNSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(*data_mem);
  key.AddSign(static_cast<int>(flags));
  key.AddSign(fuse_relu);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd_pd = _GetFwd(*data_mem, ctx.is_train, fuse_relu, param.eps, flags);
    DNNLBNForward fwd(fwd_pd, ctx.is_train && !param.use_global_stats);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLBNForward::Execute(const OpContext& ctx,
                            const BatchNormParam& param,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs,
                            bool fuse_relu) {
  std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);

  mxnet::TShape shape = inputs[batchnorm::kData].shape();
  const int real_axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  CHECK_LT(real_axis, shape.ndim());
  NDArray out = outputs[batchnorm::kOut];
  if (param.axis != 1 || shape.ndim() != 4) {
    // reshape to (N, C, 1, D)
    mxnet::TShape new_shape{
        static_cast<index_t>(shape.ProdShape(0, real_axis)),
        shape[real_axis],
        1,
        static_cast<index_t>(shape.ProdShape(real_axis + 1, static_cast<int>(shape.ndim())))};
    in_data[batchnorm::kData] = in_data[batchnorm::kData].Reshape(new_shape);
    out                       = out.Reshape(new_shape);
  }

  const std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  dnnl::normalization_flags flags =
      _GetFlags(in_data, aux_states, ctx.is_train && !param.use_global_stats);
  NDArray& data = in_data[batchnorm::kData];
  if (data.IsDNNLData() && data.IsView())
    data = data.Reorder2Default();
  auto data_mem = data.GetDNNLData();

  // for output memory
  auto fwd_dst_desc = GetPd().dst_desc();
  auto out_mem      = CreateDNNLMem(out, fwd_dst_desc, req[batchnorm::kOut], &data);

  // mxnet will always use scale shift.
  // But if fix_gamma is true, then all scale elements will be set to 1.0f
  // v3: use_scale_shift was split into use_scale | use_shift.
  if (static_cast<int>(flags) & static_cast<int>(dnnl::normalization_flags::use_scale)) {
    const NDArray& gamma = in_data[batchnorm::kGamma];
    const NDArray& beta  = in_data[batchnorm::kBeta];
    CHECK_EQ(gamma.storage_type(), mxnet::kDefaultStorage);
    CHECK_EQ(beta.storage_type(), mxnet::kDefaultStorage);

    const dnnl::memory& scale_mem = GetScale();
    const dnnl::memory& shift_mem = GetShift();
    float* scale_buf = reinterpret_cast<float*>(scale_mem.get_data_handle());
    float* shift_buf = reinterpret_cast<float*>(shift_mem.get_data_handle());

    index_t channels_      = data.shape()[1];
    CHECK(scale_mem.get_desc().get_size() == channels_ * sizeof(float));
    CHECK(shift_mem.get_desc().get_size() == channels_ * sizeof(float));
    float* weight_ptr      = gamma.data().dptr<float>();
    float* bias_ptr        = beta.data().dptr<float>();
    const size_t copy_size = sizeof(float) * channels_;
    if (!param.fix_gamma) {
      memcpy(scale_buf, weight_ptr, copy_size);
      memcpy(shift_buf, bias_ptr, copy_size);
    } else if (IsBNWriting(req[batchnorm::kGamma])) {
      for (index_t i = 0; i < channels_; i++) {
        scale_buf[i]   = 1.0f;
        weight_ptr[i]  = 1.0f;
        shift_buf[i]   = bias_ptr[i];
      }
    } else {
      for (index_t i = 0; i < channels_; i++) {
        scale_buf[i] = 1.0f;
        shift_buf[i] = bias_ptr[i];
      }
    }

    dnnl_args_map_t net_args;
    net_args[DNNL_ARG_SRC]   = *data_mem;
    net_args[DNNL_ARG_SCALE] = scale_mem;
    net_args[DNNL_ARG_SHIFT] = shift_mem;
    net_args[DNNL_ARG_DST]   = *out_mem.second;
    if (!ctx.is_train || param.use_global_stats) {
      float* omean  = outputs[batchnorm::kMean].data().dptr<float>();
      float* ovar   = outputs[batchnorm::kVar].data().dptr<float>();
      float* inmean = aux_states[batchnorm::kMovingMean].data().dptr<float>();
      float* invar  = aux_states[batchnorm::kMovingVar].data().dptr<float>();
      // to align with origin implmentation: batch_norm.cc: L164
      for (index_t i = 0; i < channels_; i++) {
        KERNEL_ASSIGN(omean[i], req[batchnorm::kMean], inmean[i]);
        KERNEL_ASSIGN(ovar[i], req[batchnorm::kVar], VARIANCE_TO_INVSTD(invar[i], param.eps));
      }
      net_args[DNNL_ARG_MEAN]     = *(aux_states[batchnorm::kMovingMean].GetDNNLData());
      net_args[DNNL_ARG_VARIANCE] = *(aux_states[batchnorm::kMovingVar].GetDNNLData());
      DNNLStream::Get()->RegisterPrimArgs(GetFwd(), net_args);
      CommitOutput(out, out_mem);
      DNNLStream::Get()->Submit();
    } else {  // training
      const NDArray& outMean      = outputs[batchnorm::kMean];
      const NDArray& outVar       = outputs[batchnorm::kVar];
      NDArray saved_mean(outMean.shape(), outMean.ctx(), false, outMean.dtype());
      NDArray saved_var(outVar.shape(), outVar.ctx(), false, outVar.dtype());
      auto saved_mean_mem         = saved_mean.GetDNNLData();
      auto saved_var_mem          = saved_var.GetDNNLData();
      net_args[DNNL_ARG_MEAN]     = *saved_mean_mem;
      net_args[DNNL_ARG_VARIANCE] = *saved_var_mem;
      DNNLStream::Get()->RegisterPrimArgs(GetFwd(), net_args);
      CommitOutput(out, out_mem);
      DNNLStream::Get()->Submit();

      // Update running mean/variance here in the forward pass so that they
      // are visible immediately after forward() — consistent with the cuDNN
      // (GPU) path which updates them inside cudnnBatchNormalizationForwardTrainingEx.
      // The backward pass previously did this update; now that it happens here
      // the backward pass skips it (see DNNLBNBackward::Execute).
      float* moving_mean_ptr = aux_states[batchnorm::kMovingMean].data().dptr<float>();
      float* moving_var_ptr  = aux_states[batchnorm::kMovingVar].data().dptr<float>();
      NDArray saved_mean_buffer = saved_mean.Reorder2Default();
      NDArray saved_var_buffer  = saved_var.Reorder2Default();
      float* saved_mean_ptr     = saved_mean_buffer.data().dptr<float>();
      float* saved_var_ptr      = saved_var_buffer.data().dptr<float>();
      float* out_mean_ptr       = outMean.data().dptr<float>();
      float* out_var_ptr        = outVar.data().dptr<float>();
      float minus_mom        = 1.0f - param.momentum;
      for (index_t i = 0; i < channels_; i++) {
        moving_mean_ptr[i] = moving_mean_ptr[i] * param.momentum + saved_mean_ptr[i] * minus_mom;
        // oneDNN returns raw variance; MXNet saves inv-std for backward.
        moving_var_ptr[i] = moving_var_ptr[i] * param.momentum + saved_var_ptr[i] * minus_mom;
        KERNEL_ASSIGN(out_mean_ptr[i], req[batchnorm::kMean], saved_mean_ptr[i]);
        KERNEL_ASSIGN(out_var_ptr[i],
                      req[batchnorm::kVar],
                      VARIANCE_TO_INVSTD(saved_var_ptr[i], param.eps));
      }
    }
  } else {  // no input gamma and beta
    LOG(FATAL) << "oneDNN batch normalization: should not reach here ...";
  }
}

// v3: build 1-D length-C f32 desc for each of scale/shift/diff_scale/diff_shift
//     (weights_desc/diff_weights_desc may not be valid when use_scale|use_shift
//     flags are used).
static dnnl::memory::desc BnScaleShiftMd(const t_bn_b_pdesc& pd) {
  const auto channels = pd.src_desc().get_dims()[1];
  return dnnl::memory::desc({channels}, dnnl::memory::data_type::f32,
                            dnnl::memory::format_tag::a);
}

DNNLBNBackward::DNNLBNBackward(const t_bn_b_pdesc& _pd)
    : scale_m(new dnnl::memory(BnScaleShiftMd(_pd), CpuEngine::Get()->get_engine())),
      shift_m(new dnnl::memory(BnScaleShiftMd(_pd), CpuEngine::Get()->get_engine())),
      grad_scale_m(new dnnl::memory(BnScaleShiftMd(_pd), CpuEngine::Get()->get_engine())),
      grad_shift_m(new dnnl::memory(BnScaleShiftMd(_pd), CpuEngine::Get()->get_engine())),
      pd(_pd) {
  bwd.reset(new dnnl::batch_normalization_backward(pd));
}

const dnnl::memory& DNNLBNBackward::GetScale() const { return *scale_m; }
const dnnl::memory& DNNLBNBackward::GetShift() const { return *shift_m; }
const dnnl::memory& DNNLBNBackward::GetGradScale() const { return *grad_scale_m; }
const dnnl::memory& DNNLBNBackward::GetGradShift() const { return *grad_shift_m; }

const dnnl::batch_normalization_backward& DNNLBNBackward::GetBwd() const {
  return *bwd;
}

DNNLBNBackward& DNNLBNBackward::GetCached(const BatchNormParam& param,
                                          const OpContext& ctx,
                                          const NDArray& in_data,
                                          const dnnl::memory& in_mem,
                                          const NDArray& diff_data,
                                          const dnnl::memory& diff_mem,
                                          dnnl::normalization_flags flags) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLBNSignature, DNNLBNBackward, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLBNSignature, DNNLBNBackward, OpHash> bwds;
#endif
  DNNLBNSignature key(param);
  key.AddSign(in_mem);
  key.AddSign(diff_mem);
  key.AddSign(ctx.is_train);
  key.AddSign(static_cast<int>(flags));

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd_pd = _GetBwd(in_mem, diff_mem, param.eps, flags);
    DNNLBNBackward bwd(bwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void DNNLBNBackward::Execute(const BatchNormParam& param,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 8U);
  std::vector<NDArray> out_grad(1);
  std::vector<NDArray> out_data(3);
  std::vector<NDArray> in_data(3);
  std::vector<NDArray> aux_states(2);
  out_grad[0]                         = inputs[0];
  out_data[batchnorm::kMean]          = inputs[1];
  out_data[batchnorm::kVar]           = inputs[2];
  in_data[batchnorm::kData]           = inputs[3];
  in_data[batchnorm::kGamma]          = inputs[4];
  in_data[batchnorm::kBeta]           = inputs[5];
  aux_states[batchnorm::kMovingMean]  = inputs[6];
  aux_states[batchnorm::kMovingVar]   = inputs[7];
  const std::vector<NDArray>& in_grad = outputs;
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  dnnl::normalization_flags flags =
      _GetFlags(in_data, aux_states, ctx.is_train && !param.use_global_stats);

  NDArray data               = in_data[batchnorm::kData];
  NDArray diff               = out_grad[batchnorm::kOut];
  NDArray gradIn             = in_grad[batchnorm::kData];
  const NDArray& moving_mean = aux_states[batchnorm::kMovingMean];
  const NDArray& moving_var  = aux_states[batchnorm::kMovingVar];
  const NDArray& out_mean    = out_data[batchnorm::kMean];
  const NDArray& out_var     = out_data[batchnorm::kVar];

  CHECK(out_mean.IsDefaultData());
  CHECK(out_var.IsDefaultData());
  CHECK(moving_mean.IsDefaultData());
  CHECK(moving_var.IsDefaultData());

  mxnet::TShape shape = data.shape();
  const int real_axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  CHECK_LT(real_axis, shape.ndim());
  if (param.axis != 1 || shape.ndim() != 4) {
    // reshape to (N, C, 1, D)
    mxnet::TShape new_shape{
        static_cast<index_t>(shape.ProdShape(0, real_axis)),
        shape[real_axis],
        1,
        static_cast<index_t>(shape.ProdShape(real_axis + 1, static_cast<int>(shape.ndim())))};
    data   = data.Reshape(new_shape);
    diff   = diff.Reshape(new_shape);
    gradIn = gradIn.Reshape(new_shape);
  }

  if (data.IsDNNLData()) {
    data = data.Reorder2Default();
  }
  if (diff.IsDNNLData()) {
    diff = diff.Reorder2Default();
  }

  auto data_mem = data.GetDNNLData();
  auto diff_mem = diff.GetDNNLData();
  // DNNL batchnorm should run on special layouts. If one of them isn't, we
  // should reorder them.
  if (data.IsDefaultData()) {
    auto diff_desc = diff_mem->get_desc();
    data_mem       = data.GetDNNLDataReorder(&diff_desc);
  } else if (diff.IsDefaultData()) {
    auto data_desc = data_mem->get_desc();
    diff_mem       = diff.GetDNNLDataReorder(&data_desc);
  }
  dnnl_output_t gradi_mem;
  if (req[batchnorm::kData] == kAddTo) {
    gradi_mem = dnnl_output_t(OutDataOp::AddBack, TmpMemMgr::Get()->Alloc(pd.diff_src_desc()));
  } else if (IsBNWriting(req[batchnorm::kData])) {
    gradi_mem = dnnl_output_t(OutDataOp::CopyBack, TmpMemMgr::Get()->Alloc(pd.diff_src_desc()));
  } else {
    gradi_mem = CreateDNNLMem(gradIn, pd.diff_src_desc(), req[batchnorm::kData]);
  }

  // v3: use_scale_shift was split into use_scale | use_shift.
  if (static_cast<int>(flags) & static_cast<int>(dnnl::normalization_flags::use_scale)) {
    const NDArray& gamma   = in_data[batchnorm::kGamma];
    const NDArray& beta    = in_data[batchnorm::kBeta];
    float* scale_buf       = reinterpret_cast<float*>(GetScale().get_data_handle());
    float* shift_buf       = reinterpret_cast<float*>(GetShift().get_data_handle());
    index_t channels_      = data.shape()[1];
    float* weight_ptr      = gamma.data().dptr<float>();
    float* bias_ptr        = beta.data().dptr<float>();
    const size_t copy_size = sizeof(float) * channels_;
    if (!param.fix_gamma) {
      memcpy(scale_buf, weight_ptr, copy_size);
    } else {
      for (index_t i = 0; i < channels_; i++) {
        scale_buf[i] = 1.0f;
      }
    }
    memcpy(shift_buf, bias_ptr, copy_size);
    // v3: backward batch_normalization needs DNNL_ARG_SCALE but NOT
    //     DNNL_ARG_SHIFT (beta is not used in gradient computation; only
    //     gamma feeds the chain rule). Strictly v3 rejects the extra arg.
    dnnl_args_map_t net_args;
    net_args[DNNL_ARG_SRC]        = *data_mem;
    net_args[DNNL_ARG_DIFF_SRC]   = *gradi_mem.second;
    net_args[DNNL_ARG_SCALE]      = GetScale();
    net_args[DNNL_ARG_DIFF_SCALE] = GetGradScale();
    net_args[DNNL_ARG_DIFF_SHIFT] = GetGradShift();
    net_args[DNNL_ARG_DIFF_DST]   = *diff_mem;

    // training but no input mean and variance
    if (ctx.is_train && !param.use_global_stats) {
      // Running mean/variance were already updated in the forward pass
      // (DNNLBNForward::Execute training branch).  Here we only need to
      // supply the saved batch mean and the reconstructed batch variance to
      // the oneDNN backward primitive.
      float* out_var_ptr = out_var.data().dptr<float>();
      dnnl::memory var_mem(pd.variance_desc(), CpuEngine::Get()->get_engine());
      float* tmp_var_ptr = reinterpret_cast<float*>(var_mem.get_data_handle());
      index_t channels_  = data.shape()[1];
      for (index_t i = 0; i < channels_; i++) {
        // out_var holds inv-std; convert back to variance for oneDNN backward.
        tmp_var_ptr[i] = INVSTD_TO_VARIANCE(out_var_ptr[i], param.eps);
      }
      net_args[DNNL_ARG_MEAN]     = *(out_mean.GetDNNLData());
      net_args[DNNL_ARG_VARIANCE] = var_mem;
    } else {
      net_args[DNNL_ARG_MEAN]     = *(moving_mean.GetDNNLData());
      net_args[DNNL_ARG_VARIANCE] = *(moving_var.GetDNNLData());
    }
    DNNLStream::Get()->RegisterPrimArgs(GetBwd(), net_args);
    CommitOutput(gradIn, gradi_mem);
    DNNLStream::Get()->Submit();

    // v3: split scale gradient (length C) and shift gradient (length C).
    float* g_scale_buf = reinterpret_cast<float*>(GetGradScale().get_data_handle());
    float* g_shift_buf = reinterpret_cast<float*>(GetGradShift().get_data_handle());

    // the gradient of gamma
    if (!param.fix_gamma) {
      if (req[batchnorm::kGamma] != kNullOp) {
        float* w_grad_1 = in_grad[batchnorm::kGamma].data().dptr<float>();
        if (req[batchnorm::kGamma] != kAddTo) {
          memcpy(w_grad_1, g_scale_buf, copy_size);
        } else {
          for (index_t i = 0; i < channels_; i++) {
            w_grad_1[i] += g_scale_buf[i];
          }
        }
      }
    } else if (req[batchnorm::kGamma] != kNullOp) {
      float* w_grad_1 = in_grad[batchnorm::kGamma].data().dptr<float>();
      if (req[batchnorm::kGamma] != kAddTo) {
        for (index_t i = 0; i < channels_; i++) {
          w_grad_1[i] = 0.0f;
        }
      }
    }

    // the gradient of beta
    if (req[batchnorm::kBeta] != kNullOp) {
      float* w_grad_2 = in_grad[batchnorm::kBeta].data().dptr<float>();
      if (req[batchnorm::kBeta] != kAddTo) {
        memcpy(w_grad_2, g_shift_buf, copy_size);
      } else {
        for (index_t i = 0; i < channels_; i++) {
          w_grad_2[i] += g_shift_buf[i];
        }
      }
    }
  } else {
    LOG(FATAL) << "oneDNN batch normalization backward: should not reach here ...";
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
