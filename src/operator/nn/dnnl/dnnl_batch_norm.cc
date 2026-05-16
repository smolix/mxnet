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
  // v3 has separate scale_desc()/shift_desc() helpers (both 1-D, length=C).
  auto engine = CpuEngine::Get()->get_engine();
  scale_m.reset(new dnnl::memory(pd.weights_desc(), engine));
  shift_m.reset(new dnnl::memory(pd.weights_desc(), engine));
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
  auto out_mem      = const_cast<NDArray&>(out).CreateDNNLData(&fwd_dst_desc);

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
    net_args[DNNL_ARG_DST]   = *out_mem;
    if (!ctx.is_train || param.use_global_stats) {
      float* omean  = outputs[batchnorm::kMean].data().dptr<float>();
      float* ovar   = outputs[batchnorm::kVar].data().dptr<float>();
      float* inmean = aux_states[batchnorm::kMovingMean].data().dptr<float>();
      float* invar  = aux_states[batchnorm::kMovingVar].data().dptr<float>();
      // to align with origin implmentation: batch_norm.cc: L164
      for (index_t i = 0; i < channels_; i++) {
        omean[i] = inmean[i];
        ovar[i]  = VARIANCE_TO_INVSTD(invar[i], param.eps);
      }
      net_args[DNNL_ARG_MEAN]     = *(aux_states[batchnorm::kMovingMean].GetDNNLData());
      net_args[DNNL_ARG_VARIANCE] = *(aux_states[batchnorm::kMovingVar].GetDNNLData());
      DNNLStream::Get()->RegisterPrimArgs(GetFwd(), net_args);
      DNNLStream::Get()->Submit();
    } else {  // training
      const NDArray& outMean      = outputs[batchnorm::kMean];
      const NDArray& outVar       = outputs[batchnorm::kVar];
      net_args[DNNL_ARG_MEAN]     = *(outMean.GetDNNLData());
      net_args[DNNL_ARG_VARIANCE] = *(outVar.GetDNNLData());
      DNNLStream::Get()->RegisterPrimArgs(GetFwd(), net_args);
      DNNLStream::Get()->Submit();

      float* ovar = outVar.data().dptr<float>();
      for (index_t i = 0; i < channels_; i++) {
        ovar[i] = VARIANCE_TO_INVSTD(ovar[i], param.eps);
      }
    }
  } else {  // no input gamma and beta
    LOG(FATAL) << "oneDNN batch normalization: should not reach here ...";
  }
}

DNNLBNBackward::DNNLBNBackward(const t_bn_b_pdesc& _pd)
    : scale_m(new dnnl::memory(_pd.weights_desc(), CpuEngine::Get()->get_engine())),
      shift_m(new dnnl::memory(_pd.weights_desc(), CpuEngine::Get()->get_engine())),
      grad_scale_m(
          new dnnl::memory(_pd.diff_weights_desc(), CpuEngine::Get()->get_engine())),
      grad_shift_m(
          new dnnl::memory(_pd.diff_weights_desc(), CpuEngine::Get()->get_engine())),
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
  key.AddSign(in_data);
  key.AddSign(diff_data);
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
  auto gradi_mem =
      CreateDNNLMem(const_cast<NDArray&>(gradIn), pd.diff_src_desc(), req[batchnorm::kData]);

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
    dnnl_args_map_t net_args;
    net_args[DNNL_ARG_SRC]        = *data_mem;
    net_args[DNNL_ARG_DIFF_SRC]   = *gradi_mem.second;
    net_args[DNNL_ARG_SCALE]      = GetScale();
    net_args[DNNL_ARG_SHIFT]      = GetShift();
    net_args[DNNL_ARG_DIFF_SCALE] = GetGradScale();
    net_args[DNNL_ARG_DIFF_SHIFT] = GetGradShift();
    net_args[DNNL_ARG_DIFF_DST]   = *diff_mem;

    // training but no input mean and variance
    if (ctx.is_train && !param.use_global_stats) {
      float* moving_mean_ptr = moving_mean.data().dptr<float>();
      float* moving_var_ptr  = moving_var.data().dptr<float>();
      float* out_mean_ptr    = out_mean.data().dptr<float>();
      float* out_var_ptr     = out_var.data().dptr<float>();
      dnnl::memory var_mem(pd.variance_desc(), CpuEngine::Get()->get_engine());
      float* tmp_var_ptr = reinterpret_cast<float*>(var_mem.get_data_handle());

      float minus_mom = (1.0f - param.momentum);
      for (index_t i = 0; i < channels_; i++) {
        moving_mean_ptr[i] = moving_mean_ptr[i] * param.momentum + out_mean_ptr[i] * minus_mom;
        float variance     = INVSTD_TO_VARIANCE(out_var_ptr[i], param.eps);
        tmp_var_ptr[i]     = variance;
        moving_var_ptr[i]  = moving_var_ptr[i] * param.momentum + variance * minus_mom;
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
    float* w_grad_1    = in_grad[batchnorm::kGamma].data().dptr<float>();
    float* w_grad_2    = in_grad[batchnorm::kBeta].data().dptr<float>();

    // the gradient of gamma
    if (!param.fix_gamma) {
      if (req[batchnorm::kGamma] != kNullOp) {
        if (req[batchnorm::kGamma] != kAddTo) {
          memcpy(w_grad_1, g_scale_buf, copy_size);
        } else {
          for (index_t i = 0; i < channels_; i++) {
            w_grad_1[i] += g_scale_buf[i];
          }
        }
      }
    } else {
      for (index_t i = 0; i < channels_; i++) {
        (in_grad[1].data().dptr<float>())[i] = 0.0f;
      }
    }

    // the gradient of beta
    if (req[batchnorm::kBeta] != kNullOp) {
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
