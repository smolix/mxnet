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
 * \file cudnn_batch_norm.cu
 * \brief
 * \author Junyuan Xie, Da Zheng
 */

#include "cudnn_batch_norm.h"

#include "../../../common/cuda/utils.h"

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1

namespace {

struct Globals {
  cudnnTensorDescriptor_t io_desc;
  cudnnTensorDescriptor_t mean_desc;
  bool internal_aux_states_lock = false;

  static Globals& Get() {
    thread_local Globals ret;
    return ret;
  }

  Globals() {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&io_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&mean_desc));
  }

  ~Globals() {
    CUDNN_CALL_NONFATAL(cudnnDestroyTensorDescriptor(io_desc));
    CUDNN_CALL_NONFATAL(cudnnDestroyTensorDescriptor(mean_desc));
  }
};

void SetDescriptors(const BatchNormParam& param, const TBlob& x) {
  CHECK_GE(x.shape_.ndim(), 3);
  CHECK(param.axis == 1 || param.axis == x.shape_.ndim() - 1);

  cudnnTensorFormat_t format = param.axis == 1 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  int n                      = x.shape_[0];
  int c                      = x.shape_[param.axis];
  size_t last_spatial_i      = param.axis == 1 ? x.shape_.ndim() - 1 : x.shape_.ndim() - 2;
  int w                      = x.shape_[last_spatial_i];
  int h = x.shape_.ProdShape(last_spatial_i - (x.shape_.ndim() - 3), last_spatial_i);

  MSHADOW_REAL_TYPE_SWITCH(x.type_flag_, DType, {
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        Globals::Get().io_desc, format, mshadow::DataType<DType>::kCudnnFlag, n, c, h, w));
  })
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(
      Globals::Get().mean_desc, Globals::Get().io_desc, CUDNN_BATCHNORM_SPATIAL));
}

mshadow::TypeFlag ParamType(int x_type) {
  auto xt = static_cast<mshadow::TypeFlag>(x_type);
  return xt == mshadow::kFloat16 ? mshadow::kFloat32 : xt;
}

template <typename DType>
__global__ void FillKernel(DType* data, int size, DType value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i] = value;
  }
}

template <typename DType>
__global__ void SetInferenceStatsKernel(DType* save_mean,
                                        DType* save_invstd,
                                        const DType* moving_mean,
                                        const DType* moving_var,
                                        double eps,
                                        int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    save_mean[i] = moving_mean[i];
    save_invstd[i] = DType(1.0) / sqrt(moving_var[i] + DType(eps));
  }
}

inline size_t AlignWorkspaceSize(size_t size) {
  constexpr size_t alignment = 256;
  return ((size + alignment - 1) / alignment) * alignment;
}

}  // namespace

bool CudnnBatchNormSupports(const BatchNormParam& param, const TBlob& x) {
  int n = x.shape_.ndim();
  return n >= 3 && (param.axis == 1 || param.axis == n - 1);
}

void CudnnBatchNormForward(const BatchNormParam& param,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 5);
  if (ctx.is_train) {
    CHECK_EQ(outputs.size(), 3);
    CHECK_EQ(req.size(), 3);
  } else {
    CHECK_GE(outputs.size(), 1);
    CHECK_GE(req.size(), 1);
  }
  CHECK_EQ(req[batchnorm::kOut], kWriteTo);
  CHECK_GE(inputs[batchnorm::kData].ndim(), 2);

  SetDescriptors(param, inputs[batchnorm::kData]);

  auto s = ctx.get_stream<gpu>();
  MSHADOW_REAL_TYPE_SWITCH(ParamType(inputs[batchnorm::kData].type_flag_), DType, {
    DType a = 1.0f;
    DType b = 0.0f;
    if (ctx.is_train) {
      size_t workspace_size = 0;
      CUDNN_CALL(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
          s->dnn_handle_,
          CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
          CUDNN_BATCHNORM_OPS_BN,
          Globals::Get().io_desc,
          nullptr,
          Globals::Get().io_desc,
          Globals::Get().mean_desc,
          nullptr,
          &workspace_size));
      const int channel_count = inputs[batchnorm::kGamma].shape_.Size();
      const size_t aligned_workspace_size = AlignWorkspaceSize(workspace_size);
      const size_t gamma_bytes = param.fix_gamma ? channel_count * sizeof(DType) : 0;
      char* workspace_base = static_cast<char*>(ctx.requested[0].get_space_internal(
          aligned_workspace_size + gamma_bytes, "CudnnBatchNormForward"));
      void* workspace = workspace_base;
      DType* gamma = inputs[batchnorm::kGamma].dptr<DType>();
      if (param.fix_gamma) {
        gamma = reinterpret_cast<DType*>(workspace_base + aligned_workspace_size);
        constexpr int threads = 256;
        const int blocks      = (channel_count + threads - 1) / threads;
        FillKernel<DType><<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
            gamma, channel_count, DType(1.0f));
      }

      // If the lock on the auxiliary states is set, then this implies that
      // the preceding call is also a `Forward()` call, which further
      // indicates that we are in the backward mirroring mode, and therefore
      // update to the auxiliary states is disabled. This is done by setting
      // the `momentum` to `1` (or `factor` to `0`).
      double factor =
          ((dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0) || dmlc::GetEnv("MXNET_MEMORY_OPT", 0)) &&
           Globals::Get().internal_aux_states_lock) ?
              0 :
              (1 - param.momentum);
      CUDNN_CALL(
          cudnnBatchNormalizationForwardTrainingEx(s->dnn_handle_,
                                                   CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
                                                   CUDNN_BATCHNORM_OPS_BN,
                                                   &a,
                                                   &b,
                                                   Globals::Get().io_desc,
                                                   inputs[batchnorm::kData].dptr_,
                                                   nullptr,
                                                   nullptr,  // zDesc, zData
                                                   Globals::Get().io_desc,
                                                   outputs[batchnorm::kOut].dptr_,
                                                   Globals::Get().mean_desc,
                                                   gamma,
                                                   inputs[batchnorm::kBeta].dptr_,
                                                   factor,
                                                   inputs[batchnorm::kInMovingMean].dptr_,
                                                   inputs[batchnorm::kInMovingVar].dptr_,
                                                   param.eps,
                                                   outputs[batchnorm::kMean].dptr_,
                                                   outputs[batchnorm::kVar].dptr_,
                                                   nullptr,  // activation desc
                                                   workspace,
                                                   workspace_size,
                                                   nullptr,
                                                   0));  // reserveSpace, reserveSpaceSizeInBytes
    } else {
      const int channel_count = inputs[batchnorm::kGamma].shape_.Size();
      DType* gamma = inputs[batchnorm::kGamma].dptr<DType>();
      if (param.fix_gamma) {
        gamma = static_cast<DType*>(ctx.requested[0].get_space_internal(
            channel_count * sizeof(DType), "CudnnBatchNormForward"));
        constexpr int threads = 256;
        const int blocks      = (channel_count + threads - 1) / threads;
        FillKernel<DType><<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
            gamma, channel_count, DType(1.0f));
      }
      CUDNN_CALL(cudnnBatchNormalizationForwardInference(s->dnn_handle_,
                                                         CUDNN_BATCHNORM_SPATIAL,
                                                         &a,
                                                         &b,
                                                         Globals::Get().io_desc,
                                                         inputs[batchnorm::kData].dptr_,
                                                         Globals::Get().io_desc,
                                                         outputs[batchnorm::kOut].dptr_,
                                                         Globals::Get().mean_desc,
                                                         gamma,
                                                         inputs[batchnorm::kBeta].dptr_,
                                                         inputs[batchnorm::kInMovingMean].dptr_,
                                                         inputs[batchnorm::kInMovingVar].dptr_,
                                                         param.eps));
      if (outputs.size() > batchnorm::kVar) {
        constexpr int threads = 256;
        const int blocks      = (channel_count + threads - 1) / threads;
        SetInferenceStatsKernel<DType><<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
            outputs[batchnorm::kMean].dptr<DType>(),
            outputs[batchnorm::kVar].dptr<DType>(),
            inputs[batchnorm::kInMovingMean].dptr<DType>(),
            inputs[batchnorm::kInMovingVar].dptr<DType>(),
            param.eps,
            channel_count);
      }
    }
  })
  // Set the lock on the auxiliary states.
  // If the next call to the operator is a `Forward()` call,
  // then `momentum` will be set to `1` and hence auxiliary states will not be updated.
  Globals::Get().internal_aux_states_lock = true;
}

void CudnnBatchNormBackward(const BatchNormParam& param,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 8);
  CHECK_EQ(outputs.size(), 3);
  CHECK_EQ(req.size(), 3);

  SetDescriptors(param, inputs[3 + batchnorm::kData]);
  auto s                = ctx.get_stream<gpu>();
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetBatchNormalizationBackwardExWorkspaceSize(s->dnn_handle_,
                                                               CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
                                                               CUDNN_BATCHNORM_OPS_BN,
                                                               Globals::Get().io_desc,
                                                               Globals::Get().io_desc,
                                                               Globals::Get().io_desc,
                                                               nullptr,
                                                               Globals::Get().io_desc,
                                                               Globals::Get().mean_desc,
                                                               nullptr,
                                                               &workspace_size));
  MSHADOW_REAL_TYPE_SWITCH(ParamType(inputs[3 + batchnorm::kData].type_flag_), DType, {
    const int channel_count = inputs[3 + batchnorm::kGamma].shape_.Size();
    const bool use_temp_gamma_grad =
        param.fix_gamma || req[batchnorm::kGamma] == kNullOp;
    const size_t aligned_workspace_size = AlignWorkspaceSize(workspace_size);
    const size_t fixed_gamma_bytes = param.fix_gamma ? channel_count * sizeof(DType) : 0;
    const size_t gamma_grad_bytes = use_temp_gamma_grad ? channel_count * sizeof(DType) : 0;
    char* workspace_base = static_cast<char*>(ctx.requested[0].get_space_internal(
        aligned_workspace_size + fixed_gamma_bytes + gamma_grad_bytes, "CudnnBatchNormBackward"));
    void* workspace = workspace_base;
    DType* gamma = inputs[3 + batchnorm::kGamma].dptr<DType>();
    if (param.fix_gamma) {
      gamma = reinterpret_cast<DType*>(workspace_base + aligned_workspace_size);
      constexpr int threads = 256;
      const int blocks      = (channel_count + threads - 1) / threads;
      FillKernel<DType><<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
          gamma, channel_count, DType(1.0f));
    }
    DType* gamma_grad = use_temp_gamma_grad ?
                            reinterpret_cast<DType*>(
                                workspace_base + aligned_workspace_size + fixed_gamma_bytes) :
                            outputs[batchnorm::kGamma].dptr<DType>();
    bool grad_add_gamma_beta = req[batchnorm::kGamma] == kAddTo || req[batchnorm::kBeta] == kAddTo;
    if (use_temp_gamma_grad && grad_add_gamma_beta) {
      constexpr int threads = 256;
      const int blocks      = (channel_count + threads - 1) / threads;
      FillKernel<DType><<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
          gamma_grad, channel_count, DType(0.0f));
    }
    if (grad_add_gamma_beta) {
      if (IsBNWriting(req[batchnorm::kGamma]))
        outputs[batchnorm::kGamma].FlatTo1D<gpu, DType>(s) = 0.0f;
      if (IsBNWriting(req[batchnorm::kBeta]))
        outputs[batchnorm::kBeta].FlatTo1D<gpu, DType>(s) = 0.0f;
    }
    DType a                 = 1.0f;
    DType b                 = 0.0f;
    DType b_add             = 1.0f;
    const bool global_stats = !ctx.is_train || param.use_global_stats;
    CUDNN_CALL(
        cudnnBatchNormalizationBackwardEx(s->dnn_handle_,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
                                          CUDNN_BATCHNORM_OPS_BN,
                                          &a,
                                          req[batchnorm::kData] == kAddTo ? &b_add : &b,
                                          &a,
                                          grad_add_gamma_beta ? &b_add : &b,
                                          Globals::Get().io_desc,
                                          inputs[3 + batchnorm::kData].dptr_,
                                          nullptr,
                                          nullptr,  // yDesc, yData
                                          Globals::Get().io_desc,
                                          inputs[batchnorm::kOut].dptr_,
                                          nullptr,
                                          nullptr,  // dzDesc, dzData
                                          Globals::Get().io_desc,
                                          outputs[batchnorm::kData].dptr_,
                                          Globals::Get().mean_desc,
                                          gamma,
                                          inputs[3 + batchnorm::kBeta].dptr_,
                                          gamma_grad,
                                          outputs[batchnorm::kBeta].dptr_,
                                          param.eps,
                                          global_stats ? nullptr : inputs[batchnorm::kMean].dptr_,
                                          global_stats ? nullptr : inputs[batchnorm::kVar].dptr_,
                                          nullptr,  // activationDesc
                                          workspace,
                                          workspace_size,
                                          nullptr,
                                          0));  // reserveSpace, reserveSpaceSizeInBytes
    if (param.fix_gamma && IsBNWriting(req[batchnorm::kGamma]))
      outputs[batchnorm::kGamma].FlatTo1D<gpu, DType>(s) = 0.0f;
  })
  Globals::Get().internal_aux_states_lock = false;
}

#endif  // MXNET_USE_CUDNN == 1
}  // namespace op
}  // namespace mxnet
