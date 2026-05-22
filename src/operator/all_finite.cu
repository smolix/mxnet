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
 * \file all_finite.cu
 * \brief operator for checking if a group of array is all finite
 * \author Clement Fuji Tsang
 */

#include "./all_finite-inl.h"
#include <limits>

namespace mxnet {
namespace op {

template <typename DType>
__global__ void AllFiniteGPUKernel(const int size, const DType* in, float* out) {
  bool is_finite = true;
  CUDA_KERNEL_LOOP(i, size) {
    is_finite = isfinite(static_cast<float>(in[i])) ? is_finite : false;
  }
  __syncthreads();
  if (!is_finite) {
    out[0] = 0.;
  }
}

inline void AllFiniteGPU(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<gpu>* s                 = ctx.get_stream<gpu>();
  const AllFiniteParam& op_param = nnvm::get<AllFiniteParam>(attrs.parsed);
  Tensor<gpu, 2, float> out      = outputs[0].FlatTo2D<gpu, float>(s);
  if (op_param.init_output)
    out = 1.;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<gpu, 2, DType> in = inputs[0].FlatTo2D<gpu, DType>(s);
    CHECK_LE(in.shape_.Size(), static_cast<size_t>(std::numeric_limits<int>::max()))
        << "all_finite supports at most INT_MAX elements per input.";
    const int n = static_cast<int>(in.shape_.Size());
    AllFiniteGPUKernel<DType><<<cuda_get_num_blocks(n),
                                mshadow::cuda_impl::kBaseThreadNum,
                                0,
                                mshadow::Stream<gpu>::GetStream(s)>>>(n, in.dptr_, out.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(AllFiniteGPUKernel<DType>);
  });
}

template <typename DType>
__global__ void MultiAllFiniteGPUKernel(const MultiAllFiniteKernelParam<DType> param, float* out) {
  bool is_finite = true;
  for (int index = 0; index < param.count; ++index) {
    CUDA_KERNEL_LOOP(i, param.sizes[index]) {
      is_finite = isfinite(static_cast<float>(param.arrays[index][i])) ? is_finite : false;
    }
  }
  __syncthreads();
  if (!is_finite) {
    out[0] = 0.;
  }
}

inline void MultiAllFiniteGPU(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<gpu>* s                      = ctx.get_stream<gpu>();
  const MultiAllFiniteParam& op_param = nnvm::get<MultiAllFiniteParam>(attrs.parsed);
  Tensor<gpu, 2, float> out           = outputs[0].FlatTo2D<gpu, float>(s);
  if (op_param.init_output)
    out = 1.;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MultiAllFiniteKernelParam<DType> param =
        FillMultiAllFiniteParam<gpu, DType>(op_param, ctx, inputs);
    CHECK_LE(param.max_size, static_cast<size_t>(std::numeric_limits<int>::max()))
        << "multi_all_finite supports at most INT_MAX elements per input.";
    const int max_size = static_cast<int>(param.max_size);
    MultiAllFiniteGPUKernel<DType><<<cuda_get_num_blocks(max_size),
                                     mshadow::cuda_impl::kBaseThreadNum,
                                     1,
                                     mshadow::Stream<gpu>::GetStream(s)>>>(param, out.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MultiAllFiniteGPUKernel<DType>);
  });
}

NNVM_REGISTER_OP(all_finite).set_attr<FCompute>("FCompute<gpu>", AllFiniteGPU);

NNVM_REGISTER_OP(multi_all_finite).set_attr<FCompute>("FCompute<gpu>", MultiAllFiniteGPU);

}  // namespace op
}  // namespace mxnet
