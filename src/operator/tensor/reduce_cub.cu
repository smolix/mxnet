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
 * \file reduce_cub.cu
 * \brief Fast global (scalar-output) sum/mean reduction on GPU via CUB.
 *
 * The generic RTC reduction is bandwidth-inefficient for the all-reduce case
 * (scalar output): on an RTX 4090 it sustains ~190 GB/s (~19% of peak) because
 * the launch collapses to a single grid column and the loads are not
 * vectorized. cub::DeviceReduce is a well-tuned device-wide reduction (~peak
 * bandwidth). This routes the global sum/mean case to it, accumulating in
 * double to preserve the precision of the existing safe-accumulation path.
 */

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

namespace {

template <typename DType>
struct CastToDouble {
  __host__ __device__ double operator()(const DType& x) const {
    return static_cast<double>(static_cast<float>(x));
  }
};

// double specialization: avoid the float round-trip.
template <>
struct CastToDouble<double> {
  __host__ __device__ double operator()(const double& x) const {
    return x;
  }
};

template <typename DType>
__global__ void FinalizeGlobalReduceKernel(const double* total_sum,
                                           DType* out,
                                           const double count,
                                           const bool mean,
                                           const bool addto) {
  double v = mean ? (*total_sum) / count : (*total_sum);
  if (addto) {
    out[0] = static_cast<DType>(static_cast<double>(out[0]) + v);
  } else {
    out[0] = static_cast<DType>(v);
  }
}

}  // namespace

template <typename DType>
void CubGlobalSumReduce(const OpContext& ctx,
                        const TBlob& in,
                        const TBlob& out,
                        const bool mean,
                        const double count,
                        const bool addto) {
  using namespace mshadow;
  Stream<gpu>* s        = ctx.get_stream<gpu>();
  cudaStream_t stream   = Stream<gpu>::GetStream(s);
  const DType* in_ptr   = in.dptr<DType>();
  const index_t total   = static_cast<index_t>(in.Size());
  CastToDouble<DType> conv;
  auto it = thrust::make_transform_iterator(in_ptr, conv);

  // Query temp storage size, then carve one workspace holding temp + a double
  // result, both from the op's kTempSpace resource.
  size_t temp_bytes = 0;
  cub::DeviceReduce::Sum(nullptr, temp_bytes, it, static_cast<double*>(nullptr), total, stream);
  const size_t result_off = ((temp_bytes + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  Tensor<gpu, 1, char> ws =
      ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(result_off + sizeof(double)), s);
  void* d_temp     = ws.dptr_;
  double* d_result = reinterpret_cast<double*>(ws.dptr_ + result_off);

  cub::DeviceReduce::Sum(d_temp, temp_bytes, it, d_result, total, stream);
  FinalizeGlobalReduceKernel<DType>
      <<<1, 1, 0, stream>>>(d_result, out.dptr<DType>(), count, mean, addto);
  MSHADOW_CUDA_POST_KERNEL_CHECK(FinalizeGlobalReduceKernel);
}

template void CubGlobalSumReduce<float>(
    const OpContext&, const TBlob&, const TBlob&, const bool, const double, const bool);
template void CubGlobalSumReduce<double>(
    const OpContext&, const TBlob&, const TBlob&, const bool, const double, const bool);
template void CubGlobalSumReduce<mshadow::half::half_t>(
    const OpContext&, const TBlob&, const TBlob&, const bool, const double, const bool);

}  // namespace op
}  // namespace mxnet
