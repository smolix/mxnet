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
 * \file np_interp_op.cu
 * \brief GPU Implementation of Numpy-compatible interp
 */

#include "np_interp_op-inl.h"

namespace mxnet {
namespace op {

// GPU backward kernel for np.interp. One thread per output element of `x`; it
// scatter-accumulates into fp_grad/xp_grad with atomics (multiple x can map to
// the same bin edge), and writes the per-element x_grad. Mirrors the CPU loop
// in NumpyInterpBackward (np_interp_op-inl.h). All buffers are float64. Callers
// zero kWriteTo outputs first, so accumulation is uniform for kWriteTo/kAddTo.
__global__ void NumpyInterpBackwardGPUKernel(const index_t x_size,
                                             const double* ograd,
                                             const double* xp,
                                             const double* fp,
                                             const double* x,
                                             const bool x_is_scalar,
                                             const double x_scalar,
                                             double* xp_grad,
                                             double* fp_grad,
                                             double* x_grad,
                                             const bool has_left,
                                             const bool has_right,
                                             const index_t dsize) {
  const index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= x_size)
    return;
  const double x_value = x_is_scalar ? x_scalar : x[i];
  const double grad    = ograd[i];
  if (x_value > xp[dsize - 1]) {
    if (!has_right && fp_grad)
      atomicAdd(&fp_grad[dsize - 1], grad);
  } else if (x_value < xp[0]) {
    if (!has_left && fp_grad)
      atomicAdd(&fp_grad[0], grad);
  } else {
    index_t imin = 0;
    index_t imax = dsize;
    while (imin < imax) {
      const index_t imid = (imax + imin) / 2;
      if (x_value >= xp[imid]) {
        imin = imid + 1;
      } else {
        imax = imid;
      }
    }
    const index_t j = imin;
    if (j == dsize) {
      if (fp_grad)
        atomicAdd(&fp_grad[dsize - 1], grad);
    } else if (x_value == xp[j - 1]) {
      if (fp_grad)
        atomicAdd(&fp_grad[j - 1], grad);
    } else {
      const double xp_below     = xp[j - 1];
      const double xp_above     = xp[j];
      const double fp_below     = fp[j - 1];
      const double fp_above     = fp[j];
      const double denom        = xp_above - xp_below;
      const double weight_above = (x_value - xp_below) / denom;
      const double weight_below = 1.0 - weight_above;
      if (fp_grad) {
        atomicAdd(&fp_grad[j - 1], grad * weight_below);
        atomicAdd(&fp_grad[j], grad * weight_above);
      }
      if (x_grad) {
        // x_grad[i] is unique per thread; accumulate (outputs pre-zeroed for
        // kWriteTo) so kWriteTo and kAddTo are handled uniformly.
        atomicAdd(&x_grad[i], grad * (fp_above - fp_below) / denom);
      }
      if (xp_grad) {
        const double scale = grad * (fp_above - fp_below) / (denom * denom);
        atomicAdd(&xp_grad[j - 1], scale * (x_value - xp_above));
        atomicAdd(&xp_grad[j], -scale * (x_value - xp_below));
      }
    }
  }
}

template <>
void NumpyInterpBackward<gpu>(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  CHECK(!param.period.has_value()) << "np.interp backward does not support period";
  CHECK_EQ(inputs.size(), param.x_is_scalar ? 3U : 4U);
  CHECK_EQ(outputs.size(), param.x_is_scalar ? 2U : 3U);

  const TBlob& ograd  = inputs[0];
  const TBlob& xp     = inputs[1];
  const TBlob& fp     = inputs[2];
  const TBlob& x      = param.x_is_scalar ? inputs[0] : inputs[3];
  const index_t x_size = param.x_is_scalar ? 1 : static_cast<index_t>(x.Size());
  CHECK_EQ(ograd.Size(), static_cast<size_t>(x_size));
  CHECK_EQ(xp.Size(), fp.Size());
  CHECK_GE(xp.Size(), 1U) << "ValueError: array of sample points is empty";

  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();

  // Zero the kWriteTo outputs so the atomic scatter below accumulates correctly
  // for both kWriteTo (onto 0) and kAddTo (onto the existing gradient).
  if (req[0] == kWriteTo || req[0] == kWriteInplace)
    Kernel<set_zero, gpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<double>());
  if (req[1] == kWriteTo || req[1] == kWriteInplace)
    Kernel<set_zero, gpu>::Launch(s, outputs[1].Size(), outputs[1].dptr<double>());
  if (!param.x_is_scalar && (req[2] == kWriteTo || req[2] == kWriteInplace))
    Kernel<set_zero, gpu>::Launch(s, outputs[2].Size(), outputs[2].dptr<double>());

  double* xp_grad = req[0] == kNullOp ? nullptr : outputs[0].dptr<double>();
  double* fp_grad = req[1] == kNullOp ? nullptr : outputs[1].dptr<double>();
  double* x_grad =
      param.x_is_scalar || req[2] == kNullOp ? nullptr : outputs[2].dptr<double>();

  if (x_size == 0)
    return;
  const int threads      = 256;
  const int blocks       = static_cast<int>((x_size + threads - 1) / threads);
  cudaStream_t stream    = mshadow::Stream<gpu>::GetStream(s);
  NumpyInterpBackwardGPUKernel<<<blocks, threads, 0, stream>>>(
      x_size,
      ograd.dptr<double>(),
      xp.dptr<double>(),
      fp.dptr<double>(),
      param.x_is_scalar ? nullptr : x.dptr<double>(),
      param.x_is_scalar,
      param.x_scalar,
      xp_grad,
      fp_grad,
      x_grad,
      param.left.has_value(),
      param.right.has_value(),
      static_cast<index_t>(xp.Size()));
  MSHADOW_CUDA_POST_KERNEL_CHECK(NumpyInterpBackwardGPUKernel);
}

NNVM_REGISTER_OP(_npi_interp)
    .set_attr<FCompute>("FCompute<gpu>", NumpyInterpForward<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_npi_interp)
    .set_attr<FCompute>("FCompute<gpu>", NumpyInterpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
