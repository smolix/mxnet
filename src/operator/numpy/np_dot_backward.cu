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
 * \file np_dot_backward.cu
 * \brief GPU Implementation of numpy-compatible dot
 */

#include "./np_dot-inl.h"
#include "../../common/cuda/cublaslt_gemm.h"

namespace mxnet {
namespace op {

// See np_dot_forward.cu: the backward gemm also flows through MatrixDot, which is now
// capture-safe (linalg_gemm / cuBLASLt), so the numpy dot backward is capturable too
// (OI-16). MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=0 opts out.
NNVM_REGISTER_OP(_backward_npi_dot)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) { return mxnet::common::cuda::AllowGemmCapture(); })
    .set_attr<FCompute>("FCompute<gpu>", NumpyDotBackward<gpu>);

}  // namespace op
}  // namespace mxnet
