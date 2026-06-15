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
 * \file np_dot_forward.cu
 * \brief GPU Implementation of numpy-compatible dot
 */

#include "./np_dot-inl.h"
#include "../../common/cuda/cublaslt_gemm.h"

namespace mxnet {
namespace op {

// _npi_dot dispatches through the tensordot machinery (MatrixDot), which now routes
// its GPU gemm through the capture-safe cuBLASLt linalg_gemm path under CUDA Graphs
// (OI-16). Capturable by default (Phase 5); set MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=0 to
// opt out.
namespace {
inline bool DotGraphsCompatible() {
  return mxnet::common::cuda::AllowGemmCapture();
}
}  // namespace

NNVM_REGISTER_OP(_npi_dot)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) { return DotGraphsCompatible(); })
    .set_attr<FCompute>("FCompute<gpu>", NumpyDotForward<gpu>);

}  // namespace op
}  // namespace mxnet
