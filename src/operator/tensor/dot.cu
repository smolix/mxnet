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
 * \file dot.cu
 * \brief GPU Implementation of matrix dot
 */

#include "./dot-inl.h"
#include "../../common/cuda/cublaslt_gemm.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(dot)
    .set_attr<FCompute>("FCompute<gpu>", DotForward_<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", DotForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_dot)
    .set_attr<FCompute>("FCompute<gpu>", DotBackward_<gpu>)
    .set_attr<FComputeEx>("FComputeEx<gpu>", DotBackwardEx<gpu>);

// batch_dot now routes its GPU gemm through linalg_batch_gemm, which is
// CUDA-graph capture-safe via cuBLASLt (see linalg_impl.h / fully_connected.cu).
// Capturable by default (Phase 5); the cuBLASLt path is auto-forced on under
// capture so warm-up and captured runs match. Set MXNET_CUDA_GRAPHS_ALLOW_CUBLAS=0
// to opt out. dot / _backward_dot remain excluded via their FComputeEx dispatch.
NNVM_REGISTER_OP(batch_dot)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) { return mxnet::common::cuda::AllowGemmCapture(); })
    .set_attr<FCompute>("FCompute<gpu>", BatchDotForward_<gpu>);

}  // namespace op
}  // namespace mxnet
