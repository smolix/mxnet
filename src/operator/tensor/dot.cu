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
// Gate capture on MXNET_CUDA_GRAPHS_ALLOW_CUBLAS (Phase-2 opt-in); the cuBLASLt
// path is auto-forced on when this is set so warm-up and captured runs match.
// dot / _backward_dot remain excluded via their FComputeEx dispatch.
NNVM_REGISTER_OP(batch_dot)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) {
          static const bool allow = dmlc::GetEnv("MXNET_CUDA_GRAPHS_ALLOW_CUBLAS", false);
          return allow;
        })
    .set_attr<FCompute>("FCompute<gpu>", BatchDotForward_<gpu>);

}  // namespace op
}  // namespace mxnet
