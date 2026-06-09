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
 * under the License.ø
 */

/*!
 * \file np_matmul_op.cu
 * \brief GPU Implementation of numpy-compatible matmul
 */

#include "np_matmul_op-inl.h"
namespace mxnet {
namespace op {

// matmul routes its GPU gemm through linalg_batch_gemm (cuBLASLt, capture-safe;
// fp32 honors MXNET_CUDA_ALLOW_TENSOR_CORE). Gate capture on
// MXNET_CUDA_GRAPHS_ALLOW_CUBLAS (Phase-2 opt-in). Backward reuses MatmulImpl,
// so it is capture-safe too.
namespace {
inline bool MatmulGraphsCompatible() {
  static const bool allow = dmlc::GetEnv("MXNET_CUDA_GRAPHS_ALLOW_CUBLAS", false);
  return allow;
}
}  // namespace

NNVM_REGISTER_OP(_npi_matmul)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) { return MatmulGraphsCompatible(); })
    .set_attr<FCompute>("FCompute<gpu>", NumpyMatmulForward<gpu>);

NNVM_REGISTER_OP(_backward_np_matmul)
    .set_attr<FIsCUDAGraphsCompatible>(
        "FIsCUDAGraphsCompatible",
        [](const NodeAttrs&, const bool) { return MatmulGraphsCompatible(); })
    .set_attr<FCompute>("FCompute<gpu>", NumpyMatmulBackward<gpu>);

}  // namespace op
}  // namespace mxnet
