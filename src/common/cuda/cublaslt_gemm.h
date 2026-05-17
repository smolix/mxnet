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
 * \file cublaslt_gemm.h
 * \brief Optional cuBLASLt-backed GEMM wrappers (PR-A: fp32 only).
 *
 * The MXNET_USE_CUBLASLT environment variable (default off) selects the
 * cuBLASLt heuristic path. On any error or zero-heuristic result the wrapper
 * returns a non-success cublasStatus_t and the caller MUST fall back to the
 * legacy cuBLAS API. See cublaslt_scope.md for the broader plan.
 */
#ifndef MXNET_COMMON_CUDA_CUBLASLT_GEMM_H_
#define MXNET_COMMON_CUDA_CUBLASLT_GEMM_H_

#if MXNET_USE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mxnet {
namespace common {
namespace cuda {

/*! \brief Returns true when MXNET_USE_CUBLASLT=1 in the environment.
 *  Default is false. Cached after first call. */
bool UseCuBlasLt();

/*!
 * \brief Attempt to run a single-precision GEMM via cuBLASLt.
 *
 * Arguments mirror legacy cublasSgemm (column-major). The wrapper builds
 * matmul descriptors on the fly, queries the heuristic cache (filling it on
 * first miss for a given key), and dispatches cublasLtMatmul on the same
 * cuda stream as the supplied legacy cublasHandle_t.
 *
 * Returns CUBLAS_STATUS_SUCCESS on success. On any failure (including zero
 * heuristic results), returns the relevant cublasStatus_t -- callers must
 * fall back to legacy cuBLAS in that case. Does NOT modify the output matrix
 * on failure.
 */
cublasStatus_t MaybeCublasLtSgemm(cublasHandle_t legacy_handle,
                                  cublasOperation_t opA,
                                  cublasOperation_t opB,
                                  int m,
                                  int n,
                                  int k,
                                  const float* alpha,
                                  const float* A,
                                  int lda,
                                  const float* B,
                                  int ldb,
                                  const float* beta,
                                  float* C,
                                  int ldc);

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_COMMON_CUDA_CUBLASLT_GEMM_H_
