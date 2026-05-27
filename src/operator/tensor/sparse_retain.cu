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
 * \file sparse_retain.cu
 * \brief GPU implementation of sparse_retain operator
 */

#include "./sparse_retain-inl.h"
#include "../../common/cuda/utils.h"

namespace mxnet {
namespace op {

template <typename IType>
void SparseRetainValidateIndices(mshadow::Stream<gpu>* s,
                                 const TBlob& idx_data,
                                 const index_t num_rows) {
  std::vector<IType> idx(idx_data.Size());
  if (idx.empty()) {
    return;
  }
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  CUDA_CALL(cudaMemcpyAsync(idx.data(),
                            idx_data.dptr<IType>(),
                            idx.size() * sizeof(IType),
                            cudaMemcpyDeviceToHost,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  for (index_t i = 0; i < static_cast<index_t>(idx.size()); ++i) {
    const index_t row = static_cast<index_t>(idx[i]);
    CHECK_GE(row, 0) << "sparse_retain index " << row << " at position " << i
                     << " is out of bounds for axis 0 with size " << num_rows;
    CHECK_LT(row, num_rows) << "sparse_retain index " << row << " at position " << i
                            << " is out of bounds for axis 0 with size " << num_rows;
  }
}

NNVM_REGISTER_OP(_sparse_retain)
    .set_attr<FComputeEx>("FComputeEx<gpu>", SparseRetainOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_sparse_retain)
    .set_attr<FComputeEx>("FComputeEx<gpu>", SparseRetainOpBackwardEx<gpu>);

}  // namespace op
}  // namespace mxnet
