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
 * \file ravel.cu
 * \brief GPU implementation of Operators for ravel/unravel.
 */
#include "./ravel.h"

namespace mxnet {
namespace op {

struct unravel_index_bound_check {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  index_t* invalid,
                                  const DType* indices,
                                  index_t max_index) {
    const int64_t idx = static_cast<int64_t>(indices[i]);
    if (idx < 0 || idx >= max_index) {
      *invalid = 1;
    }
  }
};

template <typename DType>
void CheckUnravelIndexBounds(mshadow::Stream<gpu>* s,
                             const DType* indices,
                             size_t size,
                             index_t max_index,
                             index_t* invalid_flag) {
  using namespace mxnet_op;
  index_t invalid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, invalid_flag);
  Kernel<unravel_index_bound_check, gpu>::Launch(s, size, invalid_flag, indices, max_index);
  CUDA_CALL(cudaMemcpyAsync(&invalid,
                            invalid_flag,
                            sizeof(index_t),
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  CHECK_EQ(invalid, 0) << "IndexError: index is out of bounds for array with size " << max_index;
}

NNVM_REGISTER_OP(_ravel_multi_index).set_attr<FCompute>("FCompute<gpu>", RavelForward<gpu>);

NNVM_REGISTER_OP(_unravel_index).set_attr<FCompute>("FCompute<gpu>", UnravelForward<gpu>);

}  // namespace op
}  // namespace mxnet
