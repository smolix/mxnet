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

namespace mxnet {
namespace op {

// _npi_dot dispatches through the tensordot machinery, whose GPU gemm uses the
// legacy mshadow BLASEngine path (capture-unsafe). It has no FStatefulCompute /
// FComputeEx to exclude it, so mark it explicitly incompatible with CUDA Graphs
// to avoid a capture-time crash; it runs conventionally between graphs.
// (Routing tensordot through linalg/cuBLASLt is possible future work.)
NNVM_REGISTER_OP(_npi_dot)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", NumpyDotForward<gpu>);

}  // namespace op
}  // namespace mxnet
