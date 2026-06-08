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
 * \file np_percentile_op.cu
 * \brief GPU Implementation of Numpy-compatible percentile
 */

#include "np_percentile_op-inl.h"

namespace mxnet {
namespace op {

struct is_valid_check {
  template <typename QType>
  MSHADOW_XINLINE static void Map(int i, char* invalid_ptr, const QType* data) {
    if (data[i] < 0.0 || data[i] > 100)
      *invalid_ptr = 1;
  }
};

template <typename QType, typename gpu>
bool CheckInvalidInput(mshadow::Stream<gpu>* s,
                       const QType* data,
                       const size_t& data_size,
                       char* is_valid_ptr) {
  using namespace mxnet_op;
  int32_t is_valid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
  Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data);
  CUDA_CALL(cudaMemcpyAsync(&is_valid,
                            is_valid_ptr,
                            sizeof(char),
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  return is_valid == 0;
}

// Percentile backward is a host algorithm (per-group std::sort + scatter) with
// no GPU kernel. Run it on the CPU and shuttle the (small) tensors across,
// instead of leaving _backward_npi_percentile unimplemented for GPU (which
// aborts with "not implemented for GPU"). inputs: [ograd, data, (q)];
// outputs: [data_grad, (q_grad)]. The CPU impl does not use the stream/device.
template <>
void NumpyPercentileBackward<gpu>(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<gpu>* s      = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  auto byte_size = [](const TBlob& b) -> size_t {
    size_t elem = 0;
    MSHADOW_TYPE_SWITCH_WITH_BOOL(b.type_flag_, DType, { elem = sizeof(DType); });
    return b.shape_.Size() * elem;
  };
  // Copy into a local: emplace_back perfect-forwards by reference, which would
  // odr-use mshadow::cpu::kDevMask (a static const int with no out-of-class
  // definition) and leave an undefined symbol. A by-value ctor arg constant-
  // folds, but the forwarded reference does not.
  const int cpu_mask = mshadow::cpu::kDevMask;

  std::vector<std::vector<char>> in_store(inputs.size());
  std::vector<TBlob> h_inputs;
  h_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    in_store[i].resize(byte_size(inputs[i]));
    if (!in_store[i].empty()) {
      CUDA_CALL(cudaMemcpyAsync(in_store[i].data(), inputs[i].dptr_, in_store[i].size(),
                                cudaMemcpyDeviceToHost, stream));
    }
    h_inputs.emplace_back(in_store[i].data(), inputs[i].shape_, cpu_mask,
                          inputs[i].type_flag_, 0);
  }

  std::vector<std::vector<char>> out_store(outputs.size());
  std::vector<TBlob> h_outputs;
  h_outputs.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    out_store[i].resize(byte_size(outputs[i]));
    // kAddTo accumulates onto the existing gradient, so seed the host buffer.
    if (req[i] == kAddTo && !out_store[i].empty()) {
      CUDA_CALL(cudaMemcpyAsync(out_store[i].data(), outputs[i].dptr_, out_store[i].size(),
                                cudaMemcpyDeviceToHost, stream));
    }
    h_outputs.emplace_back(out_store[i].data(), outputs[i].shape_, cpu_mask,
                           outputs[i].type_flag_, 0);
  }

  CUDA_CALL(cudaStreamSynchronize(stream));  // inputs (and kAddTo seeds) now on host
  NumpyPercentileBackward<cpu>(attrs, ctx, h_inputs, req, h_outputs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (req[i] != kNullOp && !out_store[i].empty()) {
      CUDA_CALL(cudaMemcpyAsync(outputs[i].dptr_, out_store[i].data(), out_store[i].size(),
                                cudaMemcpyHostToDevice, stream));
    }
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}

NNVM_REGISTER_OP(_npi_percentile)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", NumpyPercentileForward<gpu>);

NNVM_REGISTER_OP(_backward_npi_percentile)
    .set_attr<FCompute>("FCompute<gpu>", NumpyPercentileBackward<gpu>);

}  // namespace op
}  // namespace mxnet
