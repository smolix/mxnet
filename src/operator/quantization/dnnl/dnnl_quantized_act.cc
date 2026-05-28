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
 * \file dnnl_quantized_act.cc
 * \brief DNNL(Quantized) Activation operator based on subgraph
 * /author Zhiyuan Huang
 */
#if MXNET_USE_ONEDNN == 1

#include "operator/nn/activation-inl.h"
#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/mxnet_op.h"
#include "operator/quantization/quantized_range_utils.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

struct QuantizedReluUInt8AffineKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int8_t* out,
                                  const uint8_t* in,
                                  float min_data,
                                  float max_data,
                                  float out_max,
                                  OpReqType req) {
    const float real = static_cast<float>(in[i]) * (max_data - min_data) / kUint8Range + min_data;
    const float relu = real > 0.0f ? real : 0.0f;
    const float scaled = out_max > 0.0f ? floorf(relu * kInt8Range / out_max + 0.5f) : 0.0f;
    const int32_t q = static_cast<int32_t>(scaled > kInt8Range ? kInt8Range : scaled);
    KERNEL_ASSIGN(out[i], req, static_cast<int8_t>(q));
  }
};

static void DNNLQuantizedActForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& in_data,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& out_data) {
  CHECK(in_data[0].dtype() == mshadow::kUint8 || in_data[0].dtype() == mshadow::kInt8)
      << "_contrib_quantized_act op only supports uint8 and int8 as input "
         "type";

  const float input_min = in_data[1].data().dptr<float>()[0];
  const float input_max = in_data[2].data().dptr<float>()[0];
  if (in_data[0].dtype() == mshadow::kUint8 && input_min < 0.0f) {
    const float output_min = 0.0f;
    const float output_max = std::max(0.0f, input_max);
    if (req[0] != kNullOp) {
      NDArray input = in_data[0].Reorder2Default();
      mxnet_op::Kernel<QuantizedReluUInt8AffineKernel, cpu>::Launch(
          ctx.get_stream<cpu>(),
          input.shape().Size(),
          out_data[0].data().dptr<int8_t>(),
          input.data().dptr<uint8_t>(),
          input_min,
          input_max,
          output_max,
          req[0]);
    }
    AssignQuantizedRangeOutput(
        out_data[1].data().dptr<float>(), &output_min, req[1], "quantized_act");
    AssignQuantizedRangeOutput(
        out_data[2].data().dptr<float>(), &output_max, req[2], "quantized_act");
    return;
  }

  DNNLRun(DNNLActivationForward, attrs, ctx, in_data[0], req[0], out_data[0]);
  AssignQuantizedRangeOutput(out_data[1].data().dptr<float>(),
                             in_data[1].data().dptr<float>(),
                             req[1],
                             "quantized_act");
  AssignQuantizedRangeOutput(out_data[2].data().dptr<float>(),
                             in_data[2].data().dptr<float>(),
                             req[2],
                             "quantized_act");
}

NNVM_REGISTER_OP(_contrib_quantized_act)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedActForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
