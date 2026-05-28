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
 * \file dnnl_quantized_pooling.cc
 * \brief
 * \author Tao Lv, Xinyu Chen
 */

#if MXNET_USE_ONEDNN == 1

#include "operator/nn/dnnl/dnnl_pooling-inl.h"
#include "operator/quantization/quantized_range_utils.h"

namespace mxnet {
namespace op {

static void DNNLQuantizedPoolingForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& in_data,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& out_data) {
  CHECK(in_data[0].dtype() == mshadow::kUint8 || in_data[0].dtype() == mshadow::kInt8)
      << "dnnl_quantized_pooling op only supports uint8 and int8 as input type";
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  if (ctx.is_train && DNNLRequireWorkspace(param) && !param.IsAdaptivePooling()) {
    NDArray workspace(out_data[0].shape(), out_data[0].ctx(), false, mshadow::kInt32);
    std::vector<NDArray> dnnl_outputs{out_data[0], workspace};
    std::vector<OpReqType> dnnl_req{req[0], kWriteTo};
    DNNLRun(DNNLPoolingCompute, attrs, ctx, in_data, dnnl_req, dnnl_outputs);
  } else {
    DNNLRun(DNNLPoolingCompute, attrs, ctx, in_data, req, out_data);
  }
  AssignQuantizedRangeOutput(out_data[1].data().dptr<float>(),
                             in_data[1].data().dptr<float>(),
                             req[1],
                             "quantized_pooling");
  AssignQuantizedRangeOutput(out_data[2].data().dptr<float>(),
                             in_data[2].data().dptr<float>(),
                             req[2],
                             "quantized_pooling");
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedPoolingForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
