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
 * \file quantized_rnn.cu
 * \brief GPU fallback for quantized RNN.
 */
#include <mxnet/storage.h>
#include "operator/rnn-inl.h"
#include "operator/quantization/quantized_rnn-inl.h"
#include "operator/mxnet_op.h"

namespace mxnet {
namespace op {

struct quantized_rnn_dequantize_kernel {
  MSHADOW_XINLINE static void Map(int i,
                                  float* out,
                                  const uint8_t* in,
                                  const float* data_scale,
                                  const float* data_shift) {
    out[i] = (static_cast<float>(in[i]) - data_shift[0]) / data_scale[0];
  }
};

class QuantizedRnnGPUOp {
 public:
  explicit QuantizedRnnGPUOp(const RNNParam& param, Context ctx)
      : rnn_op_(param, ctx), dequantized_size_(0) {
    dequantized_space_.dptr = nullptr;
    dequantized_space_.size = 0;
  }

  ~QuantizedRnnGPUOp() {
    if (dequantized_space_.dptr != nullptr) {
      Storage::Get()->Free(dequantized_space_);
      dequantized_space_.dptr = nullptr;
    }
  }

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    CHECK_EQ(in_data.size(), 6U);
    CHECK_EQ(in_data[quantized_rnn::kData].type_flag_, mshadow::kUint8);
    CHECK_EQ(in_data[quantized_rnn::kParams].type_flag_, mshadow::kFloat32);
    CHECK_EQ(in_data[quantized_rnn::kState].type_flag_, mshadow::kFloat32);
    CHECK_EQ(in_data[quantized_rnn::kStateCell].type_flag_, mshadow::kFloat32);
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();

    const size_t data_size = in_data[quantized_rnn::kData].Size();
    EnsureDequantizedSpace(data_size * sizeof(float), s->dev_id);
    mxnet_op::Kernel<quantized_rnn_dequantize_kernel, gpu>::Launch(
        s,
        data_size,
        static_cast<float*>(dequantized_space_.dptr),
        in_data[quantized_rnn::kData].dptr<uint8_t>(),
        in_data[GetRnnNumInputs(rnn_op_.param_) + quantized_rnn::kDataScale].dptr<float>(),
        in_data[GetRnnNumInputs(rnn_op_.param_) + quantized_rnn::kDataShift].dptr<float>());

    std::vector<TBlob> rnn_inputs{
        TBlob(dequantized_space_.dptr,
              in_data[quantized_rnn::kData].shape_,
              gpu::kDevMask,
              mshadow::kFloat32,
              ctx.run_ctx.ctx.dev_id),
        in_data[quantized_rnn::kParams],
        in_data[quantized_rnn::kState],
        in_data[quantized_rnn::kStateCell]};
    rnn_op_.Forward(ctx, rnn_inputs, req, out_data);
  }

 private:
  void EnsureDequantizedSpace(size_t bytes, int dev_id) {
    if (dequantized_space_.dptr == nullptr || dequantized_size_ < bytes) {
      if (dequantized_space_.dptr != nullptr) {
        Storage::Get()->Free(dequantized_space_);
      }
      dequantized_space_ = Storage::Get()->Alloc(bytes, Context::GPU(dev_id));
      dequantized_space_.profiler_scope = "quantized_rnn:";
      dequantized_space_.name           = "dequantized_data";
      dequantized_size_                 = bytes;
    }
  }

  RNNOp<gpu, float, float> rnn_op_;
  Storage::Handle dequantized_space_;
  size_t dequantized_size_;
};

OpStatePtr CreateQuantizedRnnGPUState(const nnvm::NodeAttrs& attrs,
                                      const Context ctx,
                                      const mxnet::ShapeVector& in_shapes,
                                      const std::vector<int>& in_types) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  CHECK_EQ(param.mode, rnn_enum::kLstm) << "Quantized RNN operator only supports LSTM mode.";
  CHECK_EQ(in_types[quantized_rnn::kData], mshadow::kUint8);
  CHECK_EQ(in_types[quantized_rnn::kParams], mshadow::kFloat32);
  CHECK_EQ(in_types[quantized_rnn::kState], mshadow::kFloat32);
  CHECK_EQ(in_types[quantized_rnn::kStateCell], mshadow::kFloat32);
  return OpStatePtr::Create<QuantizedRnnGPUOp>(param, ctx);
}

void QuantizedRnnForwardGPU(const OpStatePtr& state_ptr,
                            const OpContext& ctx,
                            const std::vector<TBlob>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& out_data) {
  QuantizedRnnGPUOp& op = state_ptr.get_state<QuantizedRnnGPUOp>();
  op.Forward(ctx, in_data, req, out_data);
}

NNVM_REGISTER_OP(_contrib_quantized_rnn)
    .set_attr<FStatefulCompute>("FStatefulCompute<gpu>", QuantizedRnnForwardGPU);

}  // namespace op
}  // namespace mxnet
