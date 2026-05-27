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
 * \file quantized_pooling.cc
 */
#include <mxnet/op_attr_types.h>
#include "../nn/pooling-inl.h"
#include "../tensor/init_op.h"
#include "./quantization_utils.h"
#if MXNET_USE_ONEDNN == 1
#include "../nn/dnnl/dnnl_pooling-inl.h"
#endif

namespace mxnet {
namespace op {

bool QuantizedPoolingShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_shape,
                           mxnet::ShapeVector* out_shape) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 3U);
  if (!shape_is_known(in_shape->at(0)))
    return false;
  const mxnet::TShape& dshape = (*in_shape)[0];

  const int data_ndims   = dshape.ndim();
  const int kernel_ndims = param.kernel.ndim();
  const int layout       = param.GetLayout(data_ndims);

#if MXNET_USE_ONEDNN == 1
  CHECK(data_ndims == 4U || data_ndims == 5U)
      << "oneDNN QuantizedPoolingOp only supports 4D/5D layout for now, input should be 4D in "
      << "(batch, channel, y, x) or 5D in (batch, channel, d, y, x)";
  CHECK(layout == mshadow::kNCHW || layout == mshadow::kNCDHW)
      << "oneDNN QuantizedPoolingOp only supports NCHW/NCDHW layout for now, saw " << layout;
  CHECK(kernel_ndims == 2U || kernel_ndims == 3U)
      << "oneDNN QuantizedPoolingOp only supports 2D/3D pooling for now, saw" << kernel_ndims;
#else
  CHECK_EQ(data_ndims, 4U) << "quantized_pooling: Input data should be 4D in "
                           << "(batch, channel, y, x)";
  CHECK_EQ(layout, mshadow::kNCHW)
      << "QuantizedPoolingOp only supports NCHW layout for now, saw " << layout;
  CHECK_EQ(kernel_ndims, 2U) << "QuantizedPoolingOp only supports 2D pooling for now";
#endif

  const int D = (data_ndims == 5) ? 2 : 1;
  const int N = 0, H = D + 1, W = D + 2, C = 1;
  mxnet::TShape oshape(data_ndims, -1);

  int idx = 0;
  if (kernel_ndims == 3) {
    CHECK(param.kernel[idx] <= dshape[D] + 2 * param.pad[idx])
        << "kernel size (" << param.kernel[0] << ") exceeds input (" << dshape[D] << " padded to "
        << (dshape[D] + 2 * param.pad[idx]) << ")";
    ++idx;
  }
  CHECK(param.kernel[idx] <= dshape[H] + 2 * param.pad[idx])
      << "kernel size (" << param.kernel[idx] << ") exceeds input (" << dshape[H] << " padded to "
      << (dshape[H] + 2 * param.pad[idx]) << ")";
  ++idx;
  CHECK(param.kernel[idx] <= dshape[W] + 2 * param.pad[idx])
      << "kernel size (" << param.kernel[idx] << ") exceeds input (" << dshape[W] << " padded to "
      << (dshape[W] + 2 * param.pad[idx]) << ")";

#define OUTPUT_SHAPE_VALID_ASSIGN(spatial_dim, idx)                                             \
  {                                                                                             \
    oshape[spatial_dim] =                                                                       \
        1 + (dshape[spatial_dim] + 2 * param.pad[idx] - param.kernel[idx]) / param.stride[idx]; \
  }
#define OUTPUT_SHAPE_FULL_ASSIGN(spatial_dim, idx)                                                 \
  {                                                                                                \
    oshape[spatial_dim] =                                                                          \
        1 + static_cast<int>(std::ceil(                                                            \
                static_cast<float>(dshape[spatial_dim] + 2 * param.pad[idx] - param.kernel[idx]) / \
                param.stride[idx]));                                                               \
  }

  oshape[N] = dshape[N];
  oshape[C] = dshape[C];
  if (param.global_pool) {
    if (data_ndims == 5)
      oshape[D] = 1;
    oshape[H] = 1;
    oshape[W] = 1;
  } else {
    if (param.pooling_convention == pool_enum::kValid) {
      int idx = 0;
      if (data_ndims == 5) {
        OUTPUT_SHAPE_VALID_ASSIGN(D, idx);
        ++idx;
      }
      OUTPUT_SHAPE_VALID_ASSIGN(H, idx);
      ++idx;
      OUTPUT_SHAPE_VALID_ASSIGN(W, idx);
    } else {
      int idx = 0;
      if (data_ndims == 5) {
        OUTPUT_SHAPE_FULL_ASSIGN(D, idx);
        ++idx;
      }
      OUTPUT_SHAPE_FULL_ASSIGN(H, idx);
      ++idx;
      OUTPUT_SHAPE_FULL_ASSIGN(W, idx);
    }
  }

  SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape{1});

  out_shape->clear();
  out_shape->push_back(oshape);
  out_shape->push_back(mxnet::TShape{1});
  out_shape->push_back(mxnet::TShape{1});
  return true;
}

bool QuantizedPoolingType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_type,
                          std::vector<int>* out_type) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 3U);
  CHECK_EQ(out_type->size(), 3U);
  if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
#if MXNET_USE_ONEDNN == 1
    TYPE_ASSIGN_CHECK(*out_type, 0, (*in_type)[0]);
#else
    TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
    TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
#endif
  } else {
    LOG(FATAL) << "QuantizedPoolingOp only supports pool_type=max/avg for now";
  }
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

inline static bool QuantizedPoolingStorageType(const nnvm::NodeAttrs& attrs,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int>* in_attrs,
                                               std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);

  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_ONEDNN == 1
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  if (dev_mask == mshadow::cpu::kDevMask && SupportDNNLQuantizedOps() &&
      SupportDNNLPooling(param)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#else
  CHECK_EQ(out_attrs->size(), 3);
#endif
  for (int& out_attr : *out_attrs)
    out_attr = kDefaultStorage;
  return true;
}

struct QuantizedPoolingDequantizeUnsigned {
  template <typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  float* out,
                                  const SrcDType* in,
                                  const float* imin_range,
                                  const float* imax_range,
                                  const OpReqType req) {
    const float scale = (*imax_range - *imin_range) / 255.0f;
    KERNEL_ASSIGN(out[i], req, in[i] * scale + *imin_range);
  }
};

struct QuantizedPoolingDequantizeInt8 {
  template <typename SrcDType>
  MSHADOW_XINLINE static void Map(int i,
                                  float* out,
                                  const SrcDType* in,
                                  const float* imin_range,
                                  const float* imax_range,
                                  const OpReqType req) {
    const float real_range = MaxAbs(*imax_range, *imin_range);
    KERNEL_ASSIGN(out[i], req, in[i] * (real_range / 127.0f));
  }
};

struct QuantizedPoolingCastGrad {
  template <typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, float* out, const SrcDType* in, const OpReqType req) {
    KERNEL_ASSIGN(out[i], req, static_cast<float>(in[i]));
  }
};

inline bool QuantizedPoolingBackwardShape(const nnvm::NodeAttrs& attrs,
                                          mxnet::ShapeVector* in_shape,
                                          mxnet::ShapeVector* out_shape) {
  CHECK_EQ(in_shape->size(), 7U);
  CHECK_EQ(out_shape->size(), 3U);

  mxnet::ShapeVector fwd_in{in_shape->at(1), in_shape->at(2), in_shape->at(3)};
  mxnet::ShapeVector fwd_out;
  if (QuantizedPoolingShape(attrs, &fwd_in, &fwd_out)) {
    SHAPE_ASSIGN_CHECK(*in_shape, 1, fwd_in[0]);
    SHAPE_ASSIGN_CHECK(*in_shape, 2, fwd_in[1]);
    SHAPE_ASSIGN_CHECK(*in_shape, 3, fwd_in[2]);
    SHAPE_ASSIGN_CHECK(*in_shape, 4, fwd_out[0]);
    SHAPE_ASSIGN_CHECK(*in_shape, 5, fwd_out[1]);
    SHAPE_ASSIGN_CHECK(*in_shape, 6, fwd_out[2]);
    SHAPE_ASSIGN_CHECK(*in_shape, 0, fwd_out[0]);
  } else {
    SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape{1});
    SHAPE_ASSIGN_CHECK(*in_shape, 3, mxnet::TShape{1});
    SHAPE_ASSIGN_CHECK(*in_shape, 5, mxnet::TShape{1});
    SHAPE_ASSIGN_CHECK(*in_shape, 6, mxnet::TShape{1});
    if (shape_is_known(in_shape->at(4))) {
      SHAPE_ASSIGN_CHECK(*in_shape, 0, in_shape->at(4));
    }
    if (shape_is_known(in_shape->at(0))) {
      SHAPE_ASSIGN_CHECK(*in_shape, 4, in_shape->at(0));
    }
  }
  if (shape_is_known(in_shape->at(0)) && shape_is_known(in_shape->at(4))) {
    CHECK_EQ(in_shape->at(0), in_shape->at(4))
        << "_backward_contrib_quantized_pooling expects ograd and saved output shapes to match";
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0, in_shape->at(1));
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape{1});
  return shape_is_known(out_shape->at(0)) && shape_is_known(out_shape->at(1)) &&
         shape_is_known(out_shape->at(2));
}

inline bool QuantizedPoolingBackwardType(const nnvm::NodeAttrs& attrs,
                                         std::vector<int>* in_type,
                                         std::vector<int>* out_type) {
  CHECK_EQ(in_type->size(), 7U);
  CHECK_EQ(out_type->size(), 3U);
  if (in_type->at(0) != -1) {
    CHECK(in_type->at(0) == mshadow::kFloat32 || in_type->at(0) == mshadow::kInt8 ||
          in_type->at(0) == mshadow::kUint8)
        << "_backward_contrib_quantized_pooling only supports float32, int8, or uint8 "
        << "output gradients, while " << in_type->at(0) << " was given";
  } else {
    TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kFloat32);
  }
  TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 3, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 5, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 6, mshadow::kFloat32);

  if (in_type->at(1) != -1) {
    CHECK(in_type->at(1) == mshadow::kInt8 || in_type->at(1) == mshadow::kUint8 ||
          in_type->at(1) == mshadow::kFloat32)
        << "_backward_contrib_quantized_pooling only supports int8, uint8, or float32 "
        << "forward data, while " << in_type->at(1) << " was given";
    TYPE_ASSIGN_CHECK(*in_type, 4, in_type->at(1));
  } else if (in_type->at(4) != -1) {
    CHECK(in_type->at(4) == mshadow::kInt8 || in_type->at(4) == mshadow::kUint8 ||
          in_type->at(4) == mshadow::kFloat32)
        << "_backward_contrib_quantized_pooling only supports int8, uint8, or float32 "
        << "forward output, while " << in_type->at(4) << " was given";
    TYPE_ASSIGN_CHECK(*in_type, 1, in_type->at(4));
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return in_type->at(1) != -1 && in_type->at(4) != -1;
}

inline bool QuantizedPoolingBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int>* in_attrs,
                                                std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 7U);
  CHECK_EQ(out_attrs->size(), 3U);
  *dispatch_mode = DispatchMode::kFCompute;
  for (int& in_attr : *in_attrs)
    in_attr = kDefaultStorage;
  for (int& out_attr : *out_attrs)
    out_attr = kDefaultStorage;
  return true;
}

inline TBlob QuantizedPoolingDequantizeIfNeeded(mshadow::Stream<cpu>* s,
                                                const TBlob& data,
                                                const TBlob& min_range,
                                                const TBlob& max_range,
                                                float* workspace,
                                                size_t* workspace_offset) {
  if (data.type_flag_ == mshadow::kFloat32) {
    return data;
  }

  TBlob dequantized(workspace + *workspace_offset, data.shape_, data.dev_mask(), data.dev_id());
  *workspace_offset += data.Size();
  if (data.type_flag_ == mshadow::kUint8) {
    mxnet_op::Kernel<QuantizedPoolingDequantizeUnsigned, cpu>::Launch(s,
                                                                      data.Size(),
                                                                      dequantized.dptr<float>(),
                                                                      data.dptr<uint8_t>(),
                                                                      min_range.dptr<float>(),
                                                                      max_range.dptr<float>(),
                                                                      kWriteTo);
  } else if (data.type_flag_ == mshadow::kInt8) {
    mxnet_op::Kernel<QuantizedPoolingDequantizeInt8, cpu>::Launch(s,
                                                                  data.Size(),
                                                                  dequantized.dptr<float>(),
                                                                  data.dptr<int8_t>(),
                                                                  min_range.dptr<float>(),
                                                                  max_range.dptr<float>(),
                                                                  kWriteTo);
  } else {
    LOG(FATAL) << "_backward_contrib_quantized_pooling only supports int8, uint8, or float32 "
               << "saved tensors";
  }
  return dequantized;
}

inline TBlob QuantizedPoolingCastGradIfNeeded(mshadow::Stream<cpu>* s,
                                              const TBlob& grad,
                                              float* workspace,
                                              size_t* workspace_offset) {
  if (grad.type_flag_ == mshadow::kFloat32) {
    return grad;
  }

  TBlob cast_grad(workspace + *workspace_offset, grad.shape_, grad.dev_mask(), grad.dev_id());
  *workspace_offset += grad.Size();
  if (grad.type_flag_ == mshadow::kUint8) {
    mxnet_op::Kernel<QuantizedPoolingCastGrad, cpu>::Launch(s,
                                                            grad.Size(),
                                                            cast_grad.dptr<float>(),
                                                            grad.dptr<uint8_t>(),
                                                            kWriteTo);
  } else if (grad.type_flag_ == mshadow::kInt8) {
    mxnet_op::Kernel<QuantizedPoolingCastGrad, cpu>::Launch(s,
                                                            grad.Size(),
                                                            cast_grad.dptr<float>(),
                                                            grad.dptr<int8_t>(),
                                                            kWriteTo);
  } else {
    LOG(FATAL) << "_backward_contrib_quantized_pooling only supports float32, int8, or uint8 "
               << "output gradients";
  }
  return cast_grad;
}

void QuantizedPoolingBackwardComputeCPU(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<TBlob>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<TBlob>& outputs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 7U);
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(req.size(), 3U);
  CHECK(param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling)
      << "QuantizedPoolingOp only supports pool_type=max/avg for now";

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (req[0] != kNullOp) {
    const TBlob& ograd      = inputs[0];
    const TBlob& fwd_data   = inputs[1];
    const TBlob& fwd_min    = inputs[2];
    const TBlob& fwd_max    = inputs[3];
    const TBlob& fwd_output = inputs[4];
    const TBlob& out_min    = inputs[5];
    const TBlob& out_max    = inputs[6];

    const size_t grad_workspace = ograd.type_flag_ == mshadow::kFloat32 ? 0 : ograd.Size();
    const size_t data_workspace =
        fwd_data.type_flag_ == mshadow::kFloat32 ? 0 : fwd_data.Size();
    const size_t output_workspace =
        fwd_output.type_flag_ == mshadow::kFloat32 ? 0 : fwd_output.Size();
    const size_t workspace_size = grad_workspace + data_workspace + output_workspace;
    mshadow::Tensor<cpu, 1, float> workspace =
        ctx.requested[0].get_space_typed<cpu, 1, float>(mshadow::Shape1(workspace_size), s);
    size_t workspace_offset = 0;
    const TBlob float_ograd =
        QuantizedPoolingCastGradIfNeeded(s, ograd, workspace.dptr_, &workspace_offset);
    const TBlob deq_data =
        QuantizedPoolingDequantizeIfNeeded(s, fwd_data, fwd_min, fwd_max, workspace.dptr_,
                                           &workspace_offset);
    const TBlob deq_output =
        QuantizedPoolingDequantizeIfNeeded(s, fwd_output, out_min, out_max, workspace.dptr_,
                                           &workspace_offset);

    std::vector<TBlob> pooling_inputs;
    if (GetNumBackInputs(param) == 5) {
      pooling_inputs = {float_ograd, TBlob(), deq_data, deq_output, TBlob()};
    } else {
      pooling_inputs = {float_ograd, deq_data, deq_output};
    }
    PoolingGradCompute<cpu>(attrs, ctx, pooling_inputs, {req[0]}, {outputs[0]});
  }

  Fill<false>(s, outputs[1], req[1], 0);
  Fill<false>(s, outputs[2], req[2], 0);
}

std::vector<nnvm::NodeEntry> QuantizedPoolingGrad(const nnvm::ObjectPtr& n,
                                                  const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> inputs;
  inputs.reserve(7);
  inputs.push_back(ograds[0]);
  inputs.insert(inputs.end(), n->inputs.begin(), n->inputs.end());
  for (uint32_t i = 0; i < n->num_outputs(); ++i) {
    inputs.emplace_back(n, i, 0);
  }
  return MakeGradNode("_backward_contrib_quantized_pooling", n, inputs, n->attrs.dict);
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
    .add_alias("_npx_quantized_pooling")
    .describe(R"code(Pooling operator for input and output data type of int8.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.

.. Note::
    This operator only supports `pool_type` of `avg` or `max`.
    Backward propagation computes the data gradient and returns zero min/max gradients.)code"
                  ADD_FILELINE)
    .set_num_inputs(3)
    .set_num_outputs(3)
    .set_attr_parser(PoolingParamParser)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"data", "min_data", "max_data"};
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "min_output", "max_output"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedPoolingShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedPoolingType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedPoolingStorageType)
    .set_attr<nnvm::FGradient>("FGradient", QuantizedPoolingGrad)
    .set_attr<FNeedRequantize>(
        "FNeedRequantize",
        [](const NodeAttrs& attrs) {
          const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
          CHECK(param.pool_type == pool_enum::kMaxPooling ||
                param.pool_type == pool_enum::kAvgPooling)
              << "QuantizedPoolingOp only supports pool_type=max/avg for now";
          return false;
        })
    .add_argument("data", "NDArray-or-Symbol", "Input data.")
    .add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
    .add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
    .add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_quantized_pooling)
    .set_num_inputs(7)
    .set_num_outputs(3)
    .set_attr_parser(PoolingParamParser)
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedPoolingBackwardShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedPoolingBackwardType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedPoolingBackwardStorageType)
    .set_attr<FResourceRequest>(
        "FResourceRequest",
        [](const NodeAttrs& n) {
          return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
        })
    .set_attr<FCompute>("FCompute<cpu>", QuantizedPoolingBackwardComputeCPU);

NNVM_REGISTER_OP(Pooling).set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  PoolingParam param;
  param.Init(attrs.dict);
  // TODO(junwu): Uncomment the following line and remove the above lines
  // after pooling op is refactored
  // const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  nnvm::ObjectPtr node = nnvm::Node::Create();
  if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
    node->attrs.op   = Op::Get("_contrib_quantized_pooling");
    node->attrs.name = "quantized_" + attrs.name;
  } else {
    node->attrs.op   = Op::Get("Pooling");
    node->attrs.name = attrs.name;
  }
  node->attrs.dict = attrs.dict;
  if (node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
