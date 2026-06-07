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

#include "broadcast_reduce_op.h"
#include <limits>
#include "../numpy/np_broadcast_reduce_op.h"
#include "elemwise_binary_scalar_op.h"
#include "mxnet/tuple.h"

namespace mxnet {
namespace op {

#if MXNET_USE_CUDA

void ReduceAxesRTCComputeImpl(const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs,
                              const mxnet::TShape& small,
                              const std::string& reducer,
                              const mshadow::Tensor<gpu, 1, char>* workspace,
                              const bool normalize,
                              const std::string& OP,
                              const int ddof) {
  using namespace mshadow;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<gpu>* s = ctx.get_stream<gpu>();

  // Fast path: global (scalar-output) sum/mean over a floating tensor goes to
  // cub::DeviceReduce (double accumulation), which is several times faster than
  // the generic RTC all-reduce. Restricted to fp16/fp32/fp64 (the types the
  // helper instantiates); bf16/int and all axis reductions keep the RTC path.
  {
    const int dt = inputs[0].type_flag_;
    if (req[0] != kNullOp && reducer == "red::sum{}" && OP == "identity" &&
        dst_shape.Size() == 1 && inputs[0].type_flag_ == outputs[0].type_flag_ &&
        (dt == kFloat32 || dt == kFloat64 || dt == kFloat16)) {
      const double count = static_cast<double>(src_shape.Size()) - static_cast<double>(ddof);
      MSHADOW_REAL_TYPE_SWITCH(dt, DType, {
        CubGlobalSumReduce<DType>(ctx, inputs[0], outputs[0], normalize, count, req[0] == kAddTo);
      });
      return;
    }
  }

  const TBlob in_data  = inputs[0].reshape(src_shape);
  const TBlob out_data = outputs[0].reshape(dst_shape);

  // Mean (normalize) over an fp16 output: the reduce writes the *un-normalized*
  // sum to the output before the division, and that sum overflows fp16's ~65504
  // range for a large reduced extent (e.g. np.mean(fp16) over a 200k axis -> inf,
  // diverging from NumPy which stays finite). Accumulate the sum into an fp32
  // scratch, then divide-and-cast to the fp16 output. (bf16 shares fp32's
  // exponent range, so it does not overflow and keeps the in-place path. The
  // workspace==nullptr guard scopes this to the direct mean op, where we own the
  // tempspace; workspace-passing callers manage their own buffers.)
  if (workspace == nullptr && normalize && reducer == "red::sum{}" &&
      outputs[0].type_flag_ == mshadow::kFloat16) {
    const size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape, kWriteTo, src_shape);
    const size_t temp_off       = ((workspace_size + sizeof(float) - 1) / sizeof(float)) * sizeof(float);
    const size_t temp_bytes     = dst_shape.Size() * sizeof(float);
    Tensor<gpu, 1, char> buf =
        ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_off + temp_bytes), s);
    Tensor<gpu, 1, char> ws(buf.dptr_, Shape1(workspace_size), s);
    const TBlob temp_fp32(reinterpret_cast<float*>(buf.dptr_ + temp_off),
                          dst_shape,
                          gpu::kDevMask,
                          mshadow::kFloat32,
                          ctx.run_ctx.ctx.dev_id);
    BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, temp_fp32, kWriteTo, ws, in_data, reducer, NDim, OP);
    });
    NumpyBinaryScalarParam p{};
    p.scalar = static_cast<double>(src_shape.Size() / dst_shape.Size() - ddof);
    NodeAttrs a;
    a.parsed = p;
    BinaryScalarRTCCompute{"div"}(a, ctx, {temp_fp32}, {req[0]}, {out_data});  // NOLINT
    return;
  }

  Tensor<gpu, 1, char> w;
  if (workspace == nullptr) {
    size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape, req[0], src_shape);
    w         = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
    workspace = &w;
  }
  BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
    broadcast::RTCReduce(ctx, out_data, req[0], *workspace, in_data, reducer, NDim, OP);
  });
  if (normalize) {
    NumpyBinaryScalarParam p{};
    p.scalar = static_cast<double>(src_shape.Size() / dst_shape.Size() - ddof);
    NodeAttrs a;
    a.parsed = p;
    BinaryScalarRTCCompute{"div"}(a, ctx, {out_data}, {kWriteInplace}, {out_data});  // NOLINT
  }
}

namespace {
template <typename Param>
void PrepareReduce(const Param& param,
                   const std::vector<TBlob>& inputs,
                   const std::vector<TBlob>& outputs,
                   mxnet::TShape* shape,
                   int* ddof);

template <>
void PrepareReduce<ReduceAxesParam>(const ReduceAxesParam& param,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<TBlob>& outputs,
                                    mxnet::TShape* small,
                                    int* ddof) {
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, param.exclude);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesNoDTypeParam>(const NumpyReduceAxesNoDTypeParam& param,
                                                const std::vector<TBlob>& inputs,
                                                const std::vector<TBlob>& outputs,
                                                mxnet::TShape* small,
                                                int* ddof) {
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesParam>(const NumpyReduceAxesParam& param,
                                         const std::vector<TBlob>& inputs,
                                         const std::vector<TBlob>& outputs,
                                         mxnet::TShape* small,
                                         int* ddof) {
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesBoolParam>(const NumpyReduceAxesBoolParam& param,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<TBlob>& outputs,
                                             mxnet::TShape* small,
                                             int* ddof) {
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

}  // namespace

template <typename Param, int init>
void ReduceAxesRTCCompute<Param, init>::operator()(const nnvm::NodeAttrs& attrs,
                                                   const OpContext& ctx,
                                                   const std::vector<TBlob>& inputs,
                                                   const std::vector<OpReqType>& req,
                                                   const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp)
    return;
  mxnet::TShape small;
  int ddof;
  const auto& param = nnvm::get<Param>(attrs.parsed);
  CHECK_NE(req[0], kWriteInplace) << "Reduce does not support write in-place";
  PrepareReduce(param, inputs, outputs, &small, &ddof);
  if (outputs[0].shape_.Size() == 0U)
    return;  // zero-size tensor
  if (inputs[0].shape_.Size() == 0) {
    if (normalize && mxnet::common::is_float(outputs[0].type_flag_)) {
      LOG(WARNING) << "WARNING: Mean of empty slice.";
      NumpyBinaryScalarParam p{};
      p.scalar = std::numeric_limits<float>::quiet_NaN();
      NodeAttrs a;
      a.parsed = p;
      BinaryScalarRTCCompute{"right"}(a, ctx, outputs, {kWriteTo}, outputs);  // NOLINT
    } else {
      if (normalize) {
        LOG(WARNING) << "WARNING: nan is outside the range of"
                     << "representable values of type 'int'";
      }
      if (init == 0 && req[0] == kAddTo)
        return;
      NumpyBinaryScalarParam p{};
      p.scalar = init;
      NodeAttrs a;
      a.parsed = p;
      BinaryScalarRTCCompute{"right"}(a, ctx, outputs, {req[0]}, outputs);  // NOLINT
    }
    return;
  }

  ReduceAxesRTCComputeImpl(ctx, inputs, req, outputs, small, reducer, nullptr, normalize, OP, ddof);
}

template struct ReduceAxesRTCCompute<ReduceAxesParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesParam, 1>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesNoDTypeParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesBoolParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesBoolParam, 1>;

#endif

}  // namespace op
}  // namespace mxnet
