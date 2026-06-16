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
 * \file convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Bing Xu, Jun Wu, Da Zheng
 */
#ifndef MXNET_OPERATOR_NN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "./im2col.h"

namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs { kData, kWeight, kBias };
enum ConvolutionOpOutputs { kOut };
enum ConvolutionOpResource { kTempSpace };
enum ConvolutionOpCudnnTune { kOff, kLimited, kFastest };
}  // namespace conv

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad)
        .set_default(mxnet::TShape(0, 0))
        .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter)
        .set_lower_bound(1)
        .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1).describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_lower_bound(0).describe(
        "Maximum temporary workspace allowed (MB) in convolution."
        "This parameter has two usages. When CUDNN is not used, it determines the "
        "effective batch size of the convolution kernel. When CUDNN is used, it controls "
        "the maximum temporary storage used for tuning the best CUDNN kernel when "
        "`limited_workspace` strategy is used.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false).describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
        .add_enum("off", conv::kOff)
        .add_enum("limited_workspace", conv::kLimited)
        .add_enum("fastest", conv::kFastest)
        .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false).describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
        .add_enum("NCW", mshadow::kNCW)
        .add_enum("NCHW", mshadow::kNCHW)
        .add_enum("NCDHW", mshadow::kNCDHW)
        .add_enum("NWC", mshadow::kNWC)
        .add_enum("NHWC", mshadow::kNHWC)
        .add_enum("NDHWC", mshadow::kNDHWC)
        .set_default(dmlc::optional<int>())
        .describe(
            "Set layout for input, output and weight. Empty for\n    "
            "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d."
            "NHWC and NDHWC are only supported on GPU.");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const ConvolutionParam& other) const {
    return this->kernel == other.kernel && this->stride == other.stride &&
           this->dilate == other.dilate && this->pad == other.pad &&
           this->num_filter == other.num_filter && this->num_group == other.num_group &&
           this->workspace == other.workspace && this->no_bias == other.no_bias &&
           this->cudnn_tune == other.cudnn_tune && this->cudnn_off == other.cudnn_off &&
           this->layout == other.layout;
  }
  std::string CudnnTune2String(int cudnn_tune) {
    switch (cudnn_tune) {
      case conv::kOff:
        return "off";
      case conv::kLimited:
        return "limited_workspace";
      case conv::kFastest:
        return "fastest";
      default:
        LOG(FATAL) << "Unknown cudnn_tune enum " << cudnn_tune;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  std::string Layout2String(int layout) {
    switch (layout) {
      case mshadow::kNCW:
        return "NCW";
      case mshadow::kNCHW:
        return "NCHW";
      case mshadow::kNCDHW:
        return "NCDHW";
      case mshadow::kNHWC:
        return "NHWC";
      case mshadow::kNDHWC:
        return "NDHWC";
      default:
        LOG(FATAL) << "Unknown layout enum " << layout;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream kernel_s, stride_s, dilate_s, pad_s, num_filter_s, num_group_s, workspace_s,
        no_bias_s, cudnn_tune_s, cudnn_off_s, layout_s;
    kernel_s << kernel;
    stride_s << stride;
    dilate_s << dilate;
    pad_s << pad;
    num_filter_s << num_filter;
    num_group_s << num_group;
    workspace_s << workspace;
    no_bias_s << no_bias;
    cudnn_tune_s << cudnn_tune;
    cudnn_off_s << cudnn_off;
    layout_s << layout;
    (*dict)["kernel"]     = kernel_s.str();
    (*dict)["stride"]     = stride_s.str();
    (*dict)["dilate"]     = dilate_s.str();
    (*dict)["pad"]        = pad_s.str();
    (*dict)["num_filter"] = num_filter_s.str();
    (*dict)["num_group"]  = num_group_s.str();
    (*dict)["workspace"]  = workspace_s.str();
    (*dict)["no_bias"]    = no_bias_s.str();
    if (cudnn_tune.has_value()) {
      (*dict)["cudnn_tune"] = CudnnTune2String(cudnn_tune.value());
    } else {
      (*dict)["cudnn_tune"] = cudnn_tune_s.str();
    }
    (*dict)["cudnn_off"] = cudnn_off_s.str();
    if (layout.has_value()) {
      (*dict)["layout"] = Layout2String(layout.value());
    } else {
      (*dict)["layout"] = layout_s.str();
    }
  }
};

void ConvolutionParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<ConvolutionParam> ConvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::ConvolutionParam> {
  size_t operator()(const mxnet::op::ConvolutionParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.kernel);
    ret        = dmlc::HashCombine(ret, val.stride);
    ret        = dmlc::HashCombine(ret, val.dilate);
    ret        = dmlc::HashCombine(ret, val.pad);
    ret        = dmlc::HashCombine(ret, val.num_filter);
    ret        = dmlc::HashCombine(ret, val.num_group);
    ret        = dmlc::HashCombine(ret, val.workspace);
    ret        = dmlc::HashCombine(ret, val.no_bias);
    ret        = dmlc::HashCombine(ret, val.cudnn_tune);
    ret        = dmlc::HashCombine(ret, val.cudnn_off);
    ret        = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class ConvolutionOp {
 public:
  void Init(ConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    if (param_.layout.has_value()) {
      CHECK(param_.layout.value() == mshadow::kNCW || param_.layout.value() == mshadow::kNCHW ||
            param_.layout.value() == mshadow::kNCDHW)
          << "Only support NCW, NCHW and NCDHW layout";
    }
  }

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    // CHECK_EQ(req[conv::kOut], kWriteTo);
    _Forward(ctx,
             in_data[conv::kData],
             in_data[conv::kWeight],
             param_.no_bias ? nullptr : &in_data[conv::kBias],
             req[conv::kOut],
             out_data[conv::kOut]);
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    // We expect 2 inputs: in data and weight. We don't need bias for
    // computing gradient.
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);

    auto workspace = _BackwardData(
        ctx, out_grad[conv::kOut], in_data[conv::kWeight], req[conv::kData], in_grad[conv::kData]);
    _BackwardWeightsBias(workspace,
                         ctx,
                         out_grad[conv::kOut],
                         in_data[conv::kData],
                         req[conv::kWeight],
                         in_grad[conv::kWeight],
                         param_.no_bias ? OpReqType() : req[conv::kBias],
                         param_.no_bias ? nullptr : &in_grad[conv::kBias]);
  }

 private:
  Tensor<xpu, 1, DType> _Forward(const OpContext& ctx,
                                 const TBlob& in_data,
                                 const TBlob& in_weights,
                                 const TBlob* in_bias,
                                 const OpReqType req,
                                 const TBlob& out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    LayerSetUp(in_data.shape_, out_data.shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    Tensor<xpu, 1, DType> workspace;

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = conv_out_channels_ / group_;
    index_t N = conv_out_spatial_dim_;
    index_t K = kernel_dim_;
    Tensor<xpu, 3, DType> weight_3d =
        in_weights.get_with_shape<xpu, 3, DType>(Shape3(group_, M, K), s);
    Tensor<xpu, 4, DType> output_4d =
        out_data.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, M, N), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d =
          in_data.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d  = input_4d[n];
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], input_3d[g], output_3d[g], false, false, s, req);
        }
      }
    } else {
      // allocate workspace for col_buffer
      workspace = ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
          Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_data.shape_[i + 1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d =
          col_buffer.get_with_shape<xpu, 3, DType>(Shape3(group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        // Defensively zero the im2col column buffer before each fill.  The
        // requested temp-space workspace is uninitialized scratch from the
        // resource manager; if any cell is left unwritten before the GEMM reads
        // it (observed intermittently on the native CPU path for large-kernel,
        // padded configs — the unwritten cells correspond to padding, whose
        // correct value is 0), the GEMM otherwise multiplies in stale memory and
        // can emit NaN in the output. Zeroing first makes the read deterministic
        // and matches the zero-fill semantics im2col already uses for padding.
        col_buffer_3d = scalar<DType>(0);
        // transform image to col_buffer in order to use gemm
        im2col(s,
               in_data.dptr<DType>() + n * input_dim_,
               in_data.shape_,
               col_buffer.shape_,
               param_.kernel,
               param_.pad,
               param_.stride,
               param_.dilate,
               col_buffer.dptr<DType>());
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          //   Assign(output_3d[g], req, dot(weight_3d[g], col_buffer_3d[g]));
          linalg_gemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, s, req);
        }
      }
    }

    if (bias_term_) {
      CHECK(in_bias != nullptr);
      Tensor<xpu, 1, DType> bias      = in_bias->get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data.get_with_shape<xpu, 3, DType>(
          Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
    return workspace;
  }

  // Computes dLoss/dData
  Tensor<xpu, 1, DType> _BackwardData(const OpContext& ctx,
                                      const TBlob& out_grad,
                                      const TBlob& weights,
                                      const OpReqType data_grad_req,
                                      const TBlob& data_grad_dst) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(weights.CheckContiguous(), true);
    LayerSetUp(data_grad_dst.shape_, out_grad.shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    Tensor<xpu, 1, DType> workspace;

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = kernel_dim_;
    index_t N = conv_out_spatial_dim_;
    index_t K = conv_out_channels_ / group_;
    Tensor<xpu, 3, DType> weight_3d =
        weights.get_with_shape<xpu, 3, DType>(Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d =
        out_grad.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, K, N), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> in_grad_4d =
          data_grad_dst.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> in_grad_3d  = in_grad_4d[n];
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], out_grad_3d[g], in_grad_3d[g], true, false, s, data_grad_req);
        }
      }
    } else {
      // allocate workspace for col_buffer
      workspace = ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
          Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_grad.shape_[i + 1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d =
          col_buffer.get_with_shape<xpu, 3, DType>(Shape3(group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, s);
        }
        col2im(s,
               col_buffer.dptr<DType>(),
               data_grad_dst.shape_,
               col_buffer.shape_,
               param_.kernel,
               param_.pad,
               param_.stride,
               param_.dilate,
               data_grad_dst.dptr<DType>() + n * input_dim_,
               data_grad_req);
      }
    }
    return workspace;
  }

  // Computes dLoss/dWeights and dLoss/dBias
  void _BackwardWeightsBias(Tensor<xpu, 1, DType> workspace,
                            const OpContext& ctx,
                            const TBlob& out_grad,
                            const TBlob& data,
                            const OpReqType weights_grad_req,
                            const TBlob& weights_grad_dst,
                            const OpReqType bias_grad_req,
                            const TBlob* const bias_grad_dst) {
    using namespace mshadow;
    using namespace mshadow::expr;
    LayerSetUp(data.shape_, out_grad.shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = kernel_dim_;
    index_t N = conv_out_spatial_dim_;
    index_t K = conv_out_channels_ / group_;
    Tensor<xpu, 4, DType> out_grad_4d =
        out_grad.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, K, N), s);
    Tensor<xpu, 3, DType> dweight_3d =
        weights_grad_dst.get_with_shape<xpu, 3, DType>(Shape3(group_, K, M), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d =
          data.get_with_shape<xpu, 4, DType>(Shape4(num_, group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d    = input_4d[n];
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          auto request = (n == 0) ? weights_grad_req : kAddTo;
          linalg_gemm(out_grad_3d[g], input_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    } else {
      // allocate workspace for col_buffer
      if (workspace.dptr_ == nullptr) {
        workspace = ctx.requested[conv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(col_buffer_size_), s);
      }
      // calculate the shape of col_buffer
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_grad.shape_[i + 1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d =
          col_buffer.get_with_shape<xpu, 3, DType>(Shape3(group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        // dWeight should accumulate across the batch and group
        im2col(s,
               data.dptr<DType>() + n * input_dim_,
               data.shape_,
               col_buffer.shape_,
               param_.kernel,
               param_.pad,
               param_.stride,
               param_.dilate,
               col_buffer.dptr<DType>());
        for (index_t g = 0; g < group_; ++g) {
          auto request = (n == 0) ? weights_grad_req : kAddTo;
          linalg_gemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    }

    // bias gradient
    if (bias_term_) {
      CHECK(bias_grad_dst != nullptr);
      Tensor<xpu, 1, DType> dbias = bias_grad_dst->get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout  = out_grad.get_with_shape<xpu, 3, DType>(
          Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, bias_grad_req, sumall_except_dim<1>(dout));
    }
  }

  void LayerSetUp(const mxnet::TShape& ishape, const mxnet::TShape& oshape) {
    channel_axis_                    = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const int num_axes               = param_.kernel.ndim() + 2;
    num_spatial_axes_                = num_axes - first_spatial_axis;
    is_1x1_                          = true;
    for (int i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_)
        break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_             = ishape[1];
    group_                = param_.num_group;
    conv_out_channels_    = param_.num_filter;
    conv_in_channels_     = channels_;
    bias_term_            = !param_.no_bias;
    kernel_dim_           = conv_in_channels_ / group_ * param_.kernel.Size();
    weight_offset_        = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    col_offset_           = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_        = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_          = ishape.ProdShape(1, ishape.ndim());
    output_dim_         = oshape.ProdShape(1, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;
  }

 private:
  ConvolutionParam param_;
  index_t channel_axis_;          // channel axis of the input
  index_t channels_;              // number of channels of input image
  index_t num_spatial_axes_;      // number of spatial axes
  index_t num_;                   // batch size
  index_t group_;                 // number of groups
  index_t conv_out_channels_;     // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_channels_;      // number of input channels
  index_t kernel_dim_;            // number of input channels per group * kernel size
  index_t weight_offset_;         // number of output channels per group * kernel_dim_
  index_t col_offset_;
  index_t output_offset_;
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;

  template <typename xpu_, typename DType_>
  friend class DeconvolutionOp;
};  // class ConvolutionOp

inline void AssignCPUHalfFromFloat(mshadow::half::half_t* out,
                                   index_t size,
                                   const float* values,
                                   OpReqType req) {
  if (req == kNullOp) {
    return;
  }
#pragma omp parallel for
  for (index_t i = 0; i < size; ++i) {
    const float value = (req == kAddTo) ? static_cast<float>(out[i]) + values[i] : values[i];
    out[i]           = static_cast<mshadow::half::half_t>(value);
  }
}

inline void InitCPUHalfFloatAccum(const mshadow::half::half_t* src,
                                  index_t size,
                                  float* dst,
                                  OpReqType req) {
  if (req == kAddTo) {
#pragma omp parallel for
    for (index_t i = 0; i < size; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  } else {
    std::fill(dst, dst + size, 0.0f);
  }
}

template <typename AType, typename BType, typename CType>
inline void ConvCPUFloatGemm(const Tensor<cpu, 2, AType>& A,
                             const Tensor<cpu, 2, BType>& B,
                             const Tensor<cpu, 2, CType>& C,
                             bool tA,
                             bool tB,
                             OpReqType req) {
  if (req == kNullOp) {
    return;
  }
  CHECK_EQ((tA ? A.size(1) : A.size(0)), C.size(0))
      << "Non compatible matrix dimensions between inputs A and C for gemm";
  CHECK_EQ((tB ? B.size(0) : B.size(1)), C.size(1))
      << "Non compatible matrix dimensions between inputs B and C for gemm";
  CHECK_EQ((tA ? A.size(0) : A.size(1)), (tB ? B.size(1) : B.size(0)))
      << "Non compatible matrix dimensions between inputs A and B for gemm";
  const index_t m = C.size(0);
  const index_t n = C.size(1);
  const index_t k = tA ? A.size(0) : A.size(1);
#pragma omp parallel for collapse(2)
  for (index_t row = 0; row < m; ++row) {
    for (index_t col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (index_t kk = 0; kk < k; ++kk) {
        const index_t a_idx = tA ? kk * A.stride_ + row : row * A.stride_ + kk;
        const index_t b_idx = tB ? col * B.stride_ + kk : kk * B.stride_ + col;
        acc += static_cast<float>(A.dptr_[a_idx]) * static_cast<float>(B.dptr_[b_idx]);
      }
      const index_t c_idx = row * C.stride_ + col;
      if (req == kAddTo) {
        acc += static_cast<float>(C.dptr_[c_idx]);
      }
      C.dptr_[c_idx] = static_cast<CType>(acc);
    }
  }
}

inline void im2col_cpu_half_to_float(const mshadow::half::half_t* data_im,
                                     const int channels,
                                     const int height,
                                     const int width,
                                     const int kernel_h,
                                     const int kernel_w,
                                     const int pad_h,
                                     const int pad_w,
                                     const int stride_h,
                                     const int stride_w,
                                     const int dilation_h,
                                     const int dilation_w,
                                     float* data_col) {
  const int output_h     = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w     = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0.0f;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = static_cast<float>(data_im[input_row * width + input_col]);
              } else {
                *(data_col++) = 0.0f;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

inline void im2col_nd_core_cpu_half_to_float(const mshadow::half::half_t* data_input,
                                             const mxnet::TShape& im_shape,
                                             const mxnet::TShape& col_shape,
                                             const mxnet::TShape& kernel_shape,
                                             const mxnet::TShape& pad,
                                             const mxnet::TShape& stride,
                                             const mxnet::TShape& dilation,
                                             float* data_output) {
  const int num_spatial_axes = kernel_shape.ndim();
  index_t kernel_size        = 1;
  for (index_t i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const index_t channels_col = col_shape[0];
  std::vector<index_t> d_offset(num_spatial_axes, 0);
  std::vector<index_t> d_iter(num_spatial_axes, 0);
  for (index_t c_col = 0; c_col < channels_col; ++c_col) {
    index_t offset = c_col;
    for (int d_i = static_cast<int>(num_spatial_axes) - 1; d_i >= 0; --d_i) {
      if (d_i < static_cast<int>(num_spatial_axes) - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented;) {
      index_t index_col = c_col;
      index_t index_im  = c_col / kernel_size;
      bool is_padding   = false;
      for (index_t d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const index_t d = d_iter[d_i];
        const int d_im  = static_cast<int>(d * stride[d_i] + d_offset[d_i] * dilation[d_i]) -
                         static_cast<int>(pad[d_i]);
        is_padding |= d_im < 0 || d_im >= static_cast<int>(im_shape[d_i + 2]);
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= static_cast<index_t>(im_shape[d_i + 2]);
        index_im += d_im;
      }
      data_output[index_col] = is_padding ? 0.0f : static_cast<float>(data_input[index_im]);
      incremented            = false;
      for (int d_i = static_cast<int>(num_spatial_axes) - 1; d_i >= 0; --d_i) {
        const index_t d_max = col_shape[d_i + 1];
        CHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] + 1 == d_max) {
          d_iter[d_i] = 0;
        } else {
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }
  }
}

inline void im2col_cpu_half_to_float(mshadow::Stream<cpu>* s,
                                     const mshadow::half::half_t* data_im,
                                     const mxnet::TShape& im_shape,
                                     const mxnet::TShape& col_shape,
                                     const mxnet::TShape& kernel_shape,
                                     const mxnet::TShape& pad,
                                     const mxnet::TShape& stride,
                                     const mxnet::TShape& dilation,
                                     float* data_col) {
  if (2 == kernel_shape.ndim()) {
    im2col_cpu_half_to_float(data_im,
                             im_shape[1],
                             im_shape[2],
                             im_shape[3],
                             kernel_shape[0],
                             kernel_shape[1],
                             pad[0],
                             pad[1],
                             stride[0],
                             stride[1],
                             dilation[0],
                             dilation[1],
                             data_col);
  } else {
    im2col_nd_core_cpu_half_to_float(
        data_im, im_shape, col_shape, kernel_shape, pad, stride, dilation, data_col);
  }
}

template <>
inline Tensor<cpu, 1, mshadow::half::half_t> ConvolutionOp<cpu, mshadow::half::half_t>::_Forward(
    const OpContext& ctx,
    const TBlob& in_data,
    const TBlob& in_weights,
    const TBlob* in_bias,
    const OpReqType req,
    const TBlob& out_data) {
  using Half = mshadow::half::half_t;
  LayerSetUp(in_data.shape_, out_data.shape_);
  Stream<cpu>* s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, Half> workspace;

  const index_t M = conv_out_channels_ / group_;
  const index_t N = conv_out_spatial_dim_;
  const index_t K = kernel_dim_;
  Tensor<cpu, 3, Half> weight_3d = in_weights.get_with_shape<cpu, 3, Half>(Shape3(group_, M, K), s);
  Tensor<cpu, 4, Half> output_4d =
      out_data.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, M, N), s);

  if (is_1x1_) {
    Tensor<cpu, 4, Half> input_4d =
        in_data.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, K, N), s);
    for (index_t n = 0; n < num_; ++n) {
      Tensor<cpu, 3, Half> input_3d  = input_4d[n];
      Tensor<cpu, 3, Half> output_3d = output_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        ConvCPUFloatGemm(weight_3d[g], input_3d[g], output_3d[g], false, false, req);
      }
    }
  } else {
    Tensor<cpu, 1, float> float_workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<cpu, 1, float>(
            Shape1(col_buffer_size_), s);
    mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_data.shape_[i + 1];
    }
    TBlob col_buffer(float_workspace.dptr_, col_buffer_shape, cpu::kDevMask, DataType<float>::kFlag);
    Tensor<cpu, 3, float> col_buffer_3d =
        col_buffer.get_with_shape<cpu, 3, float>(Shape3(group_, K, N), s);
    for (index_t n = 0; n < num_; ++n) {
      im2col_cpu_half_to_float(s,
                               in_data.dptr<Half>() + n * input_dim_,
                               in_data.shape_,
                               col_buffer.shape_,
                               param_.kernel,
                               param_.pad,
                               param_.stride,
                               param_.dilate,
                               col_buffer.dptr<float>());
      Tensor<cpu, 3, Half> output_3d = output_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        ConvCPUFloatGemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, req);
      }
    }
  }

  if (bias_term_) {
    CHECK(in_bias != nullptr);
    Tensor<cpu, 1, Half> bias      = in_bias->get<cpu, 1, Half>(s);
    Tensor<cpu, 3, Half> output_3d = out_data.get_with_shape<cpu, 3, Half>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
    output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
  }
  return workspace;
}

template <>
inline Tensor<cpu, 1, mshadow::half::half_t> ConvolutionOp<cpu, mshadow::half::half_t>::_BackwardData(
    const OpContext& ctx,
    const TBlob& out_grad,
    const TBlob& weights,
    const OpReqType data_grad_req,
    const TBlob& data_grad_dst) {
  using Half = mshadow::half::half_t;
  CHECK_EQ(weights.CheckContiguous(), true);
  LayerSetUp(data_grad_dst.shape_, out_grad.shape_);
  Stream<cpu>* s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, Half> workspace;

  const index_t M = kernel_dim_;
  const index_t N = conv_out_spatial_dim_;
  const index_t K = conv_out_channels_ / group_;
  Tensor<cpu, 3, Half> weight_3d = weights.get_with_shape<cpu, 3, Half>(Shape3(group_, K, M), s);
  Tensor<cpu, 4, Half> out_grad_4d =
      out_grad.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, K, N), s);

  if (is_1x1_) {
    Tensor<cpu, 4, Half> in_grad_4d =
        data_grad_dst.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, M, N), s);
    for (index_t n = 0; n < num_; ++n) {
      Tensor<cpu, 3, Half> in_grad_3d  = in_grad_4d[n];
      Tensor<cpu, 3, Half> out_grad_3d = out_grad_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        ConvCPUFloatGemm(weight_3d[g], out_grad_3d[g], in_grad_3d[g], true, false, data_grad_req);
      }
    }
  } else {
    Tensor<cpu, 1, float> float_workspace =
        ctx.requested[conv::kTempSpace].get_space_typed<cpu, 1, float>(
            Shape1(col_buffer_size_ + input_dim_), s);
    float* col_buffer_ptr  = float_workspace.dptr_;
    float* data_grad_float = float_workspace.dptr_ + col_buffer_size_;
    mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_grad.shape_[i + 1];
    }
    TBlob col_buffer(col_buffer_ptr, col_buffer_shape, cpu::kDevMask, DataType<float>::kFlag);
    Tensor<cpu, 3, float> col_buffer_3d =
        col_buffer.get_with_shape<cpu, 3, float>(Shape3(group_, M, N), s);
    for (index_t n = 0; n < num_; ++n) {
      Tensor<cpu, 3, Half> out_grad_3d = out_grad_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        ConvCPUFloatGemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, kWriteTo);
      }
      col2im(s,
             col_buffer.dptr<float>(),
             data_grad_dst.shape_,
             col_buffer.shape_,
             param_.kernel,
             param_.pad,
             param_.stride,
             param_.dilate,
             data_grad_float,
             kWriteTo);
      AssignCPUHalfFromFloat(
          data_grad_dst.dptr<Half>() + n * input_dim_, input_dim_, data_grad_float, data_grad_req);
    }
  }
  return workspace;
}

template <>
inline void ConvolutionOp<cpu, mshadow::half::half_t>::_BackwardWeightsBias(
    Tensor<cpu, 1, mshadow::half::half_t> workspace,
    const OpContext& ctx,
    const TBlob& out_grad,
    const TBlob& data,
    const OpReqType weights_grad_req,
    const TBlob& weights_grad_dst,
    const OpReqType bias_grad_req,
    const TBlob* const bias_grad_dst) {
  using Half = mshadow::half::half_t;
  LayerSetUp(data.shape_, out_grad.shape_);
  Stream<cpu>* s = ctx.get_stream<cpu>();

  const index_t M = kernel_dim_;
  const index_t N = conv_out_spatial_dim_;
  const index_t K = conv_out_channels_ / group_;
  const index_t weight_grad_size = weights_grad_dst.Size();
  Tensor<cpu, 4, Half> out_grad_4d =
      out_grad.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, K, N), s);
  Tensor<cpu, 1, float> float_workspace =
      ctx.requested[conv::kTempSpace].get_space_typed<cpu, 1, float>(
          Shape1(weight_grad_size + (is_1x1_ ? 0 : col_buffer_size_)), s);
  float* dweight_float = float_workspace.dptr_;

  if (weights_grad_req != kNullOp) {
    InitCPUHalfFloatAccum(
        weights_grad_dst.dptr<Half>(), weight_grad_size, dweight_float, weights_grad_req);
    TBlob dweight_buffer(dweight_float,
                         weights_grad_dst.shape_,
                         cpu::kDevMask,
                         DataType<float>::kFlag);
    Tensor<cpu, 3, float> dweight_3d =
        dweight_buffer.get_with_shape<cpu, 3, float>(Shape3(group_, K, M), s);

    if (is_1x1_) {
      Tensor<cpu, 4, Half> input_4d =
          data.get_with_shape<cpu, 4, Half>(Shape4(num_, group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<cpu, 3, Half> input_3d    = input_4d[n];
        Tensor<cpu, 3, Half> out_grad_3d = out_grad_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          ConvCPUFloatGemm(out_grad_3d[g], input_3d[g], dweight_3d[g], false, true, kAddTo);
        }
      }
    } else {
      float* col_buffer_ptr = float_workspace.dptr_ + weight_grad_size;
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_grad.shape_[i + 1];
      }
      TBlob col_buffer(col_buffer_ptr, col_buffer_shape, cpu::kDevMask, DataType<float>::kFlag);
      Tensor<cpu, 3, float> col_buffer_3d =
          col_buffer.get_with_shape<cpu, 3, float>(Shape3(group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<cpu, 3, Half> out_grad_3d = out_grad_4d[n];
        im2col_cpu_half_to_float(s,
                                 data.dptr<Half>() + n * input_dim_,
                                 data.shape_,
                                 col_buffer.shape_,
                                 param_.kernel,
                                 param_.pad,
                                 param_.stride,
                                 param_.dilate,
                                 col_buffer.dptr<float>());
        for (index_t g = 0; g < group_; ++g) {
          ConvCPUFloatGemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, kAddTo);
        }
      }
    }
    AssignCPUHalfFromFloat(
        weights_grad_dst.dptr<Half>(), weight_grad_size, dweight_float, kWriteTo);
  }

  if (bias_term_) {
    CHECK(bias_grad_dst != nullptr);
    if (bias_grad_req != kNullOp) {
      Half* dbias = bias_grad_dst->dptr<Half>();
#pragma omp parallel for
      for (index_t c = 0; c < conv_out_channels_; ++c) {
        float acc = bias_grad_req == kAddTo ? static_cast<float>(dbias[c]) : 0.0f;
        const Half* dout = out_grad.dptr<Half>() + c * conv_out_spatial_dim_;
        for (index_t n = 0; n < num_; ++n) {
          const Half* dout_n = dout + n * conv_out_channels_ * conv_out_spatial_dim_;
          for (index_t i = 0; i < conv_out_spatial_dim_; ++i) {
            acc += static_cast<float>(dout_n[i]);
          }
        }
        dbias[c] = static_cast<Half>(acc);
      }
    }
  }
}

template <typename xpu>
void ConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[conv::kData].type_flag_, DType, {
    ConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template <typename xpu>
void ConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob& out_grad             = inputs[0];
  const std::vector<TBlob>& in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    ConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CONVOLUTION_INL_H_
