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
 * \file np_interp_op-inl.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "../tensor/ordering_op-inl.h"
#include "../tensor/matrix_op-inl.h"
#include "../tensor/elemwise_binary_scalar_op.h"
#include "../../common/utils.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyInterpParam : public dmlc::Parameter<NumpyInterpParam> {
  dmlc::optional<double> left;
  dmlc::optional<double> right;
  dmlc::optional<double> period;
  double x_scalar;
  bool x_is_scalar;
  DMLC_DECLARE_PARAMETER(NumpyInterpParam) {
    DMLC_DECLARE_FIELD(left)
        .set_default(dmlc::optional<double>())
        .describe("Value to return for x < xp[0], default is fp[0].");
    DMLC_DECLARE_FIELD(right)
        .set_default(dmlc::optional<double>())
        .describe("Value to return for x > xp[-1], default is fp[-1].");
    DMLC_DECLARE_FIELD(period)
        .set_default(dmlc::optional<double>())
        .describe(
            "A period for the x-coordinates. This parameter allows"
            "the proper interpolation of angular x-coordinates. Parameters"
            "left and right are ignored if period is specified.");
    DMLC_DECLARE_FIELD(x_scalar).set_default(0.0).describe("x is a scalar input");
    DMLC_DECLARE_FIELD(x_is_scalar)
        .set_default(false)
        .describe("Flag that determines whether input is a scalar");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream left_s, right_s, period_s, x_scalar_s, x_is_scalar_s;
    left_s << left;
    right_s << right;
    period_s << period;
    x_scalar_s << x_scalar;
    x_is_scalar_s << x_is_scalar;
    (*dict)["left"]        = left_s.str();
    (*dict)["right"]       = right_s.str();
    (*dict)["period"]      = period_s.str();
    (*dict)["x_scalar"]    = x_scalar_s.str();
    (*dict)["x_is_scalar"] = x_is_scalar_s.str();
  }
};

struct interp {
  MSHADOW_XINLINE static void Map(index_t i,
                                  double* out,
                                  const double* x,
                                  const double* xp,
                                  const double* fp,
                                  const size_t dsize,
                                  const double left,
                                  const double right,
                                  const bool has_left,
                                  const bool has_right) {
    double x_value  = x[i];
    double xp_low   = xp[0];
    double xp_above = xp[dsize - 1];
    double lval     = has_left ? left : fp[0];
    double rval     = has_right ? right : fp[dsize - 1];

    if (x_value > xp_above) {
      out[i] = rval;
    } else if (x_value < xp_low) {
      out[i] = lval;
    } else {
      index_t imin = 0;
      index_t imax = static_cast<index_t>(dsize);
      index_t imid;
      while (imin < imax) {
        imid = static_cast<index_t>((imax + imin) / 2);
        if (x_value >= xp[imid]) {
          imin = imid + 1;
        } else {
          imax = imid;
        }
      }  // biserction search

      index_t j = imin;
      if (j == dsize) {
        out[i] = fp[dsize - 1];
      } else if (x_value == xp[j - 1]) {
        out[i] = fp[j - 1];  // void potential non-finite interpolation
      } else {
        double xp_below     = xp[j - 1];
        double xp_above     = xp[j];
        double weight_above = (x_value - xp_below) / (xp_above - xp_below);
        double weigth_below = 1 - weight_above;
        double x1           = fp[j - 1] * weigth_below;
        double x2           = fp[j] * weight_above;
        out[i]              = x1 + x2;
      }
    }
  }
};

struct interp_period {
  MSHADOW_XINLINE static void Map(index_t i,
                                  double* out,
                                  const double* x,
                                  const double* xp,
                                  const double* fp,
                                  const index_t* idx,
                                  const size_t dsize,
                                  const double period) {
    double x_value = x[i];
    index_t imin   = 0;
    index_t imax   = static_cast<index_t>(dsize);
    index_t imid;
    while (imin < imax) {
      imid = static_cast<index_t>((imax + imin) / 2);
      if (x_value >= xp[idx[imid]]) {
        imin = imid + 1;
      } else {
        imax = imid;
      }
    }  // biserction search

    index_t j = imin;
    double xp_below, xp_above;
    double fp1, fp2;
    if (j == 0) {
      xp_below = xp[idx[dsize - 1]] - period;
      xp_above = xp[idx[0]];
      fp1      = fp[idx[dsize - 1]];
      fp2      = fp[idx[0]];
    } else if (j == dsize) {
      xp_below = xp[idx[dsize - 1]];
      xp_above = xp[idx[0]] + period;
      fp1      = fp[idx[dsize - 1]];
      fp2      = fp[idx[0]];
    } else {
      xp_below = xp[idx[j - 1]];
      xp_above = xp[idx[j]];
      fp1      = fp[idx[j - 1]];
      fp2      = fp[idx[j]];
    }
    double weight_above = (x_value - xp_below) / (xp_above - xp_below);
    double weigth_below = 1 - weight_above;
    double x1           = fp1 * weigth_below;
    double x2           = fp2 * weight_above;
    out[i]              = x1 + x2;
  }
};

inline bool NumpyInterpBackwardShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape>* in_attrs,
                                     std::vector<TShape>* out_attrs) {
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.x_is_scalar ? 3U : 4U);
  CHECK_EQ(out_attrs->size(), param.x_is_scalar ? 2U : 3U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
  if (!param.x_is_scalar) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 2, in_attrs->at(3));
  }
  return shape_is_known(out_attrs->at(0)) && shape_is_known(out_attrs->at(1)) &&
         (param.x_is_scalar || shape_is_known(out_attrs->at(2)));
}

inline bool NumpyInterpBackwardType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_attrs,
                                    std::vector<int>* out_attrs) {
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.x_is_scalar ? 3U : 4U);
  CHECK_EQ(out_attrs->size(), param.x_is_scalar ? 2U : 3U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
  if (!param.x_is_scalar) {
    TYPE_ASSIGN_CHECK(*out_attrs, 2, in_attrs->at(3));
  }
  return out_attrs->at(0) != -1 && out_attrs->at(1) != -1 &&
         (param.x_is_scalar || out_attrs->at(2) != -1);
}

inline void NumpyInterpSetZero(const TBlob& out, const OpReqType req) {
  if (req != kWriteTo && req != kWriteInplace)
    return;
  double* out_ptr = out.dptr<double>();
  for (size_t i = 0; i < out.Size(); ++i) {
    out_ptr[i] = 0.0;
  }
}

inline void NumpyInterpAssign(double* out, const OpReqType req, const size_t i, const double val) {
  if (req == kWriteTo || req == kWriteInplace) {
    out[i] = val;
  } else if (req == kAddTo) {
    out[i] += val;
  }
}

template <typename xpu>
void NumpyInterpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  CHECK(!param.period.has_value()) << "np.interp backward does not support period";
  CHECK_EQ(inputs.size(), param.x_is_scalar ? 3U : 4U);
  CHECK_EQ(outputs.size(), param.x_is_scalar ? 2U : 3U);

  const TBlob& ograd = inputs[0];
  const TBlob& xp    = inputs[1];
  const TBlob& fp    = inputs[2];
  const TBlob& x     = param.x_is_scalar ? inputs[0] : inputs[3];
  const size_t x_size = param.x_is_scalar ? 1 : x.Size();
  CHECK_EQ(ograd.Size(), x_size);
  CHECK_EQ(xp.Size(), fp.Size());
  CHECK_GE(xp.Size(), 1U) << "ValueError: array of sample points is empty";

  if (req[0] != kNullOp)
    NumpyInterpSetZero(outputs[0], req[0]);
  if (req[1] != kNullOp)
    NumpyInterpSetZero(outputs[1], req[1]);
  if (!param.x_is_scalar && req[2] != kNullOp)
    NumpyInterpSetZero(outputs[2], req[2]);

  const double* ograd_ptr = ograd.dptr<double>();
  const double* xp_ptr    = xp.dptr<double>();
  const double* fp_ptr    = fp.dptr<double>();
  const double* x_ptr     = param.x_is_scalar ? nullptr : x.dptr<double>();
  double* xp_grad         = req[0] == kNullOp ? nullptr : outputs[0].dptr<double>();
  double* fp_grad         = req[1] == kNullOp ? nullptr : outputs[1].dptr<double>();
  double* x_grad = param.x_is_scalar || req[2] == kNullOp ? nullptr : outputs[2].dptr<double>();
  const bool has_left     = param.left.has_value();
  const bool has_right    = param.right.has_value();
  const size_t dsize      = xp.Size();

  for (size_t i = 0; i < x_size; ++i) {
    const double x_value = param.x_is_scalar ? param.x_scalar : x_ptr[i];
    const double grad    = ograd_ptr[i];
    if (x_value > xp_ptr[dsize - 1]) {
      if (!has_right && fp_grad) {
        fp_grad[dsize - 1] += grad;
      }
    } else if (x_value < xp_ptr[0]) {
      if (!has_left && fp_grad) {
        fp_grad[0] += grad;
      }
    } else {
      index_t imin = 0;
      index_t imax = static_cast<index_t>(dsize);
      while (imin < imax) {
        index_t imid = static_cast<index_t>((imax + imin) / 2);
        if (x_value >= xp_ptr[imid]) {
          imin = imid + 1;
        } else {
          imax = imid;
        }
      }

      const index_t j = imin;
      if (j == static_cast<index_t>(dsize)) {
        if (fp_grad) {
          fp_grad[dsize - 1] += grad;
        }
      } else if (x_value == xp_ptr[j - 1]) {
        if (fp_grad) {
          fp_grad[j - 1] += grad;
        }
      } else {
        const double xp_below     = xp_ptr[j - 1];
        const double xp_above     = xp_ptr[j];
        const double fp_below     = fp_ptr[j - 1];
        const double fp_above     = fp_ptr[j];
        const double denom        = xp_above - xp_below;
        const double weight_above = (x_value - xp_below) / denom;
        const double weight_below = 1.0 - weight_above;
        if (fp_grad) {
          fp_grad[j - 1] += grad * weight_below;
          fp_grad[j] += grad * weight_above;
        }
        if (x_grad) {
          NumpyInterpAssign(x_grad, req[2], i, grad * (fp_above - fp_below) / denom);
        }
        if (xp_grad) {
          const double scale = grad * (fp_above - fp_below) / (denom * denom);
          xp_grad[j - 1] += scale * (x_value - xp_above);
          xp_grad[j] -= scale * (x_value - xp_below);
        }
      }
    }
  }
}

template <typename xpu, typename OP>
void NumpyInterpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp)
    return;
  using namespace mxnet;
  using namespace mxnet_op;
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_GE(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu>* s                = ctx.get_stream<xpu>();
  const NumpyInterpParam& param = nnvm::get<NumpyInterpParam>(attrs.parsed);
  dmlc::optional<double> left   = param.left;
  dmlc::optional<double> right  = param.right;
  bool x_is_scalar              = param.x_is_scalar;

  TBlob xp           = inputs[0];
  const TBlob& fp    = inputs[1];
  const TBlob& out   = outputs[0];
  bool has_left      = left.has_value() ? true : false;
  bool has_right     = right.has_value() ? true : false;
  bool has_period    = param.period.has_value() ? true : false;
  double left_value  = left.has_value() ? left.value() : 0.0;
  double right_value = right.has_value() ? right.value() : 0.0;
  double period_value = has_period ? param.period.value() : 0.0;

  CHECK_GE(xp.Size(), 1U) << "ValueError: array of sample points is empty";

  TopKParam topk_param = TopKParam();
  topk_param.axis      = dmlc::optional<int>(-1);
  topk_param.is_ascend = true;
  topk_param.k         = 0;
  topk_param.ret_typ   = topk_enum::kReturnIndices;

  size_t topk_temp_size;  // Used by Sort
  size_t topk_workspace_size = TopKWorkspaceSize<xpu, double>(xp, topk_param, &topk_temp_size);
  size_t size_x              = x_is_scalar ? 8 : 0;
  size_t size_norm_x         = x_is_scalar ? 8 : inputs[2].Size() * sizeof(double);
  size_t size_norm_xp        = xp.Size() * sizeof(double);
  size_t size_norm           = has_period ? size_norm_x + size_norm_xp : 0;
  size_t size_idx            = has_period ? xp.Size() * sizeof(index_t) : 0;
  size_t workspace_size      = topk_workspace_size + size_x + size_norm + size_idx;

  Tensor<xpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  char* workspace_curr_ptr = temp_mem.dptr_;

  TBlob x, idx;
  if (x_is_scalar) {
    double x_scalar = param.x_scalar;
    Tensor<cpu, 1, double> host_x(&x_scalar, Shape1(1), ctx.get_stream<cpu>());
    Tensor<xpu, 1, double> device_x(
        reinterpret_cast<double*>(workspace_curr_ptr), Shape1(1), ctx.get_stream<xpu>());
    Copy(device_x, host_x, ctx.get_stream<xpu>());
    x = TBlob(device_x.dptr_, TShape(0, 1), xpu::kDevMask);
    workspace_curr_ptr += 8;
  } else {
    x = inputs[2];
  }  // handle input x is a scalar

  // normalize the input data by periodic boundaries.
  if (has_period) {
    double* norm_xp_ptr;
    double* norm_x_ptr;
    index_t* idx_ptr;
    CHECK_NE(period_value, 0.0) << "period must be a non-zero value";

    norm_xp_ptr = reinterpret_cast<double*>(workspace_curr_ptr);
    norm_x_ptr  = reinterpret_cast<double*>(workspace_curr_ptr + size_norm_xp);
    idx_ptr     = reinterpret_cast<index_t*>(workspace_curr_ptr + size_norm_xp + size_norm_x);

    TBlob norm_x            = TBlob(norm_x_ptr, x.shape_, xpu::kDevMask);
    TBlob norm_xp           = TBlob(norm_xp_ptr, xp.shape_, xpu::kDevMask);
    const OpReqType ReqType = kWriteTo;
    Kernel<op_with_req<OP, ReqType>, xpu>::Launch(
        s, x.Size(), norm_x.dptr<double>(), x.dptr<double>(), period_value);
    Kernel<op_with_req<OP, ReqType>, xpu>::Launch(
        s, xp.Size(), norm_xp.dptr<double>(), xp.dptr<double>(), period_value);

    workspace_curr_ptr += size_x + size_norm + size_idx;
    idx                             = TBlob(idx_ptr, xp.shape_, xpu::kDevMask);
    std::vector<OpReqType> req_TopK = {kWriteTo};
    std::vector<TBlob> ret          = {idx};

    TopKImplwithWorkspace<xpu, double, index_t>(
        ctx.run_ctx, req_TopK, norm_xp, ret, topk_param, workspace_curr_ptr, topk_temp_size, s);
    Kernel<interp_period, xpu>::Launch(s,
                                       norm_x.Size(),
                                       out.dptr<double>(),
                                       norm_x.dptr<double>(),
                                       norm_xp.dptr<double>(),
                                       fp.dptr<double>(),
                                       idx.dptr<index_t>(),
                                       norm_xp.Size(),
                                       period_value);
  } else {
    Kernel<interp, xpu>::Launch(s,
                                x.Size(),
                                out.dptr<double>(),
                                x.dptr<double>(),
                                xp.dptr<double>(),
                                fp.dptr<double>(),
                                xp.Size(),
                                left_value,
                                right_value,
                                has_left,
                                has_right);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INTERP_OP_INL_H_
