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
 * \file np_percentile_op-inl.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <cmath>
#include "../tensor/ordering_op-inl.h"
#include "../tensor/matrix_op-inl.h"
#include "../../common/utils.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"
#include "../../api/operator/op_utils.h"

namespace mxnet {
namespace op {

namespace percentile_enum {
enum PercentileType { kLinear, kLower, kHigher, kMidpoint, kNearest };
}  // namespace percentile_enum

struct NumpyPercentileParam : public dmlc::Parameter<NumpyPercentileParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  int interpolation;
  bool keepdims;
  dmlc::optional<double> q_scalar;
  DMLC_DECLARE_PARAMETER(NumpyPercentileParam) {
    DMLC_DECLARE_FIELD(axis)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe(
            "Axis or axes along which a sum is performed. The default, axis=None, will sum "
            "all of the elements of the input array. If axis is negative it counts from the "
            "last to the first axis.");
    DMLC_DECLARE_FIELD(interpolation)
        .set_default(percentile_enum::kLinear)
        .add_enum("linear", percentile_enum::kLinear)
        .add_enum("lower", percentile_enum::kLower)
        .add_enum("higher", percentile_enum::kHigher)
        .add_enum("midpoint", percentile_enum::kMidpoint)
        .add_enum("nearest", percentile_enum::kNearest)
        .describe(
            "his optional parameter specifies the interpolation method to use when the"
            "desired percentile lies between two data points i < j");
    DMLC_DECLARE_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(q_scalar)
        .set_default(dmlc::optional<double>())
        .describe("inqut q is a scalar");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, keepdims_s, q_scalar_s;
    axis_s << axis;
    keepdims_s << keepdims;
    q_scalar_s << q_scalar;
    (*dict)["axis"]          = axis_s.str();
    (*dict)["interpolation"] = MXNetPercentileType2String(interpolation);
    (*dict)["keepdims"]      = keepdims_s.str();
    (*dict)["q_scalar"]      = q_scalar_s.str();
  }
};

template <int NDim>
struct percentile_take {
  template <typename DType, typename QType, typename OType>
  MSHADOW_XINLINE static void Map(int i,
                                  OType* out,
                                  const QType* q,
                                  const DType* a_sort,
                                  const int interpolation,
                                  mshadow::Shape<NDim> t_shape,
                                  mshadow::Shape<NDim> r_shape) {
    using namespace mshadow;
    using namespace mxnet_op;

    auto r_coord = unravel(i, r_shape);
    size_t q_idx = r_coord[0];

    Shape<NDim> t_coord(t_shape);

    for (int j = 0; j < NDim - 1; ++j) {
      t_coord[j] = r_coord[j + 1];
    }

    float idx        = q[q_idx] * (t_shape[NDim - 1] - 1) / 100.0;
    int integral_idx = -1;
    if (interpolation == percentile_enum::kLower) {
      integral_idx = floor(idx);
    } else if (interpolation == percentile_enum::kHigher) {
      integral_idx = ceil(idx);
    } else if (interpolation == percentile_enum::kMidpoint) {
      idx = (floor(idx) + ceil(idx)) / 2;
    } else if (interpolation == percentile_enum::kNearest) {
      integral_idx = round(idx);
    }

    if (integral_idx >= 0) {
      t_coord[NDim - 1] = integral_idx;
      size_t t_idx      = ravel(t_coord, t_shape);
      out[i]            = static_cast<OType>(a_sort[t_idx]);
    } else {
      int idx_below      = floor(idx);
      int idx_above      = idx_below + 1;
      idx_above          = idx_above > t_shape[NDim - 1] - 1 ? t_shape[NDim - 1] - 1 : idx_above;
      float weight_above = idx - idx_below;
      float weight_below = 1 - weight_above;
      t_coord[NDim - 1]  = idx_below;
      size_t t_idx1      = ravel(t_coord, t_shape);
      size_t t_idx2      = t_idx1 + (idx_above - idx_below);
      OType x1           = static_cast<OType>(a_sort[t_idx1] * weight_below);
      OType x2           = static_cast<OType>(a_sort[t_idx2] * weight_above);
      out[i]             = x1 + x2;
    }
  }
};

inline bool NumpyPercentileBackwardShape(const nnvm::NodeAttrs& attrs,
                                         std::vector<TShape>* in_attrs,
                                         std::vector<TShape>* out_attrs) {
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.q_scalar.has_value() ? 2U : 3U);
  CHECK_EQ(out_attrs->size(), param.q_scalar.has_value() ? 1U : 2U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  if (!param.q_scalar.has_value()) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
  }
  return shape_is_known(out_attrs->at(0)) &&
         (param.q_scalar.has_value() || shape_is_known(out_attrs->at(1)));
}

inline bool NumpyPercentileBackwardType(const nnvm::NodeAttrs& attrs,
                                        std::vector<int>* in_attrs,
                                        std::vector<int>* out_attrs) {
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.q_scalar.has_value() ? 2U : 3U);
  CHECK_EQ(out_attrs->size(), param.q_scalar.has_value() ? 1U : 2U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  if (!param.q_scalar.has_value()) {
    TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(2));
  }
  return out_attrs->at(0) != -1 && (param.q_scalar.has_value() || out_attrs->at(1) != -1);
}

inline void PercentileNormalizeAxes(const TShape& shape,
                                    const dmlc::optional<mxnet::Tuple<int>>& axis,
                                    std::vector<int>* axes,
                                    std::vector<int>* nonred_axes) {
  axes->clear();
  nonred_axes->clear();
  if (!axis.has_value()) {
    for (int i = 0; i < shape.ndim(); ++i) {
      axes->push_back(i);
    }
  } else {
    auto axis_tuple = axis.value();
    for (int i = 0; i < axis_tuple.ndim(); ++i) {
      int ax = axis_tuple[i] < 0 ? axis_tuple[i] + shape.ndim() : axis_tuple[i];
      axes->push_back(ax);
    }
  }
  for (int i = 0; i < shape.ndim(); ++i) {
    bool reduced = false;
    for (int ax : *axes) {
      if (ax == i) {
        reduced = true;
        break;
      }
    }
    if (!reduced) {
      nonred_axes->push_back(i);
    }
  }
}

inline size_t PercentileShapeSizeForAxes(const TShape& shape, const std::vector<int>& axes) {
  size_t size = 1;
  for (int ax : axes) {
    size *= shape[ax];
  }
  return size;
}

inline void PercentileUnravel(size_t idx,
                              const TShape& shape,
                              const std::vector<int>& axes,
                              std::vector<size_t>* coord) {
  coord->resize(axes.size());
  for (int i = static_cast<int>(axes.size()) - 1; i >= 0; --i) {
    const size_t dim = shape[axes[i]];
    (*coord)[i]      = idx % dim;
    idx /= dim;
  }
}

inline size_t PercentileRavelOriginal(const TShape& shape, const std::vector<size_t>& coord) {
  size_t idx = 0;
  for (int i = 0; i < shape.ndim(); ++i) {
    idx = idx * shape[i] + coord[i];
  }
  return idx;
}

template <typename DType, typename GType, typename QType>
void PercentileBackwardImpl(const NumpyPercentileParam& param,
                            const TBlob& ograd,
                            const TBlob& data,
                            const TBlob* q,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const bool has_q_scalar = param.q_scalar.has_value();
  const int interpolation = param.interpolation;
  std::vector<int> red_axes, nonred_axes;
  PercentileNormalizeAxes(data.shape_, param.axis, &red_axes, &nonred_axes);
  const size_t group_count = PercentileShapeSizeForAxes(data.shape_, nonred_axes);
  const size_t red_size    = PercentileShapeSizeForAxes(data.shape_, red_axes);
  const size_t q_size      = has_q_scalar ? 1 : q->Size();
  CHECK_EQ(ograd.Size(), q_size * group_count);

  DType* data_grad = outputs[0].dptr<DType>();
  if (req[0] == kWriteTo || req[0] == kWriteInplace) {
    for (size_t i = 0; i < outputs[0].Size(); ++i) {
      data_grad[i] = DType(0);
    }
  }
  QType* q_grad = nullptr;
  if (!has_q_scalar && req[1] != kNullOp) {
    q_grad = outputs[1].dptr<QType>();
    if (req[1] == kWriteTo || req[1] == kWriteInplace) {
      for (size_t i = 0; i < outputs[1].Size(); ++i) {
        q_grad[i] = QType(0);
      }
    }
  }

  const DType* data_ptr = data.dptr<DType>();
  const GType* og_ptr   = ograd.dptr<GType>();
  const QType* q_ptr    = has_q_scalar ? nullptr : q->dptr<QType>();
  std::vector<size_t> group_coord, red_coord, orig_coord(data.shape_.ndim());
  std::vector<std::pair<double, size_t>> sorted(red_size);

  for (size_t group = 0; group < group_count; ++group) {
    PercentileUnravel(group, data.shape_, nonred_axes, &group_coord);
    for (size_t i = 0; i < nonred_axes.size(); ++i) {
      orig_coord[nonred_axes[i]] = group_coord[i];
    }
    for (size_t r = 0; r < red_size; ++r) {
      PercentileUnravel(r, data.shape_, red_axes, &red_coord);
      for (size_t i = 0; i < red_axes.size(); ++i) {
        orig_coord[red_axes[i]] = red_coord[i];
      }
      const size_t data_idx = PercentileRavelOriginal(data.shape_, orig_coord);
      sorted[r]             = std::make_pair(static_cast<double>(data_ptr[data_idx]), data_idx);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });

    for (size_t qi = 0; qi < q_size; ++qi) {
      const double q_value = has_q_scalar ? param.q_scalar.value() : static_cast<double>(q_ptr[qi]);
      double idx          = q_value * (red_size - 1) / 100.0;
      int integral_idx    = -1;
      if (interpolation == percentile_enum::kLower) {
        integral_idx = std::floor(idx);
      } else if (interpolation == percentile_enum::kHigher) {
        integral_idx = std::ceil(idx);
      } else if (interpolation == percentile_enum::kMidpoint) {
        idx = (std::floor(idx) + std::ceil(idx)) / 2.0;
      } else if (interpolation == percentile_enum::kNearest) {
        integral_idx = std::round(idx);
      }

      const double grad = static_cast<double>(og_ptr[qi * group_count + group]);
      if (integral_idx >= 0) {
        data_grad[sorted[integral_idx].second] += static_cast<DType>(grad);
      } else {
        const int idx_below = std::floor(idx);
        int idx_above       = idx_below + 1;
        idx_above           = idx_above > static_cast<int>(red_size) - 1 ?
                                  static_cast<int>(red_size) - 1 :
                                  idx_above;
        const double weight_above = idx - idx_below;
        const double weight_below = 1.0 - weight_above;
        data_grad[sorted[idx_below].second] += static_cast<DType>(grad * weight_below);
        data_grad[sorted[idx_above].second] += static_cast<DType>(grad * weight_above);
        if (q_grad && interpolation == percentile_enum::kLinear) {
          const double q_scale =
              (sorted[idx_above].first - sorted[idx_below].first) * (red_size - 1) / 100.0;
          q_grad[qi] += static_cast<QType>(grad * q_scale);
        }
      }
    }
  }
}

template <typename xpu>
void NumpyPercentileBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  const bool has_q_scalar          = param.q_scalar.has_value();
  CHECK_EQ(inputs.size(), has_q_scalar ? 2U : 3U);
  CHECK_EQ(outputs.size(), has_q_scalar ? 1U : 2U);
  if (req[0] == kNullOp && (has_q_scalar || req[1] == kNullOp))
    return;
  const TBlob& ograd = inputs[0];
  const TBlob& data  = inputs[1];
  const TBlob* q     = has_q_scalar ? nullptr : &inputs[2];
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    MSHADOW_SGL_DBL_TYPE_SWITCH(ograd.type_flag_, GType, {
      if (has_q_scalar) {
        PercentileBackwardImpl<DType, GType, double>(param, ograd, data, nullptr, req, outputs);
      } else {
        MSHADOW_SGL_DBL_TYPE_SWITCH(q->type_flag_, QType, {
          PercentileBackwardImpl<DType, GType, QType>(param, ograd, data, q, req, outputs);
        });
      }
    });
  });
}

template <typename QType, typename xpu>
bool CheckInvalidInput(mshadow::Stream<xpu>* s,
                       const QType* data,
                       const size_t& data_size,
                       char* is_valid_ptr);

template <typename xpu>
void NumpyPercentileForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp)
    return;
  using namespace mxnet;
  using namespace mxnet_op;
  CHECK_GE(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu>* s                         = ctx.get_stream<xpu>();
  const TBlob& data                      = inputs[0];
  const TBlob& out                       = outputs[0];
  const NumpyPercentileParam& param      = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  const int interpolation                = param.interpolation;
  dmlc::optional<mxnet::Tuple<int>> axis = param.axis;
  const bool has_q_scalar                = param.q_scalar.has_value() ? true : false;
  const double q_scalar_value            = has_q_scalar ? param.q_scalar.value() : 0.0;

  auto small = NumpyReduceAxesShapeImpl(data.shape_, axis, false);

  TShape r_shape;
  r_shape    = TShape(small.ndim() + 1, 1);
  r_shape[0] = has_q_scalar ? 1 : inputs[1].Size();
  for (int i = 1; i < r_shape.ndim(); ++i) {
    r_shape[i] = small[i - 1];
  }
  // Origin axes
  TShape axes;
  if (!axis.has_value()) {
    axes = TShape(data.shape_.ndim(), 1);
    for (int i = 0; i < data.shape_.ndim(); ++i) {
      axes[i] = i;
    }
  } else {
    auto axis_tuple = axis.value();
    axes            = TShape(axis_tuple.ndim(), 1);
    for (int i = 0; i < axis_tuple.ndim(); ++i) {
      if (axis_tuple[i] < 0) {
        axes[i] = axis_tuple[i] + data.shape_.ndim();
      } else {
        axes[i] = axis_tuple[i];
      }
    }
  }
  // Transpose the axes
  TShape t_axes(data.shape_.ndim(), 1);
  int j = 0;
  for (int i = 0; i < t_axes.ndim(); ++i) {
    bool red = false;
    for (int k = 0; k < axes.ndim(); ++k) {
      if (axes[k] == i) {
        red = true;
      }
    }
    if (!red) {
      t_axes[j] = i;
      j++;
    }
  }
  for (int jj = j; jj < t_axes.ndim(); ++jj) {
    t_axes[jj] = axes[jj - j];
  }
  // Transpose Shape with reduced dims at dim [-1]
  TShape t_shape(small.ndim() + 1, 1);
  for (int i = 0; i < small.ndim(); ++i) {
    t_shape[i] = small[i];
  }
  size_t red_size = 1;
  for (int i = 0; i < axes.ndim(); ++i) {
    red_size *= data.shape_[axes[i]];
  }
  t_shape[t_shape.ndim() - 1] = red_size;
  // Transpose Shape extension
  TShape t_shape_ex(data.shape_.ndim(), 1);
  for (int i = 0; i < data.shape_.ndim(); ++i) {
    t_shape_ex[i] = data.shape_[t_axes[i]];
  }
  TopKParam topk_param = TopKParam();
  topk_param.axis      = dmlc::optional<int>(-1);
  topk_param.is_ascend = true;
  topk_param.k         = 0;
  topk_param.ret_typ   = topk_enum::kReturnValue;

  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    size_t temp_size;  // Used by Sort
    size_t topk_workspace_size = TopKWorkspaceSize<xpu, DType>(data, topk_param, &temp_size);

    size_t temp_data_size = data.Size() * sizeof(DType);
    size_t idx_size       = data.Size() * sizeof(index_t);
    size_t temp_mem_size  = 2 * temp_data_size + idx_size;
    size_t workspace_size = topk_workspace_size * 2 + temp_mem_size + 16;

    Tensor<xpu, 1, char> temp_mem =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

    char* workspace_curr_ptr = temp_mem.dptr_;
    DType *trans_ptr, *sort_ptr;
    index_t* idx_ptr;
    TBlob percentile;
    double q;

    if (has_q_scalar) {
      q = q_scalar_value;
      Tensor<cpu, 1, double> host_q(&q, Shape1(1), ctx.get_stream<cpu>());
      Tensor<xpu, 1, double> device_q(
          reinterpret_cast<double*>(workspace_curr_ptr), Shape1(1), ctx.get_stream<xpu>());
      mshadow::Copy(device_q, host_q, ctx.get_stream<xpu>());
      percentile = TBlob(device_q.dptr_, TShape(0, 1), xpu::kDevMask);
      workspace_curr_ptr += 8;
    } else {
      percentile = inputs[1];
    }  // handle input q is a scalar

    char* is_valid_ptr = reinterpret_cast<char*>(workspace_curr_ptr);
    MSHADOW_TYPE_SWITCH(percentile.type_flag_, QType, {
      bool is_valid = CheckInvalidInput<QType, xpu>(
          s, percentile.dptr<QType>(), percentile.Size(), is_valid_ptr);
      CHECK(is_valid) << "ValueError: percentile exceeds the valid range";
    })  // check the invalid percentile
    workspace_curr_ptr += 8;

    if (sizeof(DType) >= sizeof(index_t)) {
      trans_ptr = reinterpret_cast<DType*>(workspace_curr_ptr);
      sort_ptr  = reinterpret_cast<DType*>(workspace_curr_ptr + temp_data_size);
      idx_ptr   = reinterpret_cast<index_t*>(workspace_curr_ptr + 2 * temp_data_size);
    } else {
      idx_ptr   = reinterpret_cast<index_t*>(workspace_curr_ptr);
      trans_ptr = reinterpret_cast<DType*>(workspace_curr_ptr + idx_size);
      sort_ptr  = reinterpret_cast<DType*>(workspace_curr_ptr + temp_data_size + idx_size);
    }
    workspace_curr_ptr += 2 * temp_data_size + idx_size;

    TBlob a_trans = TBlob(trans_ptr, t_shape_ex, xpu::kDevMask);
    TransposeImpl<xpu>(ctx.run_ctx, data, a_trans, t_axes);
    TBlob a_sort                    = TBlob(sort_ptr, t_shape, xpu::kDevMask);
    TBlob a_idx                     = TBlob(idx_ptr, t_shape, xpu::kDevMask);
    std::vector<OpReqType> req_TopK = {kWriteTo, kNullOp};
    TBlob src                       = a_trans.reshape(t_shape);
    std::vector<TBlob> ret          = {a_sort, a_idx};

    TopKImplwithWorkspace<xpu, DType, index_t>(
        ctx.run_ctx, req_TopK, src, ret, topk_param, workspace_curr_ptr, temp_size, s);
    MSHADOW_TYPE_SWITCH(percentile.type_flag_,
                        QType,
                        {MSHADOW_SGL_DBL_TYPE_SWITCH(
                            out.type_flag_, OType, {MXNET_NDIM_SWITCH(small.ndim() + 1, NDim, {
                              Kernel<percentile_take<NDim>, xpu>::Launch(s,
                                                                         r_shape.Size(),
                                                                         out.dptr<OType>(),
                                                                         percentile.dptr<QType>(),
                                                                         a_sort.dptr<DType>(),
                                                                         interpolation,
                                                                         t_shape.get<NDim>(),
                                                                         r_shape.get<NDim>());
                            })})})
  })
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_
