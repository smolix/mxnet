/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal.cc
 * \brief
 * \author Xizhou Zhu, Kan Wu
 */

#include "./multi_proposal-inl.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

inline int64_t CheckedMul(int64_t lhs, int64_t rhs, const char* what) {
  CHECK_GE(lhs, 0);
  CHECK_GE(rhs, 0);
  CHECK(rhs == 0 || lhs <= std::numeric_limits<int64_t>::max() / rhs)
      << what << " exceeds int64_t range: " << lhs << " * " << rhs;
  return lhs * rhs;
}

inline int64_t CheckedAdd(int64_t lhs, int64_t rhs, const char* what) {
  CHECK_GE(lhs, 0);
  CHECK_GE(rhs, 0);
  CHECK_LE(lhs, std::numeric_limits<int64_t>::max() - rhs)
      << what << " exceeds int64_t range: " << lhs << " + " << rhs;
  return lhs + rhs;
}

inline int64_t CheckedMul3(int64_t a, int64_t b, int64_t c, const char* what) {
  return CheckedMul(CheckedMul(a, b, what), c, what);
}

inline mshadow::index_t CheckedIndexT(int64_t value, const char* what) {
  CHECK_GE(value, 0);
  CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<mshadow::index_t>::max()))
      << what << " exceeds mshadow index_t range: " << value;
  return static_cast<mshadow::index_t>(value);
}

inline mshadow::index_t CheckedWorkspaceShape(int64_t elements, const char* what) {
  const uint64_t max_elements =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(mxnet::real_t));
  CHECK_GE(elements, 0);
  CHECK_LE(static_cast<uint64_t>(elements), max_elements)
      << what << " exceeds size_t allocation range: " << elements;
  return CheckedIndexT(elements, what);
}

}  // namespace

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

// bbox prediction and clip to the image borders
inline void BBoxTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                             const mshadow::Tensor<cpu, 3>& deltas,
                             const float im_height,
                             const float im_width,
                             const int real_height,
                             const int real_width,
                             mshadow::Tensor<cpu, 2>* out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  const index_t anchors = deltas.size(0) / 4;
  const index_t heights = deltas.size(1);
  const index_t widths  = deltas.size(2);
  const index_t anchor_stride =
      CheckedIndexT(CheckedMul(static_cast<int64_t>(widths),
                               static_cast<int64_t>(anchors),
                               "multi_proposal bbox transform anchor stride"),
                    "multi_proposal bbox transform anchor stride");
  const index_t count = CheckedIndexT(CheckedMul(static_cast<int64_t>(heights),
                                                 static_cast<int64_t>(anchor_stride),
                                                 "multi_proposal bbox transform size"),
                                      "multi_proposal bbox transform size");

#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_index = 0;
       raw_index < static_cast<mshadow::openmp_index_t>(count);
       ++raw_index) {
    // index_t index = h * (widths * anchors) + w * (anchors) + a;
    const index_t index = static_cast<index_t>(raw_index);
    const index_t a     = index % anchors;
    const index_t w     = (index / anchors) % widths;
    const index_t h     = index / anchor_stride;

    float width  = boxes[index][2] - boxes[index][0] + 1.0;
    float height = boxes[index][3] - boxes[index][1] + 1.0;
    float ctr_x  = boxes[index][0] + 0.5 * (width - 1.0);
    float ctr_y  = boxes[index][1] + 0.5 * (height - 1.0);

    float dx = deltas[a * 4 + 0][h][w];
    float dy = deltas[a * 4 + 1][h][w];
    float dw = deltas[a * 4 + 2][h][w];
    float dh = deltas[a * 4 + 3][h][w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w     = std::exp(dw) * width;
    float pred_h     = std::exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

    (*out_pred_boxes)[index][0] = pred_x1;
    (*out_pred_boxes)[index][1] = pred_y1;
    (*out_pred_boxes)[index][2] = pred_x2;
    (*out_pred_boxes)[index][3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      (*out_pred_boxes)[index][4] = -1.0;
    }
  }
}

// iou prediction and clip to the image border
inline void IoUTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                            const mshadow::Tensor<cpu, 3>& deltas,
                            const float im_height,
                            const float im_width,
                            const int real_height,
                            const int real_width,
                            mshadow::Tensor<cpu, 2>* out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  const index_t anchors = deltas.size(0) / 4;
  const index_t heights = deltas.size(1);
  const index_t widths  = deltas.size(2);
  const index_t anchor_stride =
      CheckedIndexT(CheckedMul(static_cast<int64_t>(widths),
                               static_cast<int64_t>(anchors),
                               "multi_proposal iou transform anchor stride"),
                    "multi_proposal iou transform anchor stride");
  const index_t count = CheckedIndexT(CheckedMul(static_cast<int64_t>(heights),
                                                 static_cast<int64_t>(anchor_stride),
                                                 "multi_proposal iou transform size"),
                                      "multi_proposal iou transform size");

#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_index = 0;
       raw_index < static_cast<mshadow::openmp_index_t>(count);
       ++raw_index) {
    // index_t index = h * (widths * anchors) + w * (anchors) + a;
    const index_t index = static_cast<index_t>(raw_index);
    const index_t a     = index % anchors;
    const index_t w     = (index / anchors) % widths;
    const index_t h     = index / anchor_stride;

    float x1 = boxes[index][0];
    float y1 = boxes[index][1];
    float x2 = boxes[index][2];
    float y2 = boxes[index][3];

    float dx1 = deltas[a * 4 + 0][h][w];
    float dy1 = deltas[a * 4 + 1][h][w];
    float dx2 = deltas[a * 4 + 2][h][w];
    float dy2 = deltas[a * 4 + 3][h][w];

    float pred_x1 = x1 + dx1;
    float pred_y1 = y1 + dy1;
    float pred_x2 = x2 + dx2;
    float pred_y2 = y2 + dy2;

    pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

    (*out_pred_boxes)[index][0] = pred_x1;
    (*out_pred_boxes)[index][1] = pred_y1;
    (*out_pred_boxes)[index][2] = pred_x2;
    (*out_pred_boxes)[index][3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      (*out_pred_boxes)[index][4] = -1.0f;
    }
  }
}

// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(mshadow::Tensor<cpu, 2>* dets, const float min_size) {
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_i = 0;
       raw_i < static_cast<mshadow::openmp_index_t>(dets->size(0));
       ++raw_i) {
    const index_t i = static_cast<index_t>(raw_i);
    float iw        = (*dets)[i][2] - (*dets)[i][0] + 1.0f;
    float ih        = (*dets)[i][3] - (*dets)[i][1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      (*dets)[i][0] -= min_size / 2;
      (*dets)[i][1] -= min_size / 2;
      (*dets)[i][2] += min_size / 2;
      (*dets)[i][3] += min_size / 2;
      (*dets)[i][4] = -1.0f;
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace utils {

struct ReverseArgsortCompl {
  const float* val_;
  explicit ReverseArgsortCompl(float* val) : val_(val) {}
  bool operator()(float i, float j) {
    return (val_[static_cast<index_t>(i)] > val_[static_cast<index_t>(j)]);
  }
};

// copy score and init order
inline void CopyScore(const mshadow::Tensor<cpu, 2>& dets,
                      mshadow::Tensor<cpu, 1>* score,
                      mshadow::Tensor<cpu, 1>* order) {
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_i = 0;
       raw_i < static_cast<mshadow::openmp_index_t>(dets.size(0));
       ++raw_i) {
    const index_t i = static_cast<index_t>(raw_i);
    (*score)[i]     = dets[i][4];
    (*order)[i]     = i;
  }
}

// sort order array according to score
inline void ReverseArgsort(const mshadow::Tensor<cpu, 1>& score, mshadow::Tensor<cpu, 1>* order) {
  ReverseArgsortCompl cmpl(score.dptr_);
  std::stable_sort(order->dptr_, order->dptr_ + score.size(0), cmpl);
}

// reorder proposals according to order and keep the pre_nms_top_n proposals
// dets.size(0) == pre_nms_top_n
inline void ReorderProposals(const mshadow::Tensor<cpu, 2>& prev_dets,
                             const mshadow::Tensor<cpu, 1>& order,
                             const index_t pre_nms_top_n,
                             mshadow::Tensor<cpu, 2>* dets) {
  CHECK_EQ(dets->size(0), pre_nms_top_n);
  const index_t dets_size0 = dets->size(0);
  const index_t dets_size1 = dets->size(1);
  const index_t dets_size  = CheckedIndexT(CheckedMul(static_cast<int64_t>(dets_size0),
                                                     static_cast<int64_t>(dets_size1),
                                                     "multi_proposal reorder size"),
                                          "multi_proposal reorder size");
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_k = 0; raw_k < static_cast<mshadow::openmp_index_t>(dets_size);
       ++raw_k) {
    const index_t k     = static_cast<index_t>(raw_k);
    const index_t i     = k / dets_size1;
    const index_t j     = k % dets_size1;
    const index_t index = order[i];
    (*dets)[i][j]       = prev_dets[index][j];
  }
}

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2>& dets,
                                  const float thresh,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 1>* area,
                                  mshadow::Tensor<cpu, 1>* suppressed,
                                  mshadow::Tensor<cpu, 1>* keep,
                                  index_t* out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(area->CheckContiguous(), true);
  CHECK_EQ(suppressed->CheckContiguous(), true);
  CHECK_EQ(keep->CheckContiguous(), true);
// calculate area
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (mshadow::openmp_index_t raw_i = 0;
       raw_i < static_cast<mshadow::openmp_index_t>(dets.size(0));
       ++raw_i) {
    const index_t i = static_cast<index_t>(raw_i);
    (*area)[i]      = (dets[i][2] - dets[i][0] + 1) * (dets[i][3] - dets[i][1] + 1);
  }

  // calculate nms
  *out_size = 0;
  for (index_t i = 0; i < dets.size(0) && (*out_size) < post_nms_top_n; ++i) {
    float ix1   = dets[i][0];
    float iy1   = dets[i][1];
    float ix2   = dets[i][2];
    float iy2   = dets[i][3];
    float iarea = (*area)[i];

    if ((*suppressed)[i] > 0.0f) {
      continue;
    }

    (*keep)[(*out_size)++] = i;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (mshadow::openmp_index_t raw_j = static_cast<mshadow::openmp_index_t>(i + 1);
         raw_j < static_cast<mshadow::openmp_index_t>(dets.size(0));
         ++raw_j) {
      const index_t j = static_cast<index_t>(raw_j);
      if ((*suppressed)[j] > 0.0f) {
        continue;
      }
      float xx1   = std::max(ix1, dets[j][0]);
      float yy1   = std::max(iy1, dets[j][1]);
      float xx2   = std::min(ix2, dets[j][2]);
      float yy2   = std::min(iy2, dets[j][3]);
      float w     = std::max(0.0f, xx2 - xx1 + 1.0f);
      float h     = std::max(0.0f, yy2 - yy1 + 1.0f);
      float inter = w * h;
      float ovr   = inter / (iarea + (*area)[j] - inter);
      if (ovr > thresh) {
        (*suppressed)[j] = 1.0f;
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

namespace mxnet {
namespace op {

template <typename xpu>
class MultiProposalOp : public Operator {
 public:
  explicit MultiProposalOp(MultiProposalParam param) {
    this->param_ = param;
  }

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data,
               const std::vector<TBlob>& aux_states) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);

    Stream<xpu>* s = ctx.get_stream<xpu>();

    const mxnet::TShape& cls_shape = in_data[proposal::kClsProb].shape_;
    const index_t num_images       = CheckedIndexT(cls_shape[0], "multi_proposal batch size");
    const index_t num_score_channels =
        CheckedIndexT(cls_shape[1], "multi_proposal score channel count");
    const index_t num_anchors     = num_score_channels / 2;
    const index_t height          = CheckedIndexT(cls_shape[2], "multi_proposal height");
    const index_t width           = CheckedIndexT(cls_shape[3], "multi_proposal width");
    const int64_t count_anchors64 = CheckedMul3(static_cast<int64_t>(num_anchors),
                                                static_cast<int64_t>(height),
                                                static_cast<int64_t>(width),
                                                "multi_proposal anchor count");
    const index_t count_anchors   = CheckedIndexT(count_anchors64, "multi_proposal anchor count");
    const index_t anchor_stride   = CheckedIndexT(CheckedMul(static_cast<int64_t>(width),
                                                           static_cast<int64_t>(num_anchors),
                                                           "multi_proposal anchor stride"),
                                                "multi_proposal anchor stride");
    const int64_t num_images64    = static_cast<int64_t>(num_images);
    const int64_t image_anchor_count64 =
        CheckedMul(num_images64, count_anchors64, "multi_proposal image anchor count");
    const index_t image_anchor_count =
        CheckedIndexT(image_anchor_count64, "multi_proposal image anchor count");

    Tensor<cpu, 4> scores      = in_data[proposal::kClsProb].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> bbox_deltas = in_data[proposal::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> im_info     = in_data[proposal::kImInfo].get<cpu, 2, real_t>(s);

    Tensor<cpu, 2> out       = out_data[proposal::kOut].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> out_score = out_data[proposal::kScore].get<cpu, 2, real_t>(s);

    int64_t rpn_pre_nms_top_n64 =
        (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count_anchors64;
    rpn_pre_nms_top_n64 = std::min(rpn_pre_nms_top_n64, count_anchors64);
    const index_t rpn_pre_nms_top_n =
        CheckedIndexT(rpn_pre_nms_top_n64, "multi_proposal pre-NMS top_n");
    const int64_t rpn_post_nms_top_n64 =
        std::min(static_cast<int64_t>(param_.rpn_post_nms_top_n), rpn_pre_nms_top_n64);
    const index_t rpn_post_nms_top_n =
        CheckedIndexT(rpn_post_nms_top_n64, "multi_proposal post-NMS top_n");
    const int64_t output_stride64 = static_cast<int64_t>(param_.rpn_post_nms_top_n);
    const index_t output_stride   = CheckedIndexT(output_stride64, "multi_proposal output size");
    CheckedIndexT(CheckedMul(num_images64, output_stride64, "multi_proposal output size"),
                  "multi_proposal output size");

    const int64_t workspace_proposals_size64 =
        CheckedMul(image_anchor_count64, 5, "multi_proposal workspace size");
    const int64_t workspace_pre_nms_size64 =
        CheckedMul(image_anchor_count64, 2, "multi_proposal workspace size");
    const int64_t image_pre_nms_count64 =
        CheckedMul(num_images64, rpn_pre_nms_top_n64, "multi_proposal workspace size");
    const int64_t workspace_ordered_proposals_size64 =
        CheckedMul(image_pre_nms_count64, 5, "multi_proposal workspace size");
    const int64_t workspace_nms_size64 =
        CheckedMul(image_pre_nms_count64, 3, "multi_proposal workspace size");
    const int64_t workspace_size64 =
        CheckedAdd(CheckedAdd(CheckedAdd(workspace_proposals_size64,
                                         workspace_pre_nms_size64,
                                         "multi_proposal workspace size"),
                              workspace_ordered_proposals_size64,
                              "multi_proposal workspace size"),
                   workspace_nms_size64,
                   "multi_proposal workspace size");
    const index_t workspace_proposals_size =
        CheckedIndexT(workspace_proposals_size64, "multi_proposal workspace size");
    const index_t workspace_pre_nms_size =
        CheckedIndexT(workspace_pre_nms_size64, "multi_proposal workspace size");
    const index_t workspace_ordered_proposals_size =
        CheckedIndexT(workspace_ordered_proposals_size64, "multi_proposal workspace size");
    const index_t workspace_nms_size =
        CheckedIndexT(workspace_nms_size64, "multi_proposal workspace size");
    const index_t workspace_size =
        CheckedWorkspaceShape(workspace_size64, "multi_proposal workspace size");

    Tensor<cpu, 1> workspace =
        ctx.requested[proposal::kTempResource].get_space<cpu>(Shape1(workspace_size), s);
    index_t start = 0;
    Tensor<cpu, 3> workspace_proposals(workspace.dptr_ + start,
                                       Shape3(num_images, count_anchors, 5));
    start += workspace_proposals_size;
    Tensor<cpu, 3> workspace_pre_nms(workspace.dptr_ + start, Shape3(num_images, 2, count_anchors));
    start += workspace_pre_nms_size;
    Tensor<cpu, 3> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape3(num_images, rpn_pre_nms_top_n, 5));
    start += workspace_ordered_proposals_size;
    Tensor<cpu, 3> workspace_nms(workspace.dptr_ + start, Shape3(num_images, 3, rpn_pre_nms_top_n));
    start += workspace_nms_size;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;

    // Generate anchors
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.ndim() * param_.scales.ndim());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor, param_.ratios, param_.scales, &anchors);
    std::memcpy(workspace_proposals.dptr_, &anchors[0], sizeof(float) * anchors.size());

    Tensor<cpu, 2> workspace_proposals0 = workspace_proposals[0];
// Enumerate all shifted anchors
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (mshadow::openmp_index_t raw_index = 0;
         raw_index < static_cast<mshadow::openmp_index_t>(count_anchors);
         ++raw_index) {
      // index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
      const index_t index            = static_cast<index_t>(raw_index);
      const index_t i                = index % num_anchors;
      const index_t k                = (index / num_anchors) % width;
      const index_t j                = index / anchor_stride;
      const float shift_x            = static_cast<float>(k) * param_.feature_stride;
      const float shift_y            = static_cast<float>(j) * param_.feature_stride;
      workspace_proposals0[index][0] = workspace_proposals0[i][0] + shift_x;
      workspace_proposals0[index][1] = workspace_proposals0[i][1] + shift_y;
      workspace_proposals0[index][2] = workspace_proposals0[i][2] + shift_x;
      workspace_proposals0[index][3] = workspace_proposals0[i][3] + shift_y;
      workspace_proposals0[index][4] = scores[0][i + num_anchors][j][k];
    }

// Copy shifted anchors to other images
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (mshadow::openmp_index_t raw_t = static_cast<mshadow::openmp_index_t>(count_anchors);
         raw_t < static_cast<mshadow::openmp_index_t>(image_anchor_count);
         ++raw_t) {
      const index_t t     = static_cast<index_t>(raw_t);
      const index_t b     = t / count_anchors;
      const index_t index = t % count_anchors;
      const index_t i     = index % num_anchors;
      const index_t k     = (index / num_anchors) % width;
      const index_t j     = index / anchor_stride;
      for (index_t w = 0; w < 4; ++w) {
        workspace_proposals[b][index][w] = workspace_proposals[0][index][w];
      }
      workspace_proposals[b][index][4] = scores[b][i + num_anchors][j][k];
    }

// Assign Foreground Scores for each anchor
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (mshadow::openmp_index_t raw_b = 0;
         raw_b < static_cast<mshadow::openmp_index_t>(num_images);
         ++raw_b) {
      const index_t b = static_cast<index_t>(raw_b);
      // prevent padded predictions
      int real_height = static_cast<int>(im_info[b][0] / param_.feature_stride);
      int real_width  = static_cast<int>(im_info[b][1] / param_.feature_stride);
      CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
      CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

      Tensor<cpu, 2> workspace_proposals_i         = workspace_proposals[b];
      Tensor<cpu, 2> workspace_pre_nms_i           = workspace_pre_nms[b];
      Tensor<cpu, 2> workspace_ordered_proposals_i = workspace_ordered_proposals[b];
      Tensor<cpu, 2> workspace_nms_i               = workspace_nms[b];

      if (param_.iou_loss) {
        utils::IoUTransformInv(workspace_proposals_i,
                               bbox_deltas[b],
                               im_info[b][0],
                               im_info[b][1],
                               real_height,
                               real_width,
                               &(workspace_proposals_i));
      } else {
        utils::BBoxTransformInv(workspace_proposals_i,
                                bbox_deltas[b],
                                im_info[b][0],
                                im_info[b][1],
                                real_height,
                                real_width,
                                &(workspace_proposals_i));
      }
      utils::FilterBox(&workspace_proposals_i, param_.rpn_min_size * im_info[b][2]);

      Tensor<cpu, 1> score = workspace_pre_nms_i[0];
      Tensor<cpu, 1> order = workspace_pre_nms_i[1];

      utils::CopyScore(workspace_proposals_i, &score, &order);
      utils::ReverseArgsort(score, &order);
      utils::ReorderProposals(
          workspace_proposals_i, order, rpn_pre_nms_top_n, &workspace_ordered_proposals_i);
      index_t out_size          = 0;
      Tensor<cpu, 1> area       = workspace_nms_i[0];
      Tensor<cpu, 1> suppressed = workspace_nms_i[1];
      Tensor<cpu, 1> keep       = workspace_nms_i[2];
      suppressed                = 0;  // surprised!

      utils::NonMaximumSuppression(workspace_ordered_proposals_i,
                                   param_.threshold,
                                   rpn_post_nms_top_n,
                                   &area,
                                   &suppressed,
                                   &keep,
                                   &out_size);

// fill in output rois and output scores
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (mshadow::openmp_index_t raw_i = 0;
           raw_i < static_cast<mshadow::openmp_index_t>(output_stride);
           ++raw_i) {
        const index_t i         = static_cast<index_t>(raw_i);
        const index_t out_index = b * output_stride + i;
        out[out_index][0]       = b;
        if (i < out_size) {
          index_t index = keep[i];
          for (index_t j = 0; j < 4; ++j) {
            out[out_index][j + 1] = workspace_ordered_proposals_i[index][j];
          }
          out_score[out_index][0] = workspace_ordered_proposals_i[index][4];
        } else {
          index_t index = keep[i % out_size];
          for (index_t j = 0; j < 4; ++j) {
            out[out_index][j + 1] = workspace_ordered_proposals_i[index][j];
          }
          out_score[out_index][0] = workspace_ordered_proposals_i[index][4];
        }
      }
    }
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<TBlob>& out_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad,
                const std::vector<TBlob>& aux_states) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu>* s         = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox   = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo   = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  MultiProposalParam param_;
};  // class MultiProposalOp

template <>
Operator* CreateOp<cpu>(MultiProposalParam param) {
  return new MultiProposalOp<cpu>(param);
}

Operator* MultiProposalProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiProposalParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiProposal, MultiProposalProp)
    .describe("Generate region proposals via RPN")
    .add_argument("cls_prob", "NDArray-or-Symbol", "Score of how likely proposal is object.")
    .add_argument("bbox_pred",
                  "NDArray-or-Symbol",
                  "BBox Predicted deltas from anchors for proposals")
    .add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
    .add_arguments(MultiProposalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
