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
 * \file dnnl_reshape.cc
 * \brief Implement reshape operator via DNNL reorder primitive
 * \author Tao Lv
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/tensor/elemwise_unary_op.h"
#include "dnnl_base-inl.h"
#include "dnnl_reshape-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v3/dev_guide_reorder.html
bool SupportDNNLReshape(const NDArray& input) {
  return SupportDNNL(input) && input.IsDNNLData() && input.shape().Size() != 1;
}

DNNLReshapeFwd::DNNLReshapeFwd(const OpReqType& req, const NDArray& input, const NDArray& output) {
  const auto engine = CpuEngine::Get()->get_engine();
  auto in_mem       = input.GetDNNLData();

  // Create temp memory.  The target descriptor follows input.shape() which
  // is the post-reshape view of the data.  Note that `in_mem` may carry a
  // *different* ndim than input.shape(): an upstream in-place op (e.g.
  // reshape_like) can mutate the NDArray's metadata shape without
  // invalidating the DNNL chunk, leaving it in the producing op's layout.
  // oneDNN v3 reorder requires src and dst to have the same ndims
  // (reorder.cpp:90), so when the ndims disagree we take a 2-step path
  // through an intermediate same-ndim default buffer and reinterpret it as
  // the post-reshape shape afterwards (apache/mxnet#21199).
  auto temp_dims = dnnl::memory::dims(input.shape().begin(), input.shape().end());
  auto temp_type = static_cast<dnnl::memory::data_type>(get_dnnl_type(input.dtype()));
  auto temp_fmt  = static_cast<dnnl::memory::format_tag>(GetDefaultFormat(input.shape().ndim()));
  auto temp_desc = dnnl::memory::desc(temp_dims, temp_type, temp_fmt);

  out_ = std::make_shared<dnnl::memory>(temp_desc, engine, nullptr);

  const auto in_desc   = in_mem->get_desc();
  const int  in_ndim   = in_desc.get_ndims();
  const int  out_ndim  = input.shape().ndim();
  const bool ndim_diff = (in_ndim != out_ndim);

  // Build a default-format desc in the *source's* ndim space (used by the
  // 2-step paths below).  When ndim_diff is false this matches temp_desc.
  dnnl::memory::desc in_temp_desc;
  if (ndim_diff) {
    const auto in_dims_v = in_desc.get_dims();
    dnnl::memory::dims in_temp_dims(in_dims_v.begin(), in_dims_v.end());
    auto in_temp_fmt =
        static_cast<dnnl::memory::format_tag>(GetDefaultFormat(in_ndim));
    in_temp_desc = dnnl::memory::desc(in_temp_dims, temp_type, in_temp_fmt);
  } else {
    in_temp_desc = temp_desc;
  }

  if (req == kWriteInplace) {
    // If the input has DNNL internal layout, we need reorder it to a temporal buffer with
    // default layout and copy from the temporal buffer back to output buffer which has the same
    // address with input buffer.
    // If the input has default layout *and* the metadata ndim matches the
    // dnnl chunk's ndim, nothing needs to be done.
    if (input.IsDNNLData() || ndim_diff) {
      temp_ = std::make_shared<dnnl::memory>(in_temp_desc, engine, nullptr);
      prims_.push_back(dnnl::reorder(*in_mem, *temp_));  // reorder to default
      if (ndim_diff) {
        // Reinterpret the same buffer with the post-reshape ndim/dims so the
        // copy-back step is a same-ndim default->default reorder (memcpy).
        temp_reshaped_ = std::make_shared<dnnl::memory>(temp_desc, engine, nullptr);
        prims_.push_back(dnnl::reorder(*temp_reshaped_, *out_));
      } else {
        prims_.push_back(dnnl::reorder(*temp_, *out_));  // copy back
      }
    }
  } else if (req == kWriteTo) {
    if (ndim_diff) {
      // Same 2-step path as above for the non-inplace case.
      temp_ = std::make_shared<dnnl::memory>(in_temp_desc, engine, nullptr);
      prims_.push_back(dnnl::reorder(*in_mem, *temp_));
      temp_reshaped_ = std::make_shared<dnnl::memory>(temp_desc, engine, nullptr);
      prims_.push_back(dnnl::reorder(*temp_reshaped_, *out_));
    } else {
      prims_.push_back(dnnl::reorder(*in_mem, *out_));
    }
  } else {
    LOG(FATAL) << "not supported req type: " << req;
  }
}

int DNNLReshapeFwd::GetWorkspaceSize() {
  return temp_ ? temp_->get_desc().get_size() : 0;
}

void DNNLReshapeFwd::Execute(const NDArray& input,
                             const NDArray& output,
                             const OpReqType& req,
                             void* workspace) {
  auto stream = DNNLStream::Get();
  auto in_mem = input.GetDNNLData();
  // register primitives and arguments
  std::vector<dnnl_args_map_t> args_map;
  size_t prims_size = prims_.size();
  if (prims_size == 1) {
    args_map.push_back({{DNNL_ARG_FROM, *in_mem}, {DNNL_ARG_TO, *output.GetDNNLData()}});
  } else if (prims_size == 2) {
    if (workspace) {
      temp_->set_data_handle(workspace);
      if (temp_reshaped_) {
        // temp_reshaped_ is a same-buffer view with a different ndim/dims.
        temp_reshaped_->set_data_handle(workspace);
      }
    }
    args_map.push_back({{DNNL_ARG_FROM, *in_mem}, {DNNL_ARG_TO, *temp_}});
    // When ndim differs the second step reorders the *reshaped* view of the
    // temp buffer into the output, otherwise reuse the temp_ view directly.
    const auto& src_for_copyback = temp_reshaped_ ? temp_reshaped_ : temp_;
    args_map.push_back({{DNNL_ARG_FROM, *src_for_copyback}, {DNNL_ARG_TO, *output.GetDNNLData()}});
  } else {
    CHECK(prims_size == 0 && req != kWriteTo) << "kWriteTo should never reach here.";
  }

  for (size_t i = 0; i < prims_size; i++) {
    stream->RegisterPrimArgs(prims_[i], args_map[i]);
  }
  stream->Submit();
  // invalidate dnnl memory in output
  const_cast<NDArray&>(output).InvalidateDNNLData();
}

DNNLReshapeFwd& GetReshapeForward(const OpReqType& req,
                                  const NDArray& input,
                                  const NDArray& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLReshapeSignature, DNNLReshapeFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLReshapeSignature, DNNLReshapeFwd, OpHash> fwds;
#endif
  DNNLReshapeSignature key;
  key.AddSign(req);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLReshapeFwd fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLReshapeForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& input,
                        const OpReqType& req,
                        const NDArray& output) {
  if (req == kNullOp)
    return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";
  auto fwd     = GetReshapeForward(req, input, output);
  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws =
        ctx.requested[0].get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = static_cast<void*>(ws.dptr_);
  }
  fwd.Execute(input, output, req, ws_ptr);
}

}  // namespace op
}  // namespace mxnet
#endif
