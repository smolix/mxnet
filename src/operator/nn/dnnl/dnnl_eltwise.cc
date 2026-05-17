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
 * \file dnnl_eltwise.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_eltwise-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v3/dev_guide_eltwise.html
bool SupportDNNLEltwise(const NDArray& input) {
  return SupportDNNL<DNNLTypeMode::FloatTypes>(input);
}

DNNLEltwiseFwd& DNNLEltwiseFwd::GetCached(const NDArray& input,
                                          const NDArray& output,
                                          const dnnl::algorithm algorithm) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLEltwiseSignature, DNNLEltwiseFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLEltwiseSignature, DNNLEltwiseFwd, OpHash> fwds;
#endif

  DNNLEltwiseSignature key;
  key.AddSign(static_cast<int>(algorithm));
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLEltwiseFwd fwd(input, algorithm);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

DNNLEltwiseFwd::DNNLEltwiseFwd(const NDArray& input, const dnnl::algorithm algorithm) {
  auto src_desc = input.GetDNNLData()->get_desc();
  // v3: eltwise primitive_desc(engine, prop, algorithm, src_md, dst_md,
  //                            alpha, beta, attr={}). For algorithms that do
  //                            not need alpha/beta the values are ignored.
  // F8: alpha is hard-coded to 0 here. eltwise_soft_relu / eltwise_log_sigmoid
  // use alpha as a divisor; if either is ever added to the unary-op dispatch
  // (elemwise_unary_op.h:472-502) the call must route through DNNLActivation
  // {Forward,Backward}, not this fast path.
  CHECK(algorithm != dnnl::algorithm::eltwise_soft_relu)
      << "eltwise_soft_relu must go through DNNLActivationForward (alpha-aware path)";
  fwd_pd = std::make_shared<eltwise_fwd_pd_t>(mxnet::CpuEngine::Get()->get_engine(),
                                              dnnl::prop_kind::forward_inference,
                                              algorithm, src_desc, src_desc,
                                              0.f, 0.f);
  fwd    = std::make_shared<eltwise_fwd_t>(*fwd_pd);
}

void DNNLEltwiseFwd::Execute(const NDArray& input, const OpReqType& req, const NDArray& output) {
  auto engine           = mxnet::CpuEngine::Get()->get_engine();
  auto src              = input.GetDNNLData();
  dnnl_output_t out_mem = CreateDNNLMem(output, fwd_pd->dst_desc(), req, &input);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, *src},
      {DNNL_ARG_DST, *out_mem.second},
  };

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(output, out_mem);
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
