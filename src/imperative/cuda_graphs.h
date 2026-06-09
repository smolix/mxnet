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
 * Copyright (c) 2020 by Contributors
 * \file cuda_graphs.h
 * \brief Wrappers for use of CUDA Graphs API
 */
#ifndef MXNET_IMPERATIVE_CUDA_GRAPHS_H_
#define MXNET_IMPERATIVE_CUDA_GRAPHS_H_

#include <mxnet/base.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "./exec_pass.h"
#include "../common/cuda/utils.h"
#include "../common/cuda/cublaslt_gemm.h"

#if MXNET_USE_CUDA
#define CUDA_GRAPHS_AVAILABLE (CUDA_VERSION >= 10020)
#else
#define CUDA_GRAPHS_AVAILABLE (0)
#endif

// F3: the v3 port uses the CUDA-12+ shape of cudaGraphExecUpdate (single
// cudaGraphExecUpdateResultInfo* out-param). Older toolkits would still
// link against the legacy form and silently miscompile the switch below.
#if MXNET_USE_CUDA
static_assert(CUDA_VERSION >= 12000,
              "MXNet's onednn-v3-port branch requires CUDA 12.0+ for "
              "cudaGraphExecUpdate; build with an older CUDA toolkit "
              "is unsupported.");
#endif

#if CUDA_GRAPHS_AVAILABLE

namespace mxnet {
namespace cuda_graphs {

inline std::string CudaDim3ToString(const dim3& dims) {
  std::stringstream ss;
  if (dims.z != 1)
    ss << "(" << dims.x << "," << dims.y << "," << dims.z << ")";
  else if (dims.y != 1)
    ss << "(" << dims.x << "," << dims.y << ")";
  else
    ss << "(" << dims.x << ")";
  return ss.str();
}

// Return the list of CUDA Graph nodes from a graph
inline std::vector<cudaGraphNode_t> GetCudaGraphNodes(cudaGraph_t cuda_graph) {
  size_t numNodes;
  CUDA_CALL(cudaGraphGetNodes(cuda_graph, static_cast<cudaGraphNode_t*>(nullptr), &numNodes));
  if (numNodes == 0)
    return std::vector<cudaGraphNode_t>();
  std::vector<cudaGraphNode_t> graphNodes(numNodes);
  CUDA_CALL(cudaGraphGetNodes(cuda_graph, graphNodes.data(), &numNodes));
  return graphNodes;
}

// Create a description of a CUDA Graph node
inline std::string CudaGraphNodeToString(const cudaGraphNode_t node) {
  std::stringstream ss;

  // The following introspection calls are made through the driver API in order to bypass
  // problems that would arise if multiple statically-linked copies of the runtime exist.

  CUgraphNode cu_node = node;
  CUgraphNodeType t;
  CUDA_DRIVER_CALL(cuGraphNodeGetType(cu_node, &t));
  switch (t) {
    case CU_GRAPH_NODE_TYPE_KERNEL: {
      CUDA_KERNEL_NODE_PARAMS kparams;
      auto err = cuGraphKernelNodeGetParams(cu_node, &kparams);
      if (err == CUDA_SUCCESS) {
        ss << "GPUKernel@" << kparams.func;
        dim3 gridDim(kparams.gridDimX, kparams.gridDimY, kparams.gridDimZ);
        dim3 blockDim(kparams.blockDimX, kparams.blockDimY, kparams.blockDimZ);
        ss << "<<<gridDim=" << CudaDim3ToString(gridDim)
           << ", blkDim=" << CudaDim3ToString(blockDim) << ">>>";
        ss << "(...";
        if (kparams.sharedMemBytes != 0)
          ss << ", dynSharedMemBytes=" << kparams.sharedMemBytes;
        ss << ")";
      } else {
        ss << "GPU Kernel: cuGraphKernelNodeGetParams() fails with " << err;
      }
    } break;
    case CU_GRAPH_NODE_TYPE_MEMCPY: {
      cudaMemcpy3DParms mparams = {};
      CUDA_CALL(cudaGraphMemcpyNodeGetParams(node, &mparams));
      // If memcpy is seen, return without setting up runnable executor
      switch (mparams.kind) {
        case cudaMemcpyHostToHost:
          ss << "Host->Host ";
          break;
        case cudaMemcpyHostToDevice:
          ss << "Host->Device ";
          break;
        case cudaMemcpyDeviceToHost:
          ss << "Device->Host ";
          break;
        case cudaMemcpyDeviceToDevice:
          ss << "Device->Device ";
          break;
        default:
          break;
      }
      ss << "Memcpy";
    } break;
    case CU_GRAPH_NODE_TYPE_MEMSET: {
      cudaMemsetParams mparams = {};
      CUDA_CALL(cudaGraphMemsetNodeGetParams(node, &mparams));
      if (mparams.height == 1 && mparams.elementSize == 1) {
        ss << "cudaMemset(devPtr=" << mparams.dst << ", value=" << mparams.value
           << ", count=" << mparams.width << ")";
      } else {
        if (mparams.elementSize == 1)
          ss << "cudaMemset2D";
        else
          ss << "MemSet<elemBytes=" << mparams.elementSize << ">";
        ss << "(devPtr=" << mparams.dst << ", pitch=" << mparams.pitch
           << ", value=" << mparams.value << ", width=" << mparams.width
           << ", height=" << mparams.height << ")";
      }
    } break;
    case CU_GRAPH_NODE_TYPE_HOST:
      ss << "Host (executable) node";
      break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
      ss << "Node which executes an embedded graph";
      break;
    case CU_GRAPH_NODE_TYPE_EMPTY:
      ss << "Empty (no-op) node";
      break;
    default:
      ss << "Unknown/Invalid node type " << t;
  }
  return ss.str();
}

// ---- Phase 1: differential-replay correctness net ------------------------
// Result of comparing a graph-produced buffer against a conventionally-produced
// reference buffer for one output. Float dtypes compare by value (fp16/bf16
// widened to float, numpy-style |a-b| <= atol + rtol*|b|); all other dtypes
// compare byte-exact.
struct ReplayDiff {
  double max_abs  = 0.0;
  double max_rel  = 0.0;
  bool   violated = false;
};

template <typename DType>
inline void AccumFloatDiff(const void* a,
                           const void* b,
                           size_t n,
                           double rtol,
                           double atol,
                           ReplayDiff* d) {
  const DType* pa = static_cast<const DType*>(a);
  const DType* pb = static_cast<const DType*>(b);
  for (size_t i = 0; i < n; ++i) {
    const double va = static_cast<double>(static_cast<float>(pa[i]));
    const double vb = static_cast<double>(static_cast<float>(pb[i]));
    const double ad = std::fabs(va - vb);
    if (ad > d->max_abs)
      d->max_abs = ad;
    const double rd = ad / (std::fabs(vb) + 1e-30);
    if (rd > d->max_rel)
      d->max_rel = rd;
    if (ad > atol + rtol * std::fabs(vb))
      d->violated = true;
  }
}

inline ReplayDiff CompareHostBuffers(const void* graph_buf,
                                     const void* ref_buf,
                                     size_t n_elem,
                                     size_t bytes,
                                     int dtype,
                                     double rtol,
                                     double atol) {
  ReplayDiff d;
  switch (dtype) {
    case mshadow::kFloat32:
      AccumFloatDiff<float>(graph_buf, ref_buf, n_elem, rtol, atol, &d);
      break;
    case mshadow::kFloat64:
      AccumFloatDiff<double>(graph_buf, ref_buf, n_elem, rtol, atol, &d);
      break;
    case mshadow::kFloat16:
      AccumFloatDiff<mshadow::half::half_t>(graph_buf, ref_buf, n_elem, rtol, atol, &d);
      break;
    case mshadow::kBfloat16:
      AccumFloatDiff<mshadow::bfloat::bf16_t>(graph_buf, ref_buf, n_elem, rtol, atol, &d);
      break;
    default:
      d.violated = (std::memcmp(graph_buf, ref_buf, bytes) != 0);
      break;
  }
  return d;
}

// CUDA Graphs are managed in RAII fashion by smart pointers below.
// Function objects (preferred for readability) provide the deleter function.
class CudaGraphDeleter {
 public:
  void operator()(cudaGraph_t graph) {
    if (graph != nullptr)
      CUDA_CALL(cudaGraphDestroy(graph));
  }
};

// CUDA Graphs Executors are managed in RAII fashion by smart pointers below.
// Function objects (preferred for readability) provide the deleter function.
class CudaGraphExecDeleter {
 public:
  void operator()(cudaGraphExec_t graph_exec) {
    if (graph_exec != nullptr)
      CUDA_CALL(cudaGraphExecDestroy(graph_exec));
  }
};

// A CUDA Graphs executor for a portion of an Operator Segment (i.e. a 'SubSegment'),
// characterized by a starting index in the OpExecutor list and a number of ops.
class CudaGraphsSubSegExec {
 public:
  CudaGraphsSubSegExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                       const RunContext& rctx,
                       bool is_gpu,
                       bool verbose,
                       int from_op_idx,
                       int num_ops,
                       bool ops_are_cuda_graph_compatible = true)
      : from_op_idx_(from_op_idx),
        num_ops_(num_ops),
        graph_(nullptr),
        graph_exec_(nullptr),
        graph_exec_id_(0) {
    if (ops_are_cuda_graph_compatible) {
      MakeGraph(exec_list, rctx, is_gpu, verbose, from_op_idx, num_ops);
      MakeGraphExec(exec_list, rctx);
    }
  }

  void Update(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
              const RunContext& rctx,
              bool is_gpu,
              bool verbose) {
    // Current executor should be Runnable with the same parameters
    CHECK(IsRunnable());
    MakeGraph(exec_list, rctx, is_gpu, verbose, from_op_idx_, num_ops_);

    // CUDA 12 changed cudaGraphExecUpdate's signature to take a single
    // cudaGraphExecUpdateResultInfo* out parameter (was: error-node pointer
    // and result pointer separately). Use the new shape unconditionally; the
    // CUDA runtime ships a backwards-compat inline wrapper for the old form.
    cudaGraphExecUpdateResultInfo update_info = {};
    update_info.result = cudaGraphExecUpdateError;
    cudaError_t err = cudaGraphExecUpdate(graph_exec_.get(), graph_.get(), &update_info);
    switch (err) {
      case cudaErrorGraphExecUpdateFailure:
        MakeGraphExec(exec_list, rctx);
        break;
      case cudaSuccess:
        CHECK_EQ(update_info.result, cudaGraphExecUpdateSuccess);
        break;
      default:
        // Respond normally to unusual cudaGraphExecUpdate() ret vals
        CUDA_CALL(err);
    }
  }

  void RunSubSeg(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 const RunContext& rctx,
                 bool is_gpu,
                 bool verify             = false,
                 double rtol             = 1e-3,
                 double atol             = 1e-4,
                 bool verbose            = false,
                 const std::string& seg  = std::string()) {
    if (IsRunnable()) {
      auto s                  = rctx.get_stream<gpu>();
      const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
      if (verify) {
        VerifyReplay(exec_list, rctx, is_gpu, cu_s, rtol, atol, verbose, seg);
      } else {
        CUDA_CALL(cudaGraphLaunch(graph_exec_.get(), cu_s));
      }
    } else {
      // No CUDA Graph could be made for this portion of the OpSegment.  Run conventionally.
      for (int i = 0; i != num_ops_; ++i)
        exec_list[from_op_idx_ + i]->Run(rctx, is_gpu);
    }
  }

  bool IsRunnable() {
    return graph_exec_ != nullptr;
  }

  int NumGraphNodes() {
    size_t numNodes;
    CUDA_CALL(cudaGraphGetNodes(graph_.get(), static_cast<cudaGraphNode_t*>(nullptr), &numNodes));
    return numNodes;
  }

 private:
  void MakeGraph(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 const RunContext& rctx,
                 bool is_gpu,
                 bool verbose,
                 int from_op_idx,
                 int num_ops) {
    auto s                  = rctx.get_stream<gpu>();
    const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
    // Create CUDA Graph
    // Use of cudaStreamCaptureModeThreadLocal allows other threads like GPU Copy workers
    // to sync their streams without disturbing this capture.
    CUDA_CALL(cudaStreamBeginCapture(cu_s, cudaStreamCaptureModeThreadLocal));
    // Run those oprs in the sub segment while capturing- no actual GPU work is launched.
    static bool dbg_ops = dmlc::GetEnv("MXNET_CUDA_GRAPHS_DEBUG_OPS", false);
    for (int i = 0; i != num_ops; ++i) {
      if (dbg_ops) {
        const auto& ex     = exec_list[from_op_idx + i];
        const std::string nm = (ex->attrs.op != nullptr) ? ex->attrs.op->name : "<null>";
        LOG(INFO) << "[capture] running op " << i << "/" << num_ops << " : " << nm;
        ex->Run(rctx, is_gpu);
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaError_t e              = cudaStreamIsCapturing(cu_s, &st);
        LOG(INFO) << "[capture]   after " << nm << " : isCapturing err=" << e
                  << " status=" << static_cast<int>(st);
      } else {
        exec_list[from_op_idx + i]->Run(rctx, is_gpu);
      }
    }
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CALL(cudaStreamEndCapture(cu_s, &cuda_graph));
    graph_.reset(cuda_graph, CudaGraphDeleter());

    if (verbose) {
      std::vector<cudaGraphNode_t> graph_nodes = GetCudaGraphNodes(cuda_graph);
      size_t num_nodes                         = graph_nodes.size();
      LOG(INFO) << "  Graph has " << num_nodes << " nodes:";
      for (size_t i = 0; i != num_nodes; ++i) {
        LOG(INFO) << "    node " << i << " = " << CudaGraphNodeToString(graph_nodes[i]);
      }
    }
  }

  void MakeGraphExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                     const RunContext& rctx) {
    // Note that this routine is not invoked when a graph executor is merely updated.
    cudaGraphExec_t cuda_graph_exec;
    cudaGraphNode_t error_node;
    char log_buffer[1000];

    CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec, graph_.get(), &error_node, log_buffer, 1000));
    graph_exec_.reset(cuda_graph_exec, CudaGraphExecDeleter());

    // At this point we have a CUDA Graph executor
    static int num_graph_creations = 0;
    graph_exec_id_                 = num_graph_creations++;

    static size_t max_log_entries = dmlc::GetEnv("MXNET_CUDA_GRAPHS_MAX_LOG_ENTRIES", 0);
    if (graph_exec_id_ < max_log_entries) {
      LOG(INFO) << "Created CUDA graph " << graph_exec_id_;
      if (num_graph_creations == max_log_entries)
        LOG(INFO) << "Further CUDA graph creation log messages are suppressed.";
    }
    // Create a .dot file for graph visualization if requested
    static std::string dotfile_base = dmlc::GetEnv("MXNET_CUDA_GRAPHS_DBG_FILE", std::string());
    if (dotfile_base.size() > 0) {
#if CUDA_VERSION >= 11030
      static int dotfile_flags = dmlc::GetEnv("MXNET_CUDA_GRAPHS_DBG_FILE_FLAGS",
                                              static_cast<int>(cudaGraphDebugDotFlagsVerbose));
      std::ostringstream filename;
      const bool is_train = exec_list.size() > 0 && exec_list[0]->op_ctx.is_train;
      int dev_id          = rctx.ctx.dev_id;
      filename << dotfile_base << "-"
               << "dev" << dev_id << "-" << (is_train ? "trn" : "inf") << "-" << graph_exec_id_
               << ".dot";
      CUDA_CALL(cudaGraphDebugDotPrint(graph_.get(), filename.str().c_str(), dotfile_flags));
#else
      [[maybe_unused]] static bool dot_file_unsupported = []() {  // NOLINT
        LOG(INFO) << "MXNET_CUDA_GRAPHS_DBG_FILE setting ignored- requires CUDA version >= 11.3";
        return true;
      }();
#endif  // CUDA_VERSION >= 11030
    }
  }

  // Differential replay: run the captured graph AND a conventional execution of
  // the same ops, both starting from the identical pre-segment buffer state, and
  // assert the outputs match. Catches pointer staleness, stale workspace, and
  // any graph-vs-conventional divergence at segment granularity. Opt-in (debug)
  // because it allocates shadow buffers and runs the ops twice. Final buffer
  // state left equal to the graph result (same as the non-verify path).
  void VerifyReplay(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                    const RunContext& rctx,
                    bool is_gpu,
                    cudaStream_t cu_s,
                    double rtol,
                    double atol,
                    bool verbose,
                    const std::string& seg) {
    struct Buf {
      void* ptr     = nullptr;
      size_t bytes  = 0;
      size_t n_elem = 0;
      int dtype     = 0;
    };
    auto consider = [](const NDArray& nd, Buf* b) -> bool {
      if (nd.is_none())
        return false;
      const TBlob& blob = nd.data();
      if (blob.dptr_ == nullptr)
        return false;
      const size_t n = blob.shape_.Size();
      if (n == 0)
        return false;
      b->ptr    = blob.dptr_;
      b->n_elem = n;
      b->dtype  = blob.type_flag_;
      b->bytes  = n * mshadow::mshadow_sizeof(blob.type_flag_);
      return b->bytes > 0;
    };

    // Collect unique buffers: 'all' (in+out) for snapshot/restore, 'outs' (out)
    // for the comparison.
    std::map<void*, Buf> all;
    std::vector<Buf> outs;
    std::set<void*> out_seen;
    for (int i = 0; i != num_ops_; ++i) {
      auto& e = exec_list[from_op_idx_ + i];
      for (const auto& nd : e->in_array) {
        Buf b;
        if (consider(nd, &b))
          all[b.ptr] = b;
      }
      for (const auto& nd : e->out_array) {
        Buf b;
        if (consider(nd, &b)) {
          all[b.ptr] = b;
          if (out_seen.insert(b.ptr).second)
            outs.push_back(b);
        }
      }
    }

    // Snapshot the pre-segment state of every touched buffer.
    std::map<void*, void*> snap;
    for (auto& kv : all) {
      void* tmp = nullptr;
      CUDA_CALL(cudaMalloc(&tmp, kv.second.bytes));
      CUDA_CALL(
          cudaMemcpyAsync(tmp, kv.first, kv.second.bytes, cudaMemcpyDeviceToDevice, cu_s));
      snap[kv.first] = tmp;
    }
    CUDA_CALL(cudaStreamSynchronize(cu_s));

    // Helpers. run_conv() executes the ops conventionally into the real buffers
    // and stashes a fresh device copy of every output. restore() returns all
    // touched buffers to the snapshotted pre-segment state. compare() returns
    // false (and fills worst) if any output of 'a' differs from 'b' beyond tol.
    auto run_conv = [&](std::vector<void*>* dst) {
      for (int i = 0; i != num_ops_; ++i)
        exec_list[from_op_idx_ + i]->Run(rctx, is_gpu);
      CUDA_CALL(cudaStreamSynchronize(cu_s));
      dst->assign(outs.size(), nullptr);
      for (size_t i = 0; i < outs.size(); ++i) {
        CUDA_CALL(cudaMalloc(&(*dst)[i], outs[i].bytes));
        CUDA_CALL(cudaMemcpyAsync(
            (*dst)[i], outs[i].ptr, outs[i].bytes, cudaMemcpyDeviceToDevice, cu_s));
      }
      CUDA_CALL(cudaStreamSynchronize(cu_s));
    };
    auto restore = [&]() {
      for (auto& kv : snap)
        CUDA_CALL(cudaMemcpyAsync(
            kv.first, kv.second, all[kv.first].bytes, cudaMemcpyDeviceToDevice, cu_s));
      CUDA_CALL(cudaStreamSynchronize(cu_s));
    };
    auto compare = [&](const std::vector<void*>& a,
                       const std::vector<void*>& b,
                       ReplayDiff* worst,
                       size_t* worst_idx) -> bool {
      bool ok = true;
      for (size_t i = 0; i < outs.size(); ++i) {
        std::vector<char> ha(outs[i].bytes), hb(outs[i].bytes);
        CUDA_CALL(cudaMemcpy(ha.data(), a[i], outs[i].bytes, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hb.data(), b[i], outs[i].bytes, cudaMemcpyDeviceToHost));
        ReplayDiff d = CompareHostBuffers(
            ha.data(), hb.data(), outs[i].n_elem, outs[i].bytes, outs[i].dtype, rtol, atol);
        if (d.max_rel > worst->max_rel) {
          *worst     = d;
          *worst_idx = i;
        }
        if (d.violated)
          ok = false;
      }
      return ok;
    };
    auto free_list = [](std::vector<void*>* v) {
      for (void* p : *v)
        CUDA_CALL_NONFATAL(cudaFree(p));
      v->clear();
    };

    // Determinism self-check: run the conventional path twice. If its own
    // outputs differ (RNG ops, cuDNN dropout state, atomics with nondeterministic
    // reduction order), graph-vs-conventional equality is not a valid test, so
    // skip the comparison for this segment (RNG-under-replay correctness is a
    // separate concern, see CUDA_GRAPHS_PLAN.md Phase 4). We still run the graph
    // so downstream state matches the normal graphs-on path.
    std::vector<void*> ref1, ref2;
    run_conv(&ref1);
    restore();
    run_conv(&ref2);
    ReplayDiff dd;
    size_t dd_idx        = 0;
    bool deterministic   = compare(ref1, ref2, &dd, &dd_idx);
    free_list(&ref2);

    restore();
    CUDA_CALL(cudaGraphLaunch(graph_exec_.get(), cu_s));
    CUDA_CALL(cudaStreamSynchronize(cu_s));

    if (!deterministic) {
      free_list(&ref1);
      for (auto& kv : snap)
        CUDA_CALL_NONFATAL(cudaFree(kv.second));
      if (verbose) {
        LOG(INFO) << "CUDA Graphs replay SKIPPED (nondeterministic segment) [" << seg
                  << "] subseg[" << from_op_idx_ << ":" << (from_op_idx_ + num_ops_ - 1)
                  << "]: conventional run not self-consistent (worst output #" << dd_idx
                  << " max_rel=" << dd.max_rel << ").";
      }
      return;
    }

    // Deterministic segment: graph result (real buffers) must match conventional.
    std::vector<void*> real(outs.size());
    for (size_t i = 0; i < outs.size(); ++i)
      real[i] = outs[i].ptr;
    ReplayDiff worst;
    size_t worst_idx = 0;
    bool ok          = compare(real, ref1, &worst, &worst_idx);

    free_list(&ref1);
    for (auto& kv : snap)
      CUDA_CALL_NONFATAL(cudaFree(kv.second));

    if (!ok) {
      LOG(FATAL) << "CUDA Graphs differential-replay MISMATCH in segment [" << seg << "] subseg["
                 << from_op_idx_ << ":" << (from_op_idx_ + num_ops_ - 1) << "]: worst output #"
                 << worst_idx << " max_abs=" << worst.max_abs << " max_rel=" << worst.max_rel
                 << " (rtol=" << rtol << ", atol=" << atol << "). The captured graph diverged "
                 << "from conventional execution.";
    } else if (verbose) {
      LOG(INFO) << "CUDA Graphs replay OK: segment [" << seg << "] subseg[" << from_op_idx_ << ":"
                << (from_op_idx_ + num_ops_ - 1) << "] " << outs.size()
                << " outputs, worst max_rel=" << worst.max_rel;
    }
  }

  int from_op_idx_;
  int num_ops_;
  using cudaGraphStruct_t     = typename std::remove_pointer<cudaGraph_t>::type;
  using cudaGraphExecStruct_t = typename std::remove_pointer<cudaGraphExec_t>::type;
  std::shared_ptr<cudaGraphStruct_t> graph_;
  std::shared_ptr<cudaGraphExecStruct_t> graph_exec_;
  size_t graph_exec_id_;
};

// The CudaGraph executor and associated Tempspace ptrs for which it is valid.
struct CudaGraphInfo {
  std::vector<CudaGraphsSubSegExec> cuda_graph_subseg_execs;
  bool has_been_run_conventionally = false;
  std::vector<void*> tempspace_dptrs;
};
// A CUDA graph is maintained for every combination of cudaStream_t (i.e. GPU Worker) and
// the state of the is_train flag of the OpContext.  If the tempspace_dptrs change, we
// don't expect to ever see the old tempspace_dptrs config again, so we discard the CUDA graph.
struct CudaGraphCacheKey {
  cudaStream_t cu_s;
  bool is_train;
  // overload '<' so CudaGraphCacheKey can be used as a std::map key
  bool operator<(const CudaGraphCacheKey& other) const {
    return cu_s < other.cu_s || (cu_s == other.cu_s && is_train < other.is_train);
  }
};
using CudaGraphCache = std::map<CudaGraphCacheKey, CudaGraphInfo>;

class CudaGraphsExec {
 public:
  CudaGraphsExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 bool is_gpu,
                 const char* opr_names,
                 bool default_enable = false)
      : verbose_(false),
        is_enabled_(false),
        verify_(false),
        verify_every_(1),
        verify_rtol_(1e-3),
        verify_atol_(1e-4),
        verify_counter_(0) {
    opr_names_ = opr_names ? std::string(opr_names) : std::string();
    if (is_gpu) {
      // Phase 5: capture defaults on in the static-shape cached-op regime
      // (default_enable). MXNET_ENABLE_CUDA_GRAPHS still overrides either way.
      is_enabled_ = dmlc::GetEnv("MXNET_ENABLE_CUDA_GRAPHS", default_enable);
      // When this segment will capture, make gemm capture-safe for the whole
      // process: force cuBLASLt on now (before warm-up) so its persistent
      // per-stream workspace is allocated conventionally, not during capture.
      if (is_enabled_ && mxnet::common::cuda::AllowGemmCapture())
        mxnet::common::cuda::EnableCuBlasLtForGraphs();
      verbose_    = dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERBOSE", false);
      // Differential-replay correctness net (Phase 1): opt-in, debug-only.
      verify_       = dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERIFY", false);
      verify_every_ = std::max(1, dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERIFY_EVERY", 1));
      verify_rtol_  = dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERIFY_RTOL", 1e-3);
      verify_atol_  = dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERIFY_ATOL", 1e-4);
      SetTempSpaces(exec_list);
    }
  }

  void RunAll(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
              const RunContext& rctx,
              bool is_gpu) {
    // If this a CPU op or CUDA Graphs use isn't possible, run normally and return
    if (!is_gpu || !is_enabled_) {
      // Run all opr in the sub-graph
      exec::OpExecutor::RunAll(exec_list, rctx, is_gpu);
      return;
    }

    // Also if we're in a warm-up period where tempspace pointers are likely
    // to change, run normally and return
    auto s                  = rctx.get_stream<gpu>();
    const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
    // All the ops in the bulked segment will have the same setting of is_train as the first op
    const bool is_train         = exec_list.size() > 0 && exec_list[0]->op_ctx.is_train;
    const CudaGraphCacheKey key = {cu_s, is_train};
    // Look-up the CUDA Graph info for this combo of stream and is_train setting
    // This may create a default-initialized new entry.
    auto& cuda_graph_info = cache_[key];
    if (!cuda_graph_info.has_been_run_conventionally) {
      // Run all opr in the sub-graph
      exec::OpExecutor::RunAll(exec_list, rctx, is_gpu);
      cuda_graph_info.has_been_run_conventionally = true;
      return;
    }

    // At this point we will launch one or more CUDA Graphs through CUDA Graphs 'executors'
    //     (there might be more than one executor if some ops in the segment are not capturable)
    auto before_exec_tempspace_ptrs = GetGPUTempspacePtrs(s);

    // Executors exist, but the tempspace pts have changed, so update them in-place via 'recapture'.
    if (cuda_graph_info.cuda_graph_subseg_execs.size() > 0 &&
        cuda_graph_info.tempspace_dptrs != before_exec_tempspace_ptrs) {
      // Update all runnable executors.  Non-runnable executors launch their ops conventionally.
      for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs) {
        if (subseg_exec.IsRunnable())
          subseg_exec.Update(exec_list, rctx, is_gpu, verbose_);
      }
    } else if (cuda_graph_info.cuda_graph_subseg_execs.size() == 0) {
      // No executors exist yet, so create them.
      if (verbose_)
        LOG(INFO) << "Capturing CUDA graph of op segment " << opr_names_;
      // Make one or more CUDA Graphs, avoiding ops that are not compatible.
      for (size_t first_op_idx = 0; first_op_idx != exec_list.size();) {
        int num_good_ops = 0;
        for (size_t last_op_idx = first_op_idx; last_op_idx != exec_list.size(); ++last_op_idx) {
          if (OpOK(exec_list[last_op_idx]))
            num_good_ops++;
          else
            break;
        }
        if (num_good_ops > 0) {
          CreateSubExecOverRegion(exec_list,
                                  rctx,
                                  is_gpu,
                                  first_op_idx,
                                  first_op_idx + num_good_ops,
                                  &cuda_graph_info.cuda_graph_subseg_execs);
          first_op_idx += num_good_ops;
        }
        if (first_op_idx != exec_list.size()) {
          // We had to have hit an op that was not OK.
          if (verbose_) {
            LOG(INFO) << "Bypassing notOK op segment[" << first_op_idx << "," << first_op_idx << "]"
                      << " of op segment " << opr_names_;
          }
          CudaGraphsSubSegExec notOK_opseg(exec_list, rctx, is_gpu, false, first_op_idx, 1, false);
          cuda_graph_info.cuda_graph_subseg_execs.push_back(notOK_opseg);
          first_op_idx++;
        }
      }
      // During graph capture, the ops may be asking for the tempworkspace.  This should
      // not alter the base pointers, since this op seg has been executed before on this
      // stream (i.e. on this gpu worker).  Safest to double-check this though.
      auto after_capture_tempspace_ptrs = GetGPUTempspacePtrs(s);
      if (before_exec_tempspace_ptrs != after_capture_tempspace_ptrs)
        LOG(FATAL) << "Internal error: saw change in TempSpace ptrs during CUDA graph use.";
      cuda_graph_info.tempspace_dptrs = before_exec_tempspace_ptrs;

      // One-line capture summary (Phase 1): how the segment was carved into
      // graphs vs conventionally-run ops. Makes coverage/regressions visible.
      if (verbose_) {
        int n_graphs = 0, n_bypassed = 0, n_nodes = 0;
        for (auto& se : cuda_graph_info.cuda_graph_subseg_execs) {
          if (se.IsRunnable()) {
            n_graphs++;
            n_nodes += se.NumGraphNodes();
          } else {
            n_bypassed++;
          }
        }
        LOG(INFO) << "CUDA graph segment summary [" << opr_names_ << "]: "
                  << cuda_graph_info.cuda_graph_subseg_execs.size() << " subsegs -> " << n_graphs
                  << " graphs (" << n_nodes << " nodes), " << n_bypassed << " bypassed ops.";
      }
    }
    // Now execute the CUDA Graph that we either just created or looked-up in the cache.
    if (verbose_) {
      int runnable_execs = 0;
      int bypassed_ops   = 0;
      for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs) {
        if (subseg_exec.IsRunnable()) {
          LOG(INFO) << "Launching captured graph with " << subseg_exec.NumGraphNodes() << " nodes.";
          runnable_execs++;
        } else {
          bypassed_ops++;
        }
      }
      if (bypassed_ops > 0)
        LOG(INFO) << "    (bypassing " << bypassed_ops << " un-capturable ops)";
    }
    bool do_verify = false;
    if (verify_ && cuda_graph_info.cuda_graph_subseg_execs.size() > 0)
      do_verify = (++verify_counter_ % verify_every_ == 0);
    for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs)
      subseg_exec.RunSubSeg(
          exec_list, rctx, is_gpu, do_verify, verify_rtol_, verify_atol_, verbose_, opr_names_);
  }

 private:
  // Make a CUDA Graph of the region of ops [from_op_idx, upto_op_idx).  If such a graph
  // is not runnable, e.g. if it includes memcpys from unpinned cpu memory, then make a
  // number of smaller graphs that avoid those ops with the memcpys.
  void CreateSubExecOverRegion(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                               const RunContext& rctx,
                               bool is_gpu,
                               size_t from_op_idx,
                               size_t upto_op_idx,
                               std::vector<CudaGraphsSubSegExec>* cuda_graph_subseg_execs) {
    // Optimistically try to create a CUDA Graph of the entire op segment region

    int num_ops = upto_op_idx - from_op_idx;
    CudaGraphsSubSegExec full_opseg(exec_list, rctx, is_gpu, verbose_, from_op_idx, num_ops);
    if (full_opseg.IsRunnable()) {
      cuda_graph_subseg_execs->push_back(full_opseg);
    } else {
      if (verbose_)
        LOG(INFO) << "  Graph was not runnable- creating op sub-segments...";
      // Enter fall-back approach to making many sub-execs
      for (size_t first_op_idx = from_op_idx; first_op_idx != upto_op_idx;) {
        int num_good_ops = 0;
        for (size_t last_op_idx = first_op_idx; last_op_idx != upto_op_idx; ++last_op_idx) {
          CudaGraphsSubSegExec single_opseg(exec_list, rctx, is_gpu, false, last_op_idx, 1);
          if (single_opseg.IsRunnable())
            num_good_ops++;
          // Is it time to create a subseg exec from accumulated good ops?
          if (num_good_ops > 0 && (last_op_idx == upto_op_idx - 1 || !single_opseg.IsRunnable())) {
            if (verbose_)
              LOG(INFO) << "Capturing CUDA graph of op sub segment[" << first_op_idx << ":"
                        << (first_op_idx + num_good_ops - 1) << "]"
                        << " of op segment " << opr_names_;
            CudaGraphsSubSegExec good_opseg(
                exec_list, rctx, is_gpu, verbose_, first_op_idx, num_good_ops);
            CHECK(good_opseg.IsRunnable()) << "Unexpected issue with CUDA Graphs creation";
            cuda_graph_subseg_execs->push_back(good_opseg);
            first_op_idx += num_good_ops;
          }
          // If the last single op was not runnable, use the exec to handle that op conventionally
          if (!single_opseg.IsRunnable()) {
            if (verbose_) {
              LOG(INFO) << "Bypassing op sub segment[" << last_op_idx << "," << last_op_idx << "]"
                        << " of op segment " << opr_names_;
              // Generate throw-away exec in order to produce a diagnostic listing of graph nodes
              CudaGraphsSubSegExec dummy(exec_list, rctx, is_gpu, verbose_, last_op_idx, 1);
            }
            cuda_graph_subseg_execs->push_back(single_opseg);
            first_op_idx++;
            break;
          }
        }
      }
    }
  }

  // Is the Op OK to make part of a CUDA Graph?
  bool OpOK(const std::shared_ptr<exec::OpExecutor>& exec) {
    static auto& fgraphcompatible = Op::GetAttr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible");
    static auto& fcompute_ex      = Op::GetAttr<FComputeEx>("FComputeEx<gpu>");
    static auto& fstatefulcompute = Op::GetAttr<FStatefulCompute>("FStatefulCompute<gpu>");
    static auto& fstatefulcompute_ex = Op::GetAttr<FStatefulComputeEx>("FStatefulComputeEx<gpu>");
    const auto& attrs                = exec->attrs;
    if (attrs.op != nullptr) {
      const auto f = fgraphcompatible.get(attrs.op, nullptr);
      if (f != nullptr) {
        return f(attrs, exec->op_ctx.is_train);
      }
      if (fstatefulcompute.get(attrs.op, nullptr) != nullptr ||
          fstatefulcompute_ex.get(attrs.op, nullptr) != nullptr) {
        if (verbose_) {
          LOG(INFO) << "Omitting stateful operator " << attrs.op->name << " from CUDA graph.";
        }
        return false;
      }
      if ((fcompute_ex.get(attrs.op, nullptr) != nullptr &&
           exec->dispatch_mode == DispatchMode::kFComputeEx) ||
          exec->dispatch_mode == DispatchMode::kFComputeFallback) {
        if (verbose_) {
          LOG(INFO) << "Omitting operator " << attrs.op->name
                    << " from CUDA graph due to dispatch mode "
                    << static_cast<int>(exec->dispatch_mode);
        }
        return false;
      }
    }
    for (auto& resource : exec->op_ctx.requested) {
      if (!(resource.req.type == ResourceRequest::kTempSpace)) {
        if (verbose_) {
          LOG(INFO) << "Omitting operator " << attrs.op->name
                    << " from CUDA graph due to using the resource type "
                    << static_cast<int>(resource.req.type);
        }
        return false;
      }
    }
    return true;
  }

  // Determine Tempspaces used by ops.  Other resource uses disable CUDA Graphs.
  void SetTempSpaces(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list) {
    // Gather info about the ops use of TempSpace.
    if (is_enabled_) {
      std::set<Resource*> tempspaces_set;
      for (auto& exec : exec_list) {
        for (auto& resource : exec->op_ctx.requested) {
          if (resource.req.type == ResourceRequest::kTempSpace) {
            tempspaces_set.insert(&resource);
          }
        }
      }
      tempspaces_.assign(tempspaces_set.begin(), tempspaces_set.end());
    }
  }

  // Return the addresses of the gpu TempSpace areas
  std::vector<void*> GetGPUTempspacePtrs(mshadow::Stream<gpu>* s) {
    std::vector<void*> ret;
    for (const auto& resource : tempspaces_) {
      // Ask for minimal allocation to get base pointer without increasing the size
      auto* base_ptr = resource->get_space_typed<gpu, 1, char>(mshadow::Shape1(1), s).dptr_;
      ret.push_back(static_cast<void*>(base_ptr));
    }
    return ret;
  }

  CudaGraphCache cache_;
  std::vector<Resource*> tempspaces_;
  std::string opr_names_;
  bool verbose_;
  bool is_enabled_;
  // Differential-replay correctness net (Phase 1).
  bool verify_;
  int verify_every_;
  double verify_rtol_;
  double verify_atol_;
  size_t verify_counter_;
};

}  // namespace cuda_graphs
}  // namespace mxnet

#endif  // CUDA_GRAPHS_AVAILABLE

#endif  // MXNET_IMPERATIVE_CUDA_GRAPHS_H_
