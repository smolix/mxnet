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
 * \file cublaslt_gemm.cc
 * \brief cuBLASLt-backed fp32 GEMM (PR-A). See cublaslt_gemm.h.
 */

#include "cublaslt_gemm.h"

#if MXNET_USE_CUDA

#include <cublasLt.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>

#include <list>
#include <mutex>
#include <unordered_map>

namespace mxnet {
namespace common {
namespace cuda {

namespace {

constexpr size_t kWorkspaceBytes      = 32ull * 1024 * 1024;  // 32 MiB cap.
constexpr size_t kHeuristicCacheCap   = 256;

// Cache key. Quantizes alpha/beta to the three values the legacy fast path
// cares about (1.0, 0.0, "other"). Math-mode and dtype are fixed for fp32 in
// this PR; future PRs that add fp16/bf16 will need additional key fields.
struct GemmKey {
  int                m, n, k;
  int                lda, ldb, ldc;
  cublasOperation_t  opA, opB;
  int                alpha_is_one;
  int                beta_class;  // 0: =0, 1: =1, 2: other.
  bool operator==(const GemmKey& o) const {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb &&
           ldc == o.ldc && opA == o.opA && opB == o.opB &&
           alpha_is_one == o.alpha_is_one && beta_class == o.beta_class;
  }
};

struct GemmKeyHash {
  size_t operator()(const GemmKey& k) const noexcept {
    // 64-bit FNV-ish mix on the key bytes.
    size_t h = 1469598103934665603ull;
    auto mix = [&](size_t v) { h ^= v; h *= 1099511628211ull; };
    mix(static_cast<size_t>(k.m));
    mix(static_cast<size_t>(k.n));
    mix(static_cast<size_t>(k.k));
    mix(static_cast<size_t>(k.lda));
    mix(static_cast<size_t>(k.ldb));
    mix(static_cast<size_t>(k.ldc));
    mix(static_cast<size_t>(k.opA));
    mix(static_cast<size_t>(k.opB));
    mix(static_cast<size_t>(k.alpha_is_one));
    mix(static_cast<size_t>(k.beta_class));
    return h;
  }
};

struct CachedAlgo {
  cublasLtMatmulAlgo_t algo;
  size_t               workspace_bytes;
};

// Per-device pool: long-lived cublasLtHandle_t, workspace buffer, LRU cache.
class LtPool {
 public:
  static LtPool& Get(int dev_id) {
    static std::mutex                                  table_mu;
    static std::unordered_map<int, std::unique_ptr<LtPool>> table;
    std::lock_guard<std::mutex> lk(table_mu);
    auto it = table.find(dev_id);
    if (it == table.end()) {
      table[dev_id] = std::unique_ptr<LtPool>(new LtPool(dev_id));
      it            = table.find(dev_id);
    }
    return *it->second;
  }

  cublasLtHandle_t handle() const { return handle_; }

  // Returns workspace pointer for `bytes`. The buffer is allocated lazily on
  // the device this pool was constructed on and reused across calls. Returns
  // nullptr (and a non-success status via the caller path) on cudaMalloc
  // failure.
  void* Workspace(size_t bytes) {
    std::lock_guard<std::mutex> lk(ws_mu_);
    if (ws_bytes_ >= bytes && ws_ != nullptr) return ws_;
    if (ws_ != nullptr) {
      cudaFree(ws_);
      ws_       = nullptr;
      ws_bytes_ = 0;
    }
    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    if (prev_dev != dev_id_) cudaSetDevice(dev_id_);
    cudaError_t err = cudaMalloc(&ws_, bytes);
    if (prev_dev != -1 && prev_dev != dev_id_) cudaSetDevice(prev_dev);
    if (err != cudaSuccess) {
      ws_       = nullptr;
      ws_bytes_ = 0;
      return nullptr;
    }
    ws_bytes_ = bytes;
    return ws_;
  }

  // LRU lookup. Returns nullptr on miss.
  const CachedAlgo* Find(const GemmKey& key) {
    std::lock_guard<std::mutex> lk(cache_mu_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;
    lru_.splice(lru_.begin(), lru_, it->second.lru_it);
    return &it->second.algo;
  }

  void Insert(const GemmKey& key, const CachedAlgo& algo) {
    std::lock_guard<std::mutex> lk(cache_mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second.algo = algo;
      lru_.splice(lru_.begin(), lru_, it->second.lru_it);
      return;
    }
    lru_.push_front(key);
    cache_[key] = {algo, lru_.begin()};
    while (cache_.size() > kHeuristicCacheCap) {
      cache_.erase(lru_.back());
      lru_.pop_back();
    }
  }

 private:
  explicit LtPool(int dev_id) : dev_id_(dev_id) {
    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    if (prev_dev != dev_id_) cudaSetDevice(dev_id_);
    cublasStatus_t s = cublasLtCreate(&handle_);
    if (prev_dev != -1 && prev_dev != dev_id_) cudaSetDevice(prev_dev);
    if (s != CUBLAS_STATUS_SUCCESS) {
      handle_ = nullptr;
      LOG(WARNING) << "cublasLtCreate failed (status=" << s
                   << "); MXNET_USE_CUBLASLT will fall back to legacy cuBLAS.";
    }
  }
  // No destruction: pool lives for the process lifetime. Avoids
  // teardown-order issues with the CUDA driver.

  struct Entry {
    CachedAlgo                   algo;
    std::list<GemmKey>::iterator lru_it;
  };

  int                                                dev_id_;
  cublasLtHandle_t                                   handle_{nullptr};
  std::mutex                                         ws_mu_;
  void*                                              ws_{nullptr};
  size_t                                             ws_bytes_{0};
  std::mutex                                         cache_mu_;
  std::list<GemmKey>                                 lru_;
  std::unordered_map<GemmKey, Entry, GemmKeyHash>    cache_;
};

inline int BetaClass(float beta) {
  if (beta == 0.0f) return 0;
  if (beta == 1.0f) return 1;
  return 2;
}

}  // namespace

bool UseCuBlasLt() {
  static const bool flag =
      dmlc::GetEnv("MXNET_USE_CUBLASLT", dmlc::optional<bool>(false)).value();
  return flag;
}

cublasStatus_t MaybeCublasLtSgemm(cublasHandle_t legacy_handle,
                                  cublasOperation_t opA,
                                  cublasOperation_t opB,
                                  int m,
                                  int n,
                                  int k,
                                  const float* alpha,
                                  const float* A,
                                  int lda,
                                  const float* B,
                                  int ldb,
                                  const float* beta,
                                  float* C,
                                  int ldc) {
  int dev_id = -1;
  if (cudaGetDevice(&dev_id) != cudaSuccess || dev_id < 0) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  LtPool& pool = LtPool::Get(dev_id);
  if (pool.handle() == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;

  cudaStream_t stream = nullptr;
  cublasGetStream(legacy_handle, &stream);

  GemmKey key{m, n, k, lda, ldb, ldc, opA, opB,
              (*alpha == 1.0f) ? 1 : 0, BetaClass(*beta)};

  // Build descriptors. These are cheap (stack-resident opaques) but the algo
  // selection underneath is what we cache.
  cublasLtMatmulDesc_t   op_desc = nullptr;
  cublasLtMatrixLayout_t a_lay = nullptr, b_lay = nullptr, c_lay = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasStatus_t s;

  auto cleanup = [&]() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a_lay) cublasLtMatrixLayoutDestroy(a_lay);
    if (b_lay) cublasLtMatrixLayoutDestroy(b_lay);
    if (c_lay) cublasLtMatrixLayoutDestroy(c_lay);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
  };

  // Compute as TF32 (matches the legacy CUBLAS_TF32_TENSOR_OP_MATH set
  // around cublasSgemmEx in linalg_impl.h). On Blackwell this is what
  // dispatches to the new sm_120 TF32 kernels. Future PR can wire
  // MXNET_CUDA_ALLOW_TENSOR_CORE=0 to fall back to CUBLAS_COMPUTE_32F.
  s = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  // Attribute is documented as int32_t (see cublasLt.h around TRANSA).
  // cublasOperation_t is enum-backed by int; pass an int32_t to be safe.
  int32_t op_a_int = static_cast<int32_t>(opA);
  int32_t op_b_int = static_cast<int32_t>(opB);
  s = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_a_int, sizeof(op_a_int));
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  s = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_b_int, sizeof(op_b_int));
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }

  const int a_rows = (opA == CUBLAS_OP_N) ? m : k;
  const int a_cols = (opA == CUBLAS_OP_N) ? k : m;
  const int b_rows = (opB == CUBLAS_OP_N) ? k : n;
  const int b_cols = (opB == CUBLAS_OP_N) ? n : k;
  s = cublasLtMatrixLayoutCreate(&a_lay, CUDA_R_32F, a_rows, a_cols, lda);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  s = cublasLtMatrixLayoutCreate(&b_lay, CUDA_R_32F, b_rows, b_cols, ldb);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  s = cublasLtMatrixLayoutCreate(&c_lay, CUDA_R_32F, m, n, ldc);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }

  // Heuristic / algo selection.
  cublasLtMatmulAlgo_t algo;
  size_t               algo_ws_bytes = 0;
  const CachedAlgo*    hit           = pool.Find(key);
  if (hit != nullptr) {
    algo          = hit->algo;
    algo_ws_bytes = hit->workspace_bytes;
  } else {
    s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
    size_t ws_cap = kWorkspaceBytes;
    s = cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_cap, sizeof(ws_cap));
    if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }

    cublasLtMatmulHeuristicResult_t heur[1] = {};
    int returned = 0;
    s = cublasLtMatmulAlgoGetHeuristic(pool.handle(), op_desc, a_lay, b_lay,
                                       c_lay, c_lay, pref, 1, heur, &returned);
    if (s != CUBLAS_STATUS_SUCCESS || returned <= 0 ||
        heur[0].state != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return (s == CUBLAS_STATUS_SUCCESS) ? CUBLAS_STATUS_NOT_SUPPORTED : s;
    }
    algo          = heur[0].algo;
    algo_ws_bytes = heur[0].workspaceSize;
    pool.Insert(key, {algo, algo_ws_bytes});
  }

  void* ws = nullptr;
  if (algo_ws_bytes > 0) {
    ws = pool.Workspace(algo_ws_bytes);
    if (ws == nullptr) { cleanup(); return CUBLAS_STATUS_ALLOC_FAILED; }
  }

  s = cublasLtMatmul(pool.handle(), op_desc, alpha, A, a_lay, B, b_lay, beta,
                     C, c_lay, C, c_lay, &algo, ws, algo_ws_bytes, stream);
  cleanup();
  return s;
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
