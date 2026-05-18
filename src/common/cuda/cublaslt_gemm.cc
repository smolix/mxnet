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
 * \brief cuBLASLt-backed GEMM wrappers
 *        (PR-A: fp32; PR-B: fp16/bf16/fp64; PR-C: stride-aware variants).
 *        See cublaslt_gemm.h.
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
// cares about (1.0, 0.0, "other"). The dtype triplet (a_type, b_type, c_type)
// + compute type fold into a single uint32 to avoid bloating the key. Scale
// type is implied by compute type (fp32 for {fp16,bf16,fp32} compute, fp64
// for fp64 compute), so we don't track it separately.
//
// PR-C adds (batch, stride_a, stride_b, stride_c). For the non-strided
// (batch==1, strides==0) wrappers these are constant, so the PR-A/B keys
// remain effectively the same shape.
struct GemmKey {
  int                m, n, k;
  int                lda, ldb, ldc;
  cublasOperation_t  opA, opB;
  int                alpha_is_one;
  int                beta_class;  // 0: =0, 1: =1, 2: other.
  uint32_t           dtype_tag;   // packed (io_dtype<<16 | compute_type)
  int                batch;       // PR-C
  int64_t            stride_a;    // PR-C
  int64_t            stride_b;    // PR-C
  int64_t            stride_c;    // PR-C
  bool operator==(const GemmKey& o) const {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb &&
           ldc == o.ldc && opA == o.opA && opB == o.opB &&
           alpha_is_one == o.alpha_is_one && beta_class == o.beta_class &&
           dtype_tag == o.dtype_tag && batch == o.batch &&
           stride_a == o.stride_a && stride_b == o.stride_b &&
           stride_c == o.stride_c;
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
    mix(static_cast<size_t>(k.dtype_tag));
    mix(static_cast<size_t>(k.batch));
    mix(static_cast<size_t>(k.stride_a));
    mix(static_cast<size_t>(k.stride_b));
    mix(static_cast<size_t>(k.stride_c));
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

inline int BetaClassF32(float beta) {
  if (beta == 0.0f) return 0;
  if (beta == 1.0f) return 1;
  return 2;
}

inline int BetaClassF64(double beta) {
  if (beta == 0.0) return 0;
  if (beta == 1.0) return 1;
  return 2;
}

inline uint32_t MakeDtypeTag(cudaDataType_t io, cublasComputeType_t compute) {
  // 16 bits each — both enums are well under that.
  return (static_cast<uint32_t>(io) << 16) |
         (static_cast<uint32_t>(compute) & 0xFFFFu);
}

// Core dispatcher. All dtype-specific wrappers funnel here. alpha/beta are
// expected to be host pointers of the scale-type size (fp32 for {fp16, bf16,
// fp32-compute} or fp64 for fp64-compute).
//
// PR-C: when batch > 1, sets CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT and
// CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET on each layout. The per-operand
// stride is the element offset between consecutive matrices in the batch.
// When batch == 1 the strides are ignored.
cublasStatus_t MaybeCublasLtGemmImpl(cublasHandle_t legacy_handle,
                                     cublasOperation_t opA,
                                     cublasOperation_t opB,
                                     int m, int n, int k,
                                     const void* alpha,
                                     const void* A, int lda,
                                     const void* B, int ldb,
                                     const void* beta,
                                     void* C, int ldc,
                                     cudaDataType_t io_dtype,
                                     cudaDataType_t scale_dtype,
                                     cublasComputeType_t compute_type,
                                     int alpha_is_one,
                                     int beta_class,
                                     int batch,
                                     int64_t stride_a,
                                     int64_t stride_b,
                                     int64_t stride_c) {
  int dev_id = -1;
  if (cudaGetDevice(&dev_id) != cudaSuccess || dev_id < 0) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  LtPool& pool = LtPool::Get(dev_id);
  if (pool.handle() == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;

  cudaStream_t stream = nullptr;
  cublasGetStream(legacy_handle, &stream);

  GemmKey key{m, n, k, lda, ldb, ldc, opA, opB,
              alpha_is_one, beta_class,
              MakeDtypeTag(io_dtype, compute_type),
              batch, stride_a, stride_b, stride_c};

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

  s = cublasLtMatmulDescCreate(&op_desc, compute_type, scale_dtype);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  // Attribute is documented as int32_t.
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
  s = cublasLtMatrixLayoutCreate(&a_lay, io_dtype, a_rows, a_cols, lda);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  s = cublasLtMatrixLayoutCreate(&b_lay, io_dtype, b_rows, b_cols, ldb);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  s = cublasLtMatrixLayoutCreate(&c_lay, io_dtype, m, n, ldc);
  if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }

  // PR-C: optional batch/stride attributes.
  if (batch > 1) {
    auto set_batch = [&](cublasLtMatrixLayout_t lay, int64_t stride) {
      cublasStatus_t st = cublasLtMatrixLayoutSetAttribute(
          lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
      if (st != CUBLAS_STATUS_SUCCESS) return st;
      return cublasLtMatrixLayoutSetAttribute(
          lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &stride, sizeof(stride));
    };
    s = set_batch(a_lay, stride_a);
    if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
    s = set_batch(b_lay, stride_b);
    if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
    s = set_batch(c_lay, stride_c);
    if (s != CUBLAS_STATUS_SUCCESS) { cleanup(); return s; }
  }

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
  // Compute as TF32 (matches the legacy CUBLAS_TF32_TENSOR_OP_MATH set
  // around cublasSgemmEx in linalg_impl.h). On Blackwell this is what
  // dispatches to the new sm_120 TF32 kernels.
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_32F, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               /*batch=*/1, 0, 0, 0);
}

cublasStatus_t MaybeCublasLtHgemm(cublasHandle_t legacy_handle,
                                  cublasOperation_t opA,
                                  cublasOperation_t opB,
                                  int m,
                                  int n,
                                  int k,
                                  const float* alpha,
                                  const void* A,
                                  int lda,
                                  const void* B,
                                  int ldb,
                                  const float* beta,
                                  void* C,
                                  int ldc) {
  // Pseudo-fp16: fp16 I/O, fp32 accumulate, fp32 alpha/beta. Matches the
  // legacy cublasGemmEx(... CUBLAS_COMPUTE_32F ...) path in
  // linalg_gemm<gpu, half_t>. Blackwell sm_120 routes this to fp16 tensor
  // cores via the new heuristic kernels.
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_16F, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               /*batch=*/1, 0, 0, 0);
}

cublasStatus_t MaybeCublasLtBf16Gemm(cublasHandle_t legacy_handle,
                                     cublasOperation_t opA,
                                     cublasOperation_t opB,
                                     int m,
                                     int n,
                                     int k,
                                     const float* alpha,
                                     const void* A,
                                     int lda,
                                     const void* B,
                                     int ldb,
                                     const float* beta,
                                     void* C,
                                     int ldc) {
  // bf16 I/O, fp32 accumulate, fp32 alpha/beta. There is no legacy
  // cublasBgemm; the caller side adds a cublasGemmEx fallback if Lt fails.
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_16BF, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               /*batch=*/1, 0, 0, 0);
}

cublasStatus_t MaybeCublasLtDgemm(cublasHandle_t legacy_handle,
                                  cublasOperation_t opA,
                                  cublasOperation_t opB,
                                  int m,
                                  int n,
                                  int k,
                                  const double* alpha,
                                  const double* A,
                                  int lda,
                                  const double* B,
                                  int ldb,
                                  const double* beta,
                                  double* C,
                                  int ldc) {
  // Native fp64. Scale and compute are both fp64. No TF32-style fast path.
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_64F, CUDA_R_64F,
                               CUBLAS_COMPUTE_64F,
                               (*alpha == 1.0) ? 1 : 0, BetaClassF64(*beta),
                               /*batch=*/1, 0, 0, 0);
}

/* ---------------------------- PR-C: strided ----------------------------- */

cublasStatus_t MaybeCublasLtSgemmStrided(cublasHandle_t legacy_handle,
                                         cublasOperation_t opA,
                                         cublasOperation_t opB,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda, int64_t stride_a,
                                         const float* B, int ldb, int64_t stride_b,
                                         const float* beta,
                                         float* C, int ldc, int64_t stride_c,
                                         int batch) {
  if (batch <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_32F, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               batch, stride_a, stride_b, stride_c);
}

cublasStatus_t MaybeCublasLtHgemmStrided(cublasHandle_t legacy_handle,
                                         cublasOperation_t opA,
                                         cublasOperation_t opB,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const void* A, int lda, int64_t stride_a,
                                         const void* B, int ldb, int64_t stride_b,
                                         const float* beta,
                                         void* C, int ldc, int64_t stride_c,
                                         int batch) {
  if (batch <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_16F, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               batch, stride_a, stride_b, stride_c);
}

cublasStatus_t MaybeCublasLtBf16GemmStrided(cublasHandle_t legacy_handle,
                                            cublasOperation_t opA,
                                            cublasOperation_t opB,
                                            int m, int n, int k,
                                            const float* alpha,
                                            const void* A, int lda, int64_t stride_a,
                                            const void* B, int ldb, int64_t stride_b,
                                            const float* beta,
                                            void* C, int ldc, int64_t stride_c,
                                            int batch) {
  if (batch <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_16BF, CUDA_R_32F,
                               CUBLAS_COMPUTE_32F,
                               (*alpha == 1.0f) ? 1 : 0, BetaClassF32(*beta),
                               batch, stride_a, stride_b, stride_c);
}

cublasStatus_t MaybeCublasLtDgemmStrided(cublasHandle_t legacy_handle,
                                         cublasOperation_t opA,
                                         cublasOperation_t opB,
                                         int m, int n, int k,
                                         const double* alpha,
                                         const double* A, int lda, int64_t stride_a,
                                         const double* B, int ldb, int64_t stride_b,
                                         const double* beta,
                                         double* C, int ldc, int64_t stride_c,
                                         int batch) {
  if (batch <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  return MaybeCublasLtGemmImpl(legacy_handle, opA, opB, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc,
                               CUDA_R_64F, CUDA_R_64F,
                               CUBLAS_COMPUTE_64F,
                               (*alpha == 1.0) ? 1 : 0, BetaClassF64(*beta),
                               batch, stride_a, stride_b, stride_c);
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
