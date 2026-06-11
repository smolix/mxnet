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
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <tuple>
#include <utility>
#include "./storage_manager.h"
#include "../profiler/storage_profiler.h"

namespace mxnet {
namespace storage {

typedef enum {
  pool_type,
  pool_page_size,
  large_alloc_size,
  round_linear_cutoff,
  pool_reserve,
  per_bucket_limit,
  oom_retries,
  oom_backoff_ms,
} env_var_type;

const std::string env_var_name(const char* dev_type, env_var_type type);

#if MXNET_USE_CUDA
#define SET_DEVICE(device_store, contextHelper, ctx, flag) \
  const auto* device_store = flag ? contextHelper.get()->SetCurrentDevice(ctx) : nullptr;
#define UNSET_DEVICE(device_store) delete device_store

#define SET_GPU_PROFILER(prof, contextHelper)                                                    \
  auto prof = contextHelper->contextGPU() ? profiler::GpuDeviceStorageProfiler::Get() : nullptr; \
  if (prof && !prof->IsProfiling()) {                                                            \
    prof = nullptr;                                                                              \
  }

#define GPU_PROFILER_ON_FREE(prof, pntr) \
  if (prof) {                            \
    prof->OnFree(pntr);                  \
  }
#else
// empty macros when MxNet is compiled without CUDA support
#define SET_DEVICE(...)
#define UNSET_DEVICE(...)
#define SET_GPU_PROFILER(prof, ...)
#define GPU_PROFILER_ON_FREE(prof, ...)
#endif

/*!
 * \brief Storage manager with a memory pool for GPU/CPU/CPUPunned memory chunks
 * memory chunks which reused based on rounded size match.
 * Rounding method is defined by the template parameter BucketingStrategy.
 * Memory pool type is defined by the template parameter StoringMethod
 * Allocation/freeing of memory is done by contextHelper_, which is the pointer
 * to one of memory specific instance of the class, derived from ContextHelper
 */
template <typename BucketingStrategy, typename StoringMethod>
class PooledStorageManager : public StorageManager, public BucketingStrategy, public StoringMethod {
 public:
  explicit PooledStorageManager(const Context& ctx, int num_gpu_device) {
    const char* dev_type = nullptr;
    switch (dev_type_ = ctx.dev_type) {
#if MXNET_USE_CUDA
      case Context::kGPU:
        contextHelper_ = std::make_unique<ContextHelperGPU>();
        dev_type       = "GPU";
        break;
      case Context::kCPUPinned:
        dev_type = "CPU_PINNED";
        if (num_gpu_device > 1) {
          contextHelper_ = std::make_unique<ContextHelperPinned>();
          dev_type_      = Context::kGPU;
          break;
        }
#else
      case Context::kCPUPinned:
        dev_type = "CPU_PINNED";
#endif
        dev_type_ = Context::kCPU;
      case Context::kCPU:
        contextHelper_ = std::make_unique<ContextHelperCPU>();
        dev_type       = "CPU";
      default:
        break;
    }

    BucketingStrategy::InitRoundHelper(dev_type);
    StoringMethod::InitContainer(this);
    contextHelper_->set_initilal_context(ctx);

    // Per-bucket retention cap. 0 = unlimited (legacy behavior).
    // See apache/mxnet#17335: dynamic-shape workloads (variable-length NLP,
    // Bucket batchify, dynamic image) hit many distinct rounded sizes and
    // the pool retains every freed chunk forever, eventually OOMing while
    // device memory is mostly idle pool slots.
    if (dev_type) {
      const auto env_var = env_var_name(dev_type, per_bucket_limit);
      per_bucket_limit_  = dmlc::GetEnv(env_var.c_str(), 4);
    } else {
      per_bucket_limit_ = 0;
    }

    // percentage of reserved memory
    if (dev_type) {
      const auto env_var       = env_var_name(dev_type, pool_reserve);
      const size_t reserve     = dmlc::GetEnv(env_var.c_str(), 5);
      const size_t total       = std::get<1>(contextHelper_->getMemoryInfo());
      memory_allocation_limit_ = total * reserve / 100;
    }

    // OOM retry policy. d2l-neu observed cnn-design / sentiment-analysis-rnn
    // OOM under 2-notebooks-per-GPU contention while PyTorch / JAX / TF on
    // the same hardware survived: a transient cudaMalloc failure could be
    // satisfied a moment later once the neighbor process flushes its own
    // pool. Default 4 retries with 50ms exponential backoff (50, 100, 200,
    // 400 ms) gives ~750ms total wait before LOG(FATAL). Backoff is capped
    // at 1s per attempt to bound worst-case latency.
    if (dev_type) {
      const auto env_var_retries  = env_var_name(dev_type, oom_retries);
      const auto env_var_backoff  = env_var_name(dev_type, oom_backoff_ms);
      oom_retries_                = dmlc::GetEnv(env_var_retries.c_str(), 4);
      oom_backoff_ms_             = dmlc::GetEnv(env_var_backoff.c_str(), 50);
    } else {
      oom_retries_    = 0;
      oom_backoff_ms_ = 0;
    }
  }
  /*!
   * \brief Default destructor.
   */
  ~PooledStorageManager() override {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle, bool failsafe) override;
  void Free(Storage::Handle handle) override {
    // Insert returned memory in cache
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    const auto bucket_id = BucketingStrategy::get_bucket(handle.size);
    if (per_bucket_limit_ > 0 &&
        StoringMethod::BucketSize(bucket_id) >= per_bucket_limit_) {
      // Bucket is at retention cap: return chunk directly to the device
      // allocator instead of growing the pool unboundedly. Trades a
      // cudaFree call for bounded steady-state pool size.
      SET_DEVICE(device_store, contextHelper_, handle.ctx, true);
#if MXNET_USE_CUDA
      // M13: this chunk may still have in-flight reader/writer kernels recorded
      // against it. The normal pooled-reuse path waits on handle.sync_obj before
      // handing the buffer out again; this direct-free branch must do the same, or
      // we cudaFree device memory the GPU is still using (use-after-free).
      if (dev_type_ == Context::kGPU) {
        for (auto ev : handle.sync_obj.events) {
          auto valid_ev = ev.lock();
          if (valid_ev) {
            MSHADOW_CUDA_CALL(cudaEventSynchronize(*valid_ev));
          }
        }
      }
#endif
      contextHelper_->Free(handle.dptr);
      SET_GPU_PROFILER(profilerGPU, contextHelper_);
      GPU_PROFILER_ON_FREE(profilerGPU, handle.dptr);
      UNSET_DEVICE(device_store);
      used_memory_ -= BucketingStrategy::RoundAllocSizeForBucket(bucket_id);
      return;
    }
    StoringMethod::InsertInCache(bucket_id, handle.dptr, handle.sync_obj);
  }

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    SET_DEVICE(device_store, contextHelper_, handle.ctx, true);
    contextHelper_->Free(handle.dptr);
    SET_GPU_PROFILER(profilerGPU, contextHelper_);
    GPU_PROFILER_ON_FREE(profilerGPU, handle.dptr);
    UNSET_DEVICE(device_store);
    used_memory_ -= BucketingStrategy::RoundAllocSize(handle.size);
  }

  void ReleaseAll() override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    ReleaseAllNoLock();
  }

 private:
  void ReleaseAllNoLock(bool set_device = true) {
    SET_DEVICE(device_store, contextHelper_, contextHelper_->initilal_context(), set_device);
    used_memory_ -= StoringMethod::ReleaseAllNoLock(contextHelper_.get(), this);
    UNSET_DEVICE(device_store);
  }

  bool MemoryIsAvailable(size_t roundSize) const {
    const auto free = contextHelper_->freeMemorySize();
    return free > roundSize && memory_allocation_limit_ <= free - roundSize;
  }

  // device type of used context
  Context::DeviceType dev_type_;
  // used memory
  size_t used_memory_ = 0;
  // minimum amount of memory, which will never be allocated
  size_t memory_allocation_limit_ = 0;
  // max free chunks retained per bucket (0 = unlimited, legacy behavior)
  size_t per_bucket_limit_ = 0;
  // additional cudaMalloc retries with backoff before LOG(FATAL) on OOM
  size_t oom_retries_ = 0;
  // initial backoff between OOM retries (doubled per attempt, capped at 1000ms)
  size_t oom_backoff_ms_ = 0;
  // Pointer to the Helper, supporting some context-specific operations in GPU/CPU/CPUPinned context
  std::unique_ptr<ContextHelper> contextHelper_;
};

template <typename BucketingStrategy, typename StoringMethod>
void PooledStorageManager<BucketingStrategy, StoringMethod>::Alloc(Storage::Handle* handle,
                                                                   bool failsafe) {
  // unique_lock (not lock_guard) so the OOM-retry path below can release the
  // per-device storage mutex while it sleeps/syncs -- otherwise no other thread
  // could Free() memory back to relieve the very OOM we are waiting out (H10).
  std::unique_lock<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
  const auto bucket_id = BucketingStrategy::get_bucket(handle->size);
  size_t roundSize     = 0;
  auto reuse_pool      = StoringMethod::GetMemStorage(bucket_id);
  if (!reuse_pool) {
    SET_DEVICE(device_store, contextHelper_, handle->ctx, true);
    roundSize = BucketingStrategy::RoundAllocSizeForBucket(bucket_id);
    if (!MemoryIsAvailable(roundSize))
      ReleaseAllNoLock(false);

    void* ret = nullptr;
    auto e    = contextHelper_->Malloc(&ret, roundSize);
    if (e) {
      // retry in case of fragmentation
      ReleaseAllNoLock(false);
      e = contextHelper_->Malloc(&ret, roundSize);
#if MXNET_USE_CUDA
      // Cross-process contention retry loop: when two MXNet processes share
      // a GPU, both can race on a single cudaMalloc with neither having a
      // pool to flush. Wait briefly and retry — the neighbor may release
      // a chunk between iterations. Without this, the loser of the race
      // hits LOG(FATAL) even though physical memory frees up moments later
      // (this is what PyTorch's caching allocator does to survive the same
      // condition; see d2l-ssd-bug.md Issue 2 / D2L-Bug-2).
      if (e == cudaErrorMemoryAllocation && dev_type_ == Context::kGPU &&
          oom_retries_ > 0) {
        // Clear sticky error so subsequent cuda calls observe success state.
        cudaGetLastError();
        size_t backoff_ms = oom_backoff_ms_;
        for (size_t attempt = 0; attempt < oom_retries_ && e; ++attempt) {
          // Sync to drain any in-flight frees from this process before
          // sleeping. cudaDeviceSynchronize itself does not error after
          // a failed cudaMalloc; if it does, propagate that as the real
          // failure rather than the OOM.
          // Release the storage mutex across the sync+sleep so other threads can
          // Free() chunks back to the device/pool meanwhile (H10). Pool state
          // (used_memory_, the cache) is only touched again after re-locking.
          lock.unlock();
          cudaError_t sync_err = cudaDeviceSynchronize();
          if (sync_err != cudaSuccess && sync_err != cudaErrorMemoryAllocation) {
            cudaGetLastError();
            lock.lock();
            break;  // a real CUDA error; fall through to FATAL with original e
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
          lock.lock();
          // Flush our own pool again — a Free() may have landed in the cache
          // while we slept.
          ReleaseAllNoLock(false);
          e = contextHelper_->Malloc(&ret, roundSize);
          if (e == cudaErrorMemoryAllocation) {
            cudaGetLastError();
          }
          // Exponential backoff capped at 1s.
          backoff_ms = std::min<size_t>(backoff_ms * 2, 1000);
        }
      }
      if (failsafe && dev_type_ == Context::kGPU && e == cudaErrorMemoryAllocation) {
        // In failsafe mode, the only indication of the
        // failed allocation is a null dptr.  The used_memory_
        // should not grow.
        // Clear sticky cuda mem alloc error
        cudaGetLastError();
        ret       = nullptr;
        roundSize = 0;
        e         = cudaSuccess;
      }
#endif
      if (e) {
#if MXNET_USE_CUDA
        // Defense-in-depth for CUDA graphs (H2): a fresh cudaMalloc is illegal while
        // a stream is capturing and returns error 900 here. Surface an actionable
        // message instead of the cryptic "operation not permitted when stream is
        // capturing" -- some op in the captured segment requested a runtime-sized
        // allocation that warm-up could not predict.
        if (dev_type_ == Context::kGPU && e == cudaErrorStreamCaptureUnsupported) {
          cudaGetLastError();
          LOG(FATAL) << "Storage::Alloc of " << roundSize
                     << " bytes was attempted while a CUDA graph is capturing. An op in "
                        "the captured segment requested a fresh device allocation whose "
                        "size depends on runtime data (so warm-up could not pre-size it). "
                        "Exclude that op from CUDA graphs (FIsCUDAGraphsCompatible=false) "
                        "or pre-size its workspace.";
        }
#endif
        const std::string err(
#if MXNET_USE_CUDA
            dev_type_ == Context::kGPU ? cudaGetErrorString(static_cast<cudaError_t>(e)) :
#endif
                                         std::strerror(errno));

        // Include allocation context so a kernel that dies on this LOG(FATAL)
        // from a worker thread (the path Jupyter surfaces as
        // "DeadKernelError: Kernel died" with no traceback) leaves enough
        // breadcrumbs in stderr for the user to diagnose. d2l-ssd-bug.md
        // Issue 3 was a BERT NLI kernel that died after ~18 minutes with
        // no message; without context the user had nothing to act on.
        std::ostringstream context_msg;
        context_msg << "Memory allocation failed " << err << " (requested "
                    << roundSize << " bytes, pool used " << used_memory_;
#if MXNET_USE_CUDA
        if (dev_type_ == Context::kGPU) {
          size_t free_b = 0, total_b = 0;
          if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            context_msg << ", device free " << free_b << "/" << total_b;
          }
          if (oom_retries_ > 0) {
            context_msg << ", after " << oom_retries_ << " retries with "
                        << oom_backoff_ms_ << "ms initial backoff";
          }
        }
#endif
        context_msg << ")";
        LOG(FATAL) << context_msg.str();
      }
    }

    UNSET_DEVICE(device_store);

    used_memory_ += roundSize;
    handle->dptr = ret;
  } else {
    // Reusing memory
    auto ptr_syncobj = reuse_pool->back();
    handle->dptr     = ptr_syncobj.first;
    if (dev_type_ == Context::kGPU) {
      handle->sync_obj = ptr_syncobj.second;
#if MXNET_USE_CUDA
      for (auto ev : handle->sync_obj.events) {
        auto valid_ev = ev.lock();
        if (valid_ev) {
          MSHADOW_CUDA_CALL(cudaEventSynchronize(*valid_ev));
        }
      }
#endif
    }
    reuse_pool->pop_back();
  }
#if MXNET_USE_CUDA
  SET_GPU_PROFILER(profilerGPU, contextHelper_);
  if (profilerGPU) {
    if (reuse_pool)  // roundSize was not calculated
      roundSize = BucketingStrategy::RoundAllocSizeForBucket(bucket_id);

    // record the allocation event in the memory profiler
    if (!failsafe || handle->dptr != nullptr)
      profilerGPU->OnAlloc(*handle, roundSize, reuse_pool);
  }
#endif
}

/*!
 * \brief Base class for Rounding Method classes.
 */
class RoundHelper {
 public:
  virtual size_t get_size(size_t /*bucket*/) const {
    return 0;
  }
  virtual std::tuple<size_t, size_t> getContainerParam() const {
    return std::tuple<size_t, size_t>(0, 0);
  }

 protected:
  void InitRoundHelper(const char* dev_type) {
    const auto env_var = env_var_name(dev_type, pool_page_size);
    page_size_         = dmlc::GetEnv(env_var.c_str(), 4096);
    if (page_size_ < NDEV) {
      LOG(FATAL) << env_var << " cannot be set to a value smaller than " << NDEV << ". Got "
                 << page_size_ << ".";
    }
  }

  // page size
  size_t page_size_ = 0;

 private:
  // number of devices
  const size_t NDEV = 32;
};  // class RoundHelper

/*!
 * \brief Rounding method used by CPU/GPU mem pool.
 * Round up small allocs to multiple of page_size_ or large_alloc_round_size_
 */
class RoundMultiple : protected RoundHelper {
 protected:
  void InitRoundHelper(const char* dev_type) {
    RoundHelper::InitRoundHelper(dev_type);
    const auto env_var      = env_var_name(dev_type, large_alloc_size);
    large_alloc_round_size_ = dmlc::GetEnv(env_var.c_str(), 2 * 1024 * 1024);
    if (large_alloc_round_size_ <= 0) {
      LOG(FATAL) << env_var << " cannot be set to a value <= 0, found: " << large_alloc_round_size_;
    }
  }

  size_t RoundAllocSize(size_t size) const {
    // Round up small allocs to multiple of page_size_ to consolidate the pool lookups
    size = RoundToMultiple(size, page_size_);
    // To ensure proper freeing under some driver variants, make sure
    // large allocs entirely occupy their slabs, which cannot then be
    // locked by smaller permanent allocations sharing the slab.
    return size > large_alloc_round_size_ ? RoundToMultiple(size, large_alloc_round_size_) : size;
  }
  inline size_t get_bucket(size_t size) const {
    return RoundAllocSize(size);
  }
  inline size_t RoundAllocSizeForBucket(size_t bucket_id) const {
    return bucket_id;
  }

 private:
  // Round a value 'x' up to the next multiple of 'multiple'
  inline static size_t RoundToMultiple(size_t x, size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
  }

  // size that large allocations should be rounded to, for proper freeing.
  size_t large_alloc_round_size_;
};  // class RoundMultiple

/*!
 * \brief Rounding method used by CPU/GPU mem pool.
 *
 * This Rounding method uses a mixture of nearest pow2 (exponential) rounding and
 * nearest multiple (linear) rounding to help alleviate the memory allocation stress
 * in which the default naive exact-size-match pool falls short, such as in variable-length
 * input/output cases like RNN workloads.
 *
 * \param cutoff the cutoff at which rounding is switched from exponential to linear. It's set
 * through MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF / MXNET_CPU_MEM_POOL_ROUND_LINEAR_CUTOFF /
 * MXNET_CPU_PINNED_MEM_POOL_ROUND_LINEAR_CUTOFF environment variable.
 * Must be between 20 (1 MB) and 34 (16 GB).
 * Suppose the cutoff is X, the memory size buckets look like this:
 * exp2(0), exp2(1), ..., exp2(X), 2*exp2(X), 3*exp2(X), ...
 */
class RoundPower2 : public RoundHelper {
 public:
  size_t get_size(size_t bucket) const override {
    return bucket <= cut_off_ ? 1ul << bucket : (bucket - cut_off_ + 1) << cut_off_;
  }

 protected:
  void InitRoundHelper(const char* dev_type) {
    RoundHelper::InitRoundHelper(dev_type);
    const auto log_pager_size = common::ilog2ul(page_size_ - 1);
    if (page_size_ != 1ul << log_pager_size) {
      LOG(FATAL) << env_var_name(dev_type, pool_page_size)
                 << " must be a power of 2. Got: " << page_size_ << ".";
    }
    page_size_ = log_pager_size;

    const auto linear_cutoff = env_var_name(dev_type, round_linear_cutoff);
    cut_off_                 = dmlc::GetEnv(linear_cutoff.c_str(), 24);
    if (cut_off_ < 20 || cut_off_ > LOG2_MAX_MEM) {
      LOG(FATAL) << linear_cutoff << " cannot be set to a value "
                 << "smaller than 20 or greater than " << LOG2_MAX_MEM << ". Got: " << cut_off_
                 << ".";
    }
    if (cut_off_ < page_size_) {
      LOG(FATAL) << linear_cutoff << " cannot be set to a value smaller than log2 of "
                 << env_var_name(dev_type, pool_page_size) << ". Got: " << cut_off_ << " vs "
                 << page_size_ << ".";
    }
  }

  inline size_t get_bucket(size_t s) const {
    const size_t log_size = common::ilog2ul(s - 1);
    if (log_size > cut_off_)
      return div_pow2_round_up(s, cut_off_) - 1 + cut_off_;

    return std::max(log_size, page_size_);
  }

  inline size_t RoundAllocSizeForBucket(size_t bucket_id) const {
    return get_size(bucket_id);
  }
  inline size_t RoundAllocSize(size_t size) const {
    return get_size(get_bucket(size));
  }
  std::tuple<size_t, size_t> getContainerParam() const override {
    return std::make_tuple((1ul << (LOG2_MAX_MEM - cut_off_)) + cut_off_,
                           get_bucket(page_size_) - 1);
  }

 private:
  inline static int div_pow2_round_up(size_t s, int divisor_log2) {
    // (1025, 10) -> 2
    // (2048, 10) -> 2
    // (2049, 10) -> 3
    const size_t result = s >> divisor_log2;
    return static_cast<int>(result + (s > (result << divisor_log2) ? 1 : 0));
  }

  // log2 of maximum page size. 16GB
  const size_t LOG2_MAX_MEM = 34;
  // log2 of memory size before switching to exponential mode to linear mode
  size_t cut_off_ = 0;
};  // class RoundPower2

/*!
 * \brief Unordered map based storage container.
 *  The pointers to the portions of same rounded sizes memory
 *  allocated on CPU/GPU, are stored in separate vectors.
 *  These sizes are used as keys for accessing the vectors,
 *  which are the elements stored in an unordered map.
 */
class UnorderedMapContainer {
 protected:
  inline void InitContainer(const RoundHelper* p) {}
  inline void InsertInCache(size_t key, void* dptr, Storage::SyncObj sync_obj) {
    memory_pool_[key].emplace_back(dptr, sync_obj);
  }

  inline size_t BucketSize(size_t key) const {
    auto it = memory_pool_.find(key);
    return it == memory_pool_.end() ? 0 : it->second.size();
  }

  inline std::vector<std::pair<void*, Storage::SyncObj>>* GetMemStorage(size_t key) {
    auto&& reuse_it = memory_pool_.find(key);
    return reuse_it != memory_pool_.end() && reuse_it->second.size() ? &reuse_it->second : nullptr;
  }

  size_t ReleaseAllNoLock(const ContextHelper* contextHelper, const RoundHelper* /*rndHelper*/) {
    SET_GPU_PROFILER(profilerGPU, contextHelper);
    size_t released_memory = 0;
    for (auto&& i : memory_pool_) {
      for (auto&& j : i.second) {
        contextHelper->Free(j.first);
        GPU_PROFILER_ON_FREE(profilerGPU, j.first);
      }
      released_memory += i.first * i.second.size();
      i.second.clear();
    }
    memory_pool_.clear();
    return released_memory;
  }

 private:
  std::unordered_map<size_t, std::vector<std::pair<void*, Storage::SyncObj>>> memory_pool_;
};  // class UnorderedMapContainer

/*!
 * \brief Vector-container based storage container. It should be used ONLY with the RoundPower2.
 *  The pointers to the portions of same rounded size allocated on
 *  GPU/CPU/CPU_Pinned memory, are stored in separate vectors.
 *  The vectors themselves are stored in the vector-container and could
 *  be accessed by the indices calculated as a functions of rounded size
 *  (see description for RoundPower2 for more details)
 */
class VectorContainer {
 protected:
  inline void InitContainer(const RoundHelper* p) {
    size_t vector_size;
    std::tie(vector_size, first_bucket_) = p->getContainerParam();
    memory_pool_.resize(vector_size);
  }

  inline void InsertInCache(size_t idx, void* dptr, Storage::SyncObj sync_obj) {
    memory_pool_[idx].emplace_back(dptr, sync_obj);
  }

  inline size_t BucketSize(size_t idx) const {
    return idx < memory_pool_.size() ? memory_pool_[idx].size() : 0;
  }

  std::vector<std::pair<void*, Storage::SyncObj>>* GetMemStorage(size_t idx) {
    auto&& reuse_pool = memory_pool_[idx];
    return reuse_pool.size() ? &reuse_pool : nullptr;
  }

  size_t ReleaseAllNoLock(const ContextHelper* contextHelper, const RoundHelper* rndHelper) {
    SET_GPU_PROFILER(profilerGPU, contextHelper);
    size_t released_memory = 0;
    for (size_t i = first_bucket_; i < memory_pool_.size(); i++) {
      if (!memory_pool_[i].size())
        continue;

      for (auto& j : memory_pool_[i]) {
        contextHelper->Free(j.first);
        GPU_PROFILER_ON_FREE(profilerGPU, j.first);
      }
      released_memory += rndHelper->get_size(i) * memory_pool_[i].size();
      memory_pool_[i].clear();
    }
    return released_memory;
  }

 private:
  std::vector<std::vector<std::pair<void*, Storage::SyncObj>>> memory_pool_;
  size_t first_bucket_;
};  // class VectorContainer

// For backward compatibility, define previously used classes via new components.
// Just in case, if someone uses these classes in other places, besides
// the storage.cc, where the corresponding changes have already been made.
typedef PooledStorageManager<RoundMultiple, UnorderedMapContainer> GPUPooledStorageManager;
typedef PooledStorageManager<RoundPower2, VectorContainer> GPUPooledRoundedStorageManager;

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
