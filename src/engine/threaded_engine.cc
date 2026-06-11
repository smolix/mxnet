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
 * \file threaded_engine.cc
 * \brief implements base threaded engine.
 * \author Yutian Li
 */
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include "./threaded_engine.h"
#include "../common/cuda/utils.h"

namespace mxnet {
namespace engine {

namespace {

std::string ExceptionMessage(const std::exception_ptr& exception) {
  try {
    std::rethrow_exception(exception);
  } catch (const dmlc::Error& err) {
    return err.what();
  } catch (const std::exception& err) {
    return err.what();
  } catch (...) {
    return "unknown non-standard exception";
  }
}

void ThrowCollectedEngineExceptions(const std::vector<std::exception_ptr>& exceptions) {
  if (exceptions.empty()) {
    return;
  }
  if (exceptions.size() == 1) {
    std::rethrow_exception(exceptions.front());
  }
  std::ostringstream os;
  os << "Multiple asynchronous engine errors (" << exceptions.size() << "):";
  for (size_t i = 0; i < exceptions.size(); ++i) {
    os << "\n[" << i << "] " << ExceptionMessage(exceptions[i]);
  }
  throw dmlc::Error(os.str());
}

}  // namespace

#if ENGINE_DEBUG
std::atomic<std::size_t> OprBlock::counter{0};
std::atomic<std::size_t> VersionedVarBlock::counter{0};
std::atomic<std::size_t> ThreadedVar::counter{0};
std::atomic<std::size_t> ThreadedOpr::counter{0};
#endif  // ENGINE_DEBUG

ThreadedVar::ThreadedVar(VersionedVarBlock* head) : head_{head} {
#if ENGINE_DEBUG
  LOG(INFO) << __func__ << " " << ++counter;
#endif  // ENGINE_DEBUG
}

inline void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{mutex_};
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    CHECK_GE(num_pending_reads_, 0);
    // STATE CHANGE
    ++num_pending_reads_;
    // decrease wait counter
    opr_block->decr_wait();
  } else {
    auto&& new_var_block = VersionedVarBlock::New();
    CHECK(head_->next == nullptr);
    CHECK(head_->trigger == nullptr);
    CHECK(!head_->write);
    // append things to next.
    head_->next    = new_var_block;
    head_->trigger = opr_block;
    head_          = new_var_block;
  }
}

inline void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
  auto&& new_var_block = VersionedVarBlock::New();
  std::lock_guard<std::mutex> lock{mutex_};
  // invariant.
  CHECK(head_->next == nullptr);
  CHECK(head_->trigger == nullptr);
  CHECK(!head_->write);
  // attach to head.
  head_->next    = new_var_block;
  head_->trigger = opr_block;
  head_->write   = true;

  // check if it is ready to write
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    pending_write_ = head_;
    CHECK_GE(num_pending_reads_, 0);
    if (num_pending_reads_ == 0) {
      // STATE CHANGE
      opr_block->decr_wait();
      num_pending_reads_ = kWriteTriggered;
    }
  } else {
    CHECK_NE(num_pending_reads_, 0);
  }
  head_ = new_var_block;
}

template <typename Dispatcher>
inline void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
  OprBlock* trigger = nullptr;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{mutex_};
    CHECK_GT(num_pending_reads_, 0);

    if (--num_pending_reads_ == 0) {
      if (pending_write_ != nullptr) {
        // STATE CHANGE
        trigger            = pending_write_->trigger;
        num_pending_reads_ = kWriteTriggered;
      }
    }
  }
  if (trigger != nullptr && trigger->decr_wait() == 0) {
    dispatcher(trigger);
  }
}

template <typename Dispatcher>
inline bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  // this is lock scope
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  OprBlock* trigger_write = nullptr;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    // invariants
    CHECK(head_->next == nullptr);
    CHECK_NE(pending_write_, nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // increment version number
    ++version_;

    // really delete
    if (to_delete_) {
      VersionedVarBlock* head = pending_write_->next;
      VersionedVarBlock::Delete(pending_write_);
      // XOP23: promote from `assert` to CHECK so the linked-list integrity
      // invariant ('the pending_write_ head's next is also our tracked head')
      // is not silently compiled out in release builds.  A stripped assert
      // here would let us Delete(head) on a mis-aligned head and corrupt the
      // var pool.
      CHECK(head_ == head)
          << "ThreadedVar pending-write head/head_ chain invariant violated"
          << " in to_delete_ path";
      VersionedVarBlock::Delete(head);
      return true;
    }
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    // reset to 0 pending reads
    num_pending_reads_ = 0;
    while (end_of_read_chain != head_ && end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    if (end_of_read_chain == head_) {
      pending_write_ = nullptr;
    } else {
      // check if there is pending reads, if not trigger write
      CHECK(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        // mark write as already activated in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write      = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not rely on these two variables.
  // The linked list \in [old_pending_write, end_of_read_chain)
  // is already detached from this Var.
  // So it is safe to modify these
  VersionedVarBlock* cur_head = old_pending_write->next;
  VersionedVarBlock::Delete(old_pending_write);
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (cur_head->trigger->decr_wait() == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head  = cur_head->next;
    CHECK(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  if (trigger_write != nullptr && trigger_write->decr_wait() == 0) {
    dispatcher(trigger_write);
  }
  return false;
}

inline void ThreadedVar::SetToDelete() {
  std::lock_guard<std::mutex> lock{mutex_};
  to_delete_ = true;
}

inline bool ThreadedVar::ready_to_read() {
  std::lock_guard<std::mutex> lock{mutex_};
  return this->is_ready_to_read();
}

inline size_t ThreadedVar::version() {
  std::lock_guard<std::mutex> lock{mutex_};
  return this->version_;
}

// implementation of threaded engine
ThreadedVar* ThreadedEngine::NewVariable() {
  return ThreadedVar::New(VersionedVarBlock::New());
}

ThreadedOpr* ThreadedEngine::NewOperator(ThreadedEngine::AsyncFn fn,
                                         std::vector<VarHandle> const& const_vars,
                                         std::vector<VarHandle> const& mutable_vars,
                                         FnProperty prop,
                                         const char* opr_name,
                                         bool wait) {
  auto ret      = ThreadedOpr::New();
  ret->opr_name = opr_name ? std::string(opr_name) : std::string();
  ret->fn       = std::move(fn);
  ret->prop     = prop;
  ret->const_vars.resize(const_vars.size());
  ret->mutable_vars.resize(mutable_vars.size());
  ret->wait = wait;
  std::transform(
      const_vars.begin(), const_vars.end(), ret->const_vars.begin(), ThreadedVar::CastFromBase);
  std::transform(mutable_vars.begin(),
                 mutable_vars.end(),
                 ret->mutable_vars.begin(),
                 ThreadedVar::CastFromBase);
  // L13: always-on. Duplicate vars otherwise surface as a nondeterministic
  // engine hang in release builds (ENGINE_DEBUG=0) instead of a clear fatal.
  CheckDuplicate(const_vars, mutable_vars);
  return ret;
}

void ThreadedEngine::CheckDuplicate(std::vector<VarHandle> const& const_vars,
                                    std::vector<VarHandle> const& mutable_vars) {
  // Allocation-free O(n^2) duplicate detection. Per-op var counts are tiny
  // (typically a handful), so this is cheap enough to run on every push; the
  // previous sort-based version copied both vectors and so was gated to debug
  // builds (L13). Behaviour is identical: a duplicate var is a hard error.
  const size_t use_size    = const_vars.size();
  const size_t mutate_size = mutable_vars.size();
  for (size_t i = 0; i < use_size; ++i) {
    for (size_t j = i + 1; j < use_size; ++j) {
      if (const_vars[i] == const_vars[j])
        LOG(FATAL) << "duplicate items found in `const_vars`";
    }
  }
  for (size_t i = 0; i < mutate_size; ++i) {
    for (size_t j = i + 1; j < mutate_size; ++j) {
      if (mutable_vars[i] == mutable_vars[j])
        LOG(FATAL) << "duplicate items found in `mutable_vars`";
    }
  }
  for (size_t i = 0; i < use_size; ++i) {
    for (size_t j = 0; j < mutate_size; ++j) {
      if (const_vars[i] == mutable_vars[j])
        LOG(FATAL) << "duplicate items found between `const_vars` and `mutable_vars`";
    }
  }
}

void ThreadedEngine::DeleteOperator(OprHandle op) {
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  std::vector<VarHandle> deps;
  deps.reserve(threaded_opr->const_vars.size() + threaded_opr->mutable_vars.size());
  deps.insert(deps.end(), threaded_opr->const_vars.begin(), threaded_opr->const_vars.end());
  deps.insert(deps.end(), threaded_opr->mutable_vars.begin(), threaded_opr->mutable_vars.end());
  this->PushAsync(
      [threaded_opr](RunContext, CallbackOnStart on_start, CallbackOnComplete on_complete) {
        on_start();
        ThreadedOpr::Delete(threaded_opr);
        on_complete();
      },
      Context::CPU(),
      {},
      deps,
      FnProperty::kDeleteVar,
      0,
      "DeleteOperator");
}

void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority, bool profiling) {
  BulkFlush();
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  if (profiling) {
    threaded_opr->opr_name =
        profiler::CustomOpProfiler::Get()->GenerateDisplayName(threaded_opr->opr_name.c_str());
  }
  OprBlock* opr_block = OprBlock::New();
  opr_block->opr      = threaded_opr;

  opr_block->wait.store(
      static_cast<int>(threaded_opr->const_vars.size() + threaded_opr->mutable_vars.size() + 1));
  opr_block->ctx       = exec_ctx;
  opr_block->priority  = priority;
  opr_block->profiling = profiling;
  ++pending_;
  // Add read dependencies.
  for (auto&& i : threaded_opr->const_vars) {
    i->AppendReadDependency(opr_block);
  }
  // Add write dependencies.
  for (auto&& i : threaded_opr->mutable_vars) {
    i->AppendWriteDependency(opr_block);
  }
  if (opr_block->decr_wait() == 0) {
    this->PushToExecute(opr_block, true);
  }
}

void ThreadedEngine::PushAsync(AsyncFn fn,
                               Context exec_ctx,
                               std::vector<VarHandle> const& const_vars,
                               std::vector<VarHandle> const& mutable_vars,
                               FnProperty prop,
                               int priority,
                               const char* opr_name,
                               bool wait) {
#if MXNET_USE_CUDA
  if (exec_ctx.dev_mask() == gpu::kDevMask) {
    if (device_count_ < 0) {
      int tmp = -1;
      cudaGetDeviceCount(&tmp);
      device_count_ = tmp;
      CHECK_GT(device_count_, 0) << "GPU usage requires at least 1 GPU";
    }
    CHECK_LT(exec_ctx.dev_id, device_count_)
        << "Invalid GPU Id: " << exec_ctx.dev_id
        << ", Valid device id should be less than device_count: " << device_count_;
  }
#endif
  const bool profiling = profiler_->IsProfiling(profiler::Profiler::kImperative);
  ThreadedOpr* opr     = NewOperator(std::move(fn), const_vars, mutable_vars, prop, opr_name, wait);
  opr->temporary       = true;
  Push(opr, exec_ctx, priority, profiling);
}

void ThreadedEngine::PushSync(SyncFn exec_fn,
                              Context exec_ctx,
                              std::vector<VarHandle> const& const_vars,
                              std::vector<VarHandle> const& mutable_vars,
                              FnProperty prop,
                              int priority,
                              const char* opr_name) {
  if (!bulk_size() || prop != FnProperty::kNormal || priority) {
    this->PushAsync(
        [exec_fn](RunContext ctx, CallbackOnStart on_start, CallbackOnComplete on_complete) {
          on_start();
          exec_fn(ctx);
          on_complete();
        },
        exec_ctx,
        const_vars,
        mutable_vars,
        prop,
        priority,
        opr_name);
    return;
  }

  const BulkStatus& bulk_status = *BulkStatusStore::Get();
  if (bulk_status.count && exec_ctx != bulk_status.ctx)
    BulkFlush();
  BulkAppend(exec_fn, exec_ctx, const_vars, mutable_vars);
}

void ThreadedEngine::DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) {
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  this->PushAsync(
      [delete_fn, threaded_var](
          RunContext ctx, CallbackOnStart on_start, CallbackOnComplete on_complete) {
        // Mark variable as orphan,
        // so during `ThreadedEngine::OnComplete` it could be recycled.
        on_start();
        threaded_var->SetToDelete();
        delete_fn(ctx);
        on_complete();
      },
      exec_ctx,
      {},
      {var},
      FnProperty::kDeleteVar,
      0,
      "DeleteVariable");
}

void ThreadedEngine::WaitForVar(VarHandle var) {
  // A6/A7 diagnostic (apache/mxnet#19994, #18090): MXNET_ENGINE_DIAG=1 enables
  // a 30-second no-progress watchdog.  On timeout it logs the pending count and
  // var pointer so the caller can identify the stuck variable.  The wait
  // continues after logging (it does not abort) to avoid masking transient
  // slowdowns under heavy load.
  static const bool engine_diag = dmlc::GetEnv("MXNET_ENGINE_DIAG", false);
  static const int diag_timeout_s = dmlc::GetEnv("MXNET_ENGINE_DIAG_TIMEOUT_S", 30);

  BulkFlush();
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  if (threaded_var->ready_to_read()) {
    ThrowException(threaded_var);
    return;
  }
  if (engine_info_) {
    LOG(INFO) << "Wait for " << threaded_var;
    debug_wait_var_ = threaded_var;
  }
  auto done = std::make_shared<std::atomic<bool>>(false);
  this->PushAsync(
      [this, done](RunContext, CallbackOnStart on_start, CallbackOnComplete on_complete) {
        on_start();
        if (engine_info_) {
          LOG(INFO) << "Sync is executed";
        }
        {
          std::unique_lock<std::mutex> lock{finished_m_};
          done->store(true);
        }
        finished_cv_.notify_all();
        if (engine_info_) {
          LOG(INFO) << "Sync is notified";
        }
        on_complete();
      },
      Context::CPU(),
      {var},
      {},
      FnProperty::kNormal,
      0,
      "WaitForVar",
      true);
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    if (engine_diag) {
      // Watchdog loop: wake every diag_timeout_s seconds and log if still stuck.
      while (!finished_cv_.wait_for(lock,
                                    std::chrono::seconds(diag_timeout_s),
                                    [this, done]() { return done->load() || kill_.load(); })) {
        LOG(WARNING) << "[MXNET_ENGINE_DIAG] WaitForVar timeout after " << diag_timeout_s
                     << "s: var=" << threaded_var
                     << " pending_ops=" << pending_.load()
                     << " shutdown_phase=" << shutdown_phase_.load()
                     << " kill=" << kill_.load()
                     << ". Engine may be deadlocked. Set MXNET_ENGINE_TYPE=NaiveEngine "
                        "to debug synchronously.";
      }
    } else {
      finished_cv_.wait(lock, [this, done]() { return done->load() || kill_.load(); });
    }
  }

  ThrowException(threaded_var);
}

void ThreadedEngine::WaitForAll() {
  // A6/A7: same watchdog as WaitForVar (apache/mxnet#19994, #18090).
  static const bool engine_diag_all = dmlc::GetEnv("MXNET_ENGINE_DIAG", false);
  static const int diag_timeout_s_all = dmlc::GetEnv("MXNET_ENGINE_DIAG_TIMEOUT_S", 30);

  BulkFlush();
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    if (engine_diag_all) {
      while (!finished_cv_.wait_for(lock,
                                    std::chrono::seconds(diag_timeout_s_all),
                                    [this]() { return pending_.load() == 0 || kill_.load(); })) {
        LOG(WARNING) << "[MXNET_ENGINE_DIAG] WaitForAll timeout after " << diag_timeout_s_all
                     << "s: pending_ops=" << pending_.load()
                     << " shutdown_phase=" << shutdown_phase_.load()
                     << " kill=" << kill_.load()
                     << ". Engine may be deadlocked.";
      }
    } else {
      finished_cv_.wait(lock, [this]() { return pending_.load() == 0 || kill_.load(); });
    }
  }

  std::vector<std::exception_ptr> exceptions_to_rethrow;
  {
    std::lock_guard<std::mutex> exception_lock(exception_m_);
    if (!global_exception_refs_.empty()) {
      // iterate through all exception refs
      for (const auto& global_exception_ref : global_exception_refs_) {
        if (*global_exception_ref != nullptr) {
          exceptions_to_rethrow.push_back(*global_exception_ref);
        }
        // clear exceptions, WaitToRead following WaitForAll shouldn't throw
        *global_exception_ref = nullptr;
      }
      // A waitall following a waitall shouldn't throw any exceptions
      global_exception_refs_.clear();
    }
  }
  ThrowCollectedEngineExceptions(exceptions_to_rethrow);
}

inline void ThreadedEngine::OnComplete(ThreadedOpr* threaded_opr) {
  bool is_temporary_opr = threaded_opr->temporary;
  if (threaded_opr->opr_exception != nullptr && threaded_opr->mutable_vars.empty()) {
    std::lock_guard<std::mutex> exception_lock(exception_m_);
    global_exception_refs_.push_back(threaded_opr->opr_exception);
  }
  // Mark complete for read variables
  for (auto&& i : threaded_opr->const_vars) {
    i->CompleteReadDependency([this](OprBlock* opr) { this->PushToExecute(opr, false); });
  }
  // Mark complete for write variables.
  for (auto&& i : threaded_opr->mutable_vars) {
    SetVarExceptionAndAddToGlobal(i, threaded_opr->opr_exception);
    const bool debug_info = (engine_info_ && debug_wait_var_ == i);
    if (debug_info) {
      LOG(INFO) << "Complete write dep for " << i;
    }
    const bool to_delete = i->CompleteWriteDependency([this, debug_info](OprBlock* opr) {
      if (debug_info) {
        LOG(INFO) << "PushToExecute " << opr;
        debug_push_opr_ = opr;
      }
      this->PushToExecute(opr, false);
      if (debug_info) {
        LOG(INFO) << "Fin PushToExecute " << opr;
      }
    });
    if (to_delete) {
#if MXNET_USE_CUDA
      auto& sync_obj = i->sync_object;
      {
        std::lock_guard<std::mutex> l(sync_obj.mutex);
        sync_obj.reader_events.clear();
        sync_obj.writer_event.clear();
      }
#endif
      ThreadedVar::Delete(i);
    }
  }
  // The function been pushed from `ThreadedEngine::DeleteOperator`
  // could execute right after we mark all vars as complete, so if
  // threaded_opr is not temporary, its value is not reliable
  // anymore start from here.
  // L12: pending_ is atomic, so the decrement needs no lock on the common path.
  // Only the 0-transition must synchronize with WaitForAll: acquiring finished_m_
  // there closes the lost-wakeup window between a waiter's predicate check and its
  // cv.wait(). For npending > 0 we skip the mutex entirely.
  const int npending = --pending_;
  CHECK_GE(npending, 0);
  if (npending == 0) {
    std::lock_guard<std::mutex> lock{finished_m_};
    finished_cv_.notify_all();
  }

  // delete operator if it is temperory
  if (is_temporary_opr) {
    ThreadedOpr::Delete(threaded_opr);
  }
}

inline void ThreadedEngine::ThrowException(ThreadedVar* threaded_var) {
  std::exception_ptr tmp = ClearVarException(threaded_var);
  if (tmp != nullptr) {
    std::rethrow_exception(tmp);
  }
  return;
}

void ThreadedEngine::Throw(VarHandle var) {
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  ThrowException(threaded_var);
}

void ThreadedEngine::OnCompleteStatic(Engine* engine, void* opr_block_, const dmlc::Error* error) {
  OprBlock* opr_block       = static_cast<OprBlock*>(opr_block_);
  ThreadedOpr* threaded_opr = opr_block->opr;
  if (error != nullptr) {
    auto ex_p                   = std::make_exception_ptr(*error);
    threaded_opr->opr_exception = std::make_shared<std::exception_ptr>(ex_p);
  }
  if (opr_block->profiling && threaded_opr->opr_name.size()) {
    // record operator end timestamp
    opr_block->opr_profile->stop();
  }
  static_cast<ThreadedEngine*>(engine)->OnComplete(threaded_opr);
  OprBlock::Delete(opr_block);
}

void ThreadedEngine::OnStartStatic(Engine* engine, void* opr_block, const dmlc::Error* error) {
  // no-op
}

#if MXNET_USE_CUDA
static inline void AddEventHelper(std::unordered_map<cudaStream_t, EventInfo>* events_per_stream,
                                  const EventInfo& cuda_event) {
  auto event_stream = cuda_event.stream;
  if (events_per_stream->count(event_stream) > 0) {
    if ((*events_per_stream)[event_stream].pool_index < cuda_event.pool_index) {
      (*events_per_stream)[event_stream] = cuda_event;
    }
  } else {
    (*events_per_stream).emplace(event_stream, cuda_event);
  }
}

static inline bool IsEngineAsync() {
  // L14: must match CreateEngine()'s parsing exactly -- the "Async" tag is only
  // honored as a SUFFIX there (it is stripped to pick the base engine). Matching
  // it as a bare substring here diverged: a value like "AsyncThreadedEngine"
  // builds a non-async engine yet would report async, enabling the GPU
  // dependency path against an engine that never set it up.
  std::string type = dmlc::GetEnv("MXNET_ENGINE_TYPE", std::string(""));
  const std::string async_engine_tag("Async");
  auto tag_pos = type.find(async_engine_tag);
  return tag_pos != std::string::npos &&
         tag_pos + async_engine_tag.length() == type.length();
}

void ThreadedEngine::OnStartCPU(Engine* engine, void* opr_block, const dmlc::Error* error) {
  static bool use_new_dep_engine = IsEngineAsync();
  if (!use_new_dep_engine) {
    return;
  }
  ThreadedOpr* threaded_opr = static_cast<OprBlock*>(opr_block)->opr;
  std::unordered_map<cudaStream_t, EventInfo> event_per_stream;
  for (auto* read_var : threaded_opr->const_vars) {
    auto& sync_obj = read_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    auto& reader_events = sync_obj.reader_events;
    // check for expired events and delete them
    reader_events.erase(std::remove_if(reader_events.begin(),
                                       reader_events.end(),
                                       [&](const EventInfo e_i) { return e_i.event.expired(); }),
                        reader_events.end());
    for (auto& cuda_event : reader_events) {
      AddEventHelper(&event_per_stream, cuda_event);
    }
    if (!sync_obj.writer_event.empty()) {
      if (sync_obj.writer_event[0].event.expired()) {
        sync_obj.writer_event.clear();
      } else {
        AddEventHelper(&event_per_stream, sync_obj.writer_event[0]);
      }
    }
  }

  for (auto* write_var : threaded_opr->mutable_vars) {
    auto& sync_obj = write_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    auto& reader_events = sync_obj.reader_events;
    // check for expired events and delete them
    reader_events.erase(std::remove_if(reader_events.begin(),
                                       reader_events.end(),
                                       [&](const EventInfo e_i) { return e_i.event.expired(); }),
                        reader_events.end());
    for (auto& cuda_event : reader_events) {
      AddEventHelper(&event_per_stream, cuda_event);
    }
    if (!sync_obj.writer_event.empty()) {
      if (sync_obj.writer_event[0].event.expired()) {
        sync_obj.writer_event.clear();
      } else {
        AddEventHelper(&event_per_stream, sync_obj.writer_event[0]);
      }
    }
  }
  for (auto event : event_per_stream) {
    const EventInfo& ei = event.second;
    // If the pooled event slot has been lapped (reused for a newer, unrelated
    // record), the cached event no longer reflects the op we depend on. Wait on
    // the recorded stream directly instead -- correct (it drains that op too),
    // just coarser. Otherwise wait on the specific event.
    if (ei.pool != nullptr && ei.pool->IsLapped(ei.pool_index)) {
      MSHADOW_CUDA_CALL(cudaStreamSynchronize(ei.stream));
    } else if (auto ev = ei.event.lock()) {
      MSHADOW_CUDA_CALL(cudaEventSynchronize(*ev));
      // TOCTOU guard (see OnStartGPU): if the slot was lapped between the check
      // and the wait, host-sync the recorded stream as a correct backstop.
      if (ei.pool != nullptr && ei.pool->IsLapped(ei.pool_index)) {
        MSHADOW_CUDA_CALL(cudaStreamSynchronize(ei.stream));
      }
    }
  }
}

void ThreadedEngine::OnStartGPU(Engine* engine, void* sync_info, const dmlc::Error* error) {
  static bool use_new_dep_engine = IsEngineAsync();
  if (!use_new_dep_engine) {
    return;
  }
  auto* info = reinterpret_cast<GPUWorkerSyncInfo*>(sync_info);
  CHECK(info->stream != nullptr);
  auto* worker_stream       = reinterpret_cast<mshadow::Stream<gpu>*>(info->stream);
  ThreadedOpr* threaded_opr = static_cast<OprBlock*>(info->opr_block)->opr;
  std::unordered_map<cudaStream_t, EventInfo> event_per_stream;
  for (auto* read_var : threaded_opr->const_vars) {
    auto& sync_obj = read_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    auto& reader_events = sync_obj.reader_events;
    // check for expired events and delete them
    reader_events.erase(std::remove_if(reader_events.begin(),
                                       reader_events.end(),
                                       [&](const EventInfo e_i) { return e_i.event.expired(); }),
                        reader_events.end());
    for (auto& writer : sync_obj.writer_event) {
      if (writer.event.expired()) {
        sync_obj.writer_event.clear();
        break;
      }
      if (writer.stream != worker_stream->stream_) {
        // if there is already a reader on the same stream as us,
        // it already synced with that writer and we can rely on
        // the ongoing sync
        bool found = false;
        for (const auto& reader : reader_events) {
          if (reader.stream == worker_stream->stream_) {
            found = true;
            break;
          }
        }
        if (!found) {
          AddEventHelper(&event_per_stream, writer);
        }
      }
    }
  }
  for (auto* write_var : threaded_opr->mutable_vars) {
    auto& sync_obj = write_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    // check for expired events and delete them
    auto& reader_events = sync_obj.reader_events;
    reader_events.erase(std::remove_if(reader_events.begin(),
                                       reader_events.end(),
                                       [&](const EventInfo e_i) { return e_i.event.expired(); }),
                        reader_events.end());
    // if there are some readers, we wait for them
    for (auto& cuda_event : reader_events) {
      if (worker_stream->stream_ != cuda_event.stream) {
        AddEventHelper(&event_per_stream, cuda_event);
      }
    }
    if (!sync_obj.writer_event.empty()) {
      if (sync_obj.writer_event[0].event.expired()) {
        sync_obj.writer_event.clear();
      } else {
        if (worker_stream->stream_ != sync_obj.writer_event[0].stream) {
          AddEventHelper(&event_per_stream, sync_obj.writer_event[0]);
        }
      }
    }
  }
  for (auto event : event_per_stream) {
    const EventInfo& ei = event.second;
    // Lapped slot (reused for a newer record): the cached event no longer
    // tracks our dependency -- and since the pool owns the event the weak_ptr
    // never expires to signal this. Fall back to a host sync of the recorded
    // stream (correct: it drains the depended-on op) instead of a device-side
    // wait on the wrong (reused) event, which would under-synchronize.
    if (ei.pool != nullptr && ei.pool->IsLapped(ei.pool_index)) {
      MSHADOW_CUDA_CALL(cudaStreamSynchronize(ei.stream));
    } else if (auto ev = ei.event.lock()) {
      MSHADOW_CUDA_CALL(cudaStreamWaitEvent(worker_stream->stream_, *ev, 0));
      // TOCTOU guard: another worker may have lapped the slot (re-recording the
      // event for an unrelated op) between the IsLapped() check above and the
      // wait we just queued. If so, the queued wait may target the wrong record;
      // add the coarse-but-correct host sync of the recorded stream as a backstop.
      if (ei.pool != nullptr && ei.pool->IsLapped(ei.pool_index)) {
        MSHADOW_CUDA_CALL(cudaStreamSynchronize(ei.stream));
      }
    }
  }
}

void ThreadedEngine::OnCompleteGPU(Engine* engine, void* sync_info, const dmlc::Error* error) {
  auto* info = reinterpret_cast<GPUWorkerSyncInfo*>(sync_info);
  CHECK(info->stream != nullptr);

  auto* worker_stream            = reinterpret_cast<mshadow::Stream<gpu>*>(info->stream);
  static bool use_new_dep_engine = IsEngineAsync();

  if (!use_new_dep_engine) {
    worker_stream->Wait();
    ThreadedEngine::OnCompleteStatic(engine, info->opr_block, error);
    GPUWorkerSyncInfo::Delete(info);
    return;
  }

  ThreadedOpr* threaded_opr    = static_cast<OprBlock*>(info->opr_block)->opr;
  auto* event_pool             = static_cast<CUDAEventPool*>(info->event_pool);
  auto [event, event_pool_idx] = event_pool->GetNextEvent();  // NOLINT(*)
  auto ev                      = event.lock();
  MSHADOW_CUDA_CALL(cudaEventRecord(*ev, worker_stream->stream_));
  for (auto* read_var : threaded_opr->const_vars) {
    auto& sync_obj = read_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    // If some reader event is already recorded on the same stream,
    // we want to replace ourselves by it
    size_t i;
    for (i = 0; i < sync_obj.reader_events.size(); ++i) {
      auto stream = sync_obj.reader_events[i].stream;
      if (stream == worker_stream->stream_) {
        sync_obj.reader_events[i].event      = event;
        sync_obj.reader_events[i].pool_index = event_pool_idx;
        sync_obj.reader_events[i].pool       = event_pool;
        break;
      }
    }
    if (i == sync_obj.reader_events.size()) {
      sync_obj.reader_events.push_back({event, worker_stream->stream_, event_pool_idx, event_pool});
    }
  }

  for (auto* write_var : threaded_opr->mutable_vars) {
    auto& sync_obj = write_var->sync_object;
    std::lock_guard<std::mutex> l(sync_obj.mutex);
    sync_obj.reader_events.clear();
    sync_obj.writer_event.clear();
    sync_obj.writer_event.push_back({event, worker_stream->stream_, event_pool_idx, event_pool});
  }

  ThreadedEngine::OnCompleteStatic(engine, info->opr_block, error);
  GPUWorkerSyncInfo::Delete(info);
}
#endif

}  // namespace engine
}  // namespace mxnet
