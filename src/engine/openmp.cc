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
#include <dmlc/omp.h>
#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <climits>
#include "./openmp.h"

namespace mxnet {
namespace engine {

#if defined(__i386__) || defined(_M_X86) || defined(_M_X64) || defined(__x86_64__)
#define ARCH_IS_INTEL_X86
#endif

static inline bool is_env_set(const char* var) {
  return dmlc::GetEnv(var, INT_MIN) != INT_MIN;
}

OpenMP* OpenMP::Get() {
  static OpenMP openMP;
  return &openMP;
}

OpenMP::OpenMP() : omp_num_threads_set_in_environment_(is_env_set("OMP_NUM_THREADS")) {
#ifdef _OPENMP
  initialize_process();
  const int max = dmlc::GetEnv("MXNET_OMP_MAX_THREADS", INT_MIN);
  if (max != INT_MIN) {
    set_thread_max(max);
  } else {
    if (!omp_num_threads_set_in_environment_) {
      int thread_max = omp_get_num_procs();
#ifdef ARCH_IS_INTEL_X86
      thread_max >>= 1;
#endif
      set_thread_max(thread_max);
      omp_set_num_threads(thread_max);
    } else {
      set_thread_max(omp_get_max_threads());
    }
  }
#else
  set_enabled(false);
  set_thread_max(1);
#endif
}

void OpenMP::initialize_process() {
#ifdef _OPENMP
  omp_get_num_procs();  // will force OpenMP to be initialized
#endif
}

void OpenMP::on_start_worker_thread(bool use_omp) {
#ifdef _OPENMP
  if (!omp_num_threads_set_in_environment_) {
    omp_set_num_threads(use_omp ? GetRecommendedOMPThreadCount(true) : 1);
  }
#endif
}

void OpenMP::set_reserve_cores(int cores) {
  CHECK_GE(cores, 0);
  reserve_cores_.store(cores, std::memory_order_relaxed);
#ifdef _OPENMP
  const int thread_max = this->thread_max();
  if (cores >= thread_max) {
    omp_set_num_threads(1);
  } else {
    omp_set_num_threads(thread_max - cores);
  }
#endif
}

int OpenMP::GetRecommendedOMPThreadCount(bool exclude_reserved) const {
#ifdef _OPENMP
  if (enabled()) {
    // OMP_NUM_THREADS was set in the environment at the time of static initialization
    if (omp_num_threads_set_in_environment_) {
      return omp_get_max_threads();
    }
    int thread_count = omp_get_max_threads();
    if (exclude_reserved) {
      const int reserve_cores = this->reserve_cores();
      if (reserve_cores >= thread_count) {
        thread_count = 1;
      } else {
        thread_count -= reserve_cores;
      }
    }
    // Check that OMP doesn't suggest more than our 'omp_thread_max_' value
    const int thread_max = this->thread_max();
    if (!thread_max || thread_count < thread_max) {
      return thread_count;
    }
    return thread_max;
  } else {
    return 1;
  }
#else
  return 1;
#endif
}

OpenMP* __init_omp__ = OpenMP::Get();

}  // namespace engine
}  // namespace mxnet
