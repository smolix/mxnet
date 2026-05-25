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

#ifndef MXNET_COMMON_DENORMS_H_
#define MXNET_COMMON_DENORMS_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// FTZ/DAZ only applies to SSE and AVX instructions.
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define MXNET_SUPPORT_FTZ_DAZ 1
#else
#define MXNET_SUPPORT_FTZ_DAZ 0
#endif

#if MXNET_SUPPORT_FTZ_DAZ
#include <immintrin.h>
#include <xmmintrin.h>
#endif
#if MXNET_SUPPORT_FTZ_DAZ && !defined(_MSC_VER)
#include <x86intrin.h>
#endif

namespace mxnet {
namespace common {
namespace denorms {

inline std::atomic<int>& ConfiguredFlushDenorms() {
  // -1 means no explicit API setting has been requested in this process.
  static std::atomic<int> configured{-1};
  return configured;
}

#if MXNET_SUPPORT_FTZ_DAZ
inline bool IsDazFlagAvailable() {
  static const bool available = []() {
    // Intel 64 and IA-32 Architectures Software Developer's Manual: Vol. 1
    // "Checking for the DAZ Flag in the MXCSR Register"
    constexpr unsigned int mxcsr_mask_offset = 28;
    constexpr unsigned int daz_flag_offset   = 5;
    constexpr unsigned int fxsave_req_bytes  = 512;

    char* fxsave_area_ptr = reinterpret_cast<char*>(std::malloc(fxsave_req_bytes));
    std::memset(fxsave_area_ptr, 0, fxsave_req_bytes);
    _fxsave(fxsave_area_ptr);

    char* mxcsr_mask_ptr = fxsave_area_ptr + mxcsr_mask_offset;
    uint32_t mxcsr_mask  = *(reinterpret_cast<uint32_t*>(mxcsr_mask_ptr));
    std::free(fxsave_area_ptr);
    return ((mxcsr_mask >> daz_flag_offset) & 0x1) != 0;
  }();
  return available;
}
#endif

inline bool ApplyFlushDenormsToCurrentThread(bool value) {
#if MXNET_SUPPORT_FTZ_DAZ
  const unsigned int daz_state = value ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF;
  const unsigned int ftz_state = value ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF;
  const bool prev_state        = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
  _MM_SET_FLUSH_ZERO_MODE(ftz_state);

  // DAZ is a reserved bit on some CPUs. Writing it when unsupported can fault.
  if (IsDazFlagAvailable()) {
    _MM_SET_DENORMALS_ZERO_MODE(daz_state);
  }
  return prev_state;
#else
  return false;
#endif
}

inline bool ConfigureFlushDenorms(bool value) {
#if MXNET_SUPPORT_FTZ_DAZ
  const int previous = ConfiguredFlushDenorms().exchange(value ? 1 : 0);
  return previous < 0 ? false : previous != 0;
#else
  return false;
#endif
}

inline bool SetFlushDenorms(bool value) {
#if MXNET_SUPPORT_FTZ_DAZ
  ConfiguredFlushDenorms().store(value ? 1 : 0);
  return ApplyFlushDenormsToCurrentThread(value);
#else
  return false;
#endif
}

inline void ApplyConfiguredFlushDenormsToCurrentThread() {
#if MXNET_SUPPORT_FTZ_DAZ
  const int configured = ConfiguredFlushDenorms().load();
  if (configured >= 0) {
    ApplyFlushDenormsToCurrentThread(configured != 0);
  }
#endif
}

}  // namespace denorms
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_COMMON_DENORMS_H_
