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

#ifndef MXNET_COMMON_CUDA_RTC_HALF_INL_H_
#define MXNET_COMMON_CUDA_RTC_HALF_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char fp16_support_string[] = R"code(
struct __align__(2) __half {
  __host__ __device__ __half() : __x(0) { }
  unsigned short __x;
};
/* Definitions of intrinsics */
__device__ inline __half __float2half(const float f) {
  __half val;
 asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.__x) : "f"(f));
  return val;
}
__device__ inline float __half2float(const __half h) {
  float val;
 asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h.__x));
  return val;
}

typedef __half half;

template <typename DType>
struct AccType {
  using type = DType;

  __device__ static inline type from(const DType& val) {
    return val;
  }

  __device__ static inline DType to(type val) {
    return val;
  }

};

template<>
struct AccType<half> {
  using type = float;

  __device__ static inline type from(const half& val) {
    return __half2float(val);
  }

  __device__ static inline half to(type val) {
    return __float2half(val);
  }
};

/* bfloat16: stores the high 16 bits of an fp32 value. */
struct __align__(2) __bfloat16 {
  __host__ __device__ __bfloat16() : __x(0) { }
  unsigned short __x;
};
__device__ inline __bfloat16 __float2bfloat16(const float f) {
  __bfloat16 val;
  unsigned int u;
  asm("{  mov.b32 %0, %1;}\n" : "=r"(u) : "f"(f));
  if ((u & 0x7fffffffu) > 0x7f800000u) {     // NaN -> quiet NaN, preserve sign
    val.__x = static_cast<unsigned short>((u >> 16) | 0x0040u);
  } else {
    u += 0x00007fffu + ((u >> 16) & 1u);     // round to nearest even
    val.__x = static_cast<unsigned short>(u >> 16);
  }
  return val;
}
__device__ inline float __bfloat162float(const __bfloat16 h) {
  float val;
  unsigned int u = static_cast<unsigned int>(h.__x) << 16;
 asm("{  mov.b32 %0, %1;}\n" : "=f"(val) : "r"(u));
  return val;
}

typedef __bfloat16 bfloat16;

template<>
struct AccType<bfloat16> {
  using type = float;

  __device__ static inline type from(const bfloat16& val) {
    return __bfloat162float(val);
  }

  __device__ static inline bfloat16 to(type val) {
    return __float2bfloat16(val);
  }
};
)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_HALF_INL_H_
