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
 * \file np_unique_op.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <vector>
#include <numeric>
#include <set>
#include <string>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../contrib/boolean_mask-inl.h"
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

namespace mxnet {
namespace op {

// NaN-aware ordering/equality so np.unique collapses duplicate NaNs and sorts
// them last, matching NumPy. Shared by the CPU and GPU paths. The NaN test uses
// the self-inequality trick (value != value) so it is device-safe (no std::isnan
// in __device__ code) and correct for every dtype: integers are never NaN, and
// half/float/double NaNs compare unequal to themselves.
template <typename DType>
MSHADOW_XINLINE bool NumpyUniqueIsNan(DType value) {
  return value != value;
}

template <typename DType>
MSHADOW_XINLINE bool NumpyUniqueValueLess(DType lhs, DType rhs) {
  const bool lhs_nan = NumpyUniqueIsNan(lhs);
  const bool rhs_nan = NumpyUniqueIsNan(rhs);
  if (lhs_nan || rhs_nan) {
    return !lhs_nan && rhs_nan;
  }
  return lhs < rhs;
}

template <typename DType>
MSHADOW_XINLINE bool NumpyUniqueValueEqual(DType lhs, DType rhs) {
  return (NumpyUniqueIsNan(lhs) && NumpyUniqueIsNan(rhs)) || lhs == rhs;
}

inline bool NumpyUniqueShouldWrite(const std::vector<OpReqType>& req, const size_t output_idx) {
  CHECK_LT(output_idx, req.size());
  CHECK(req[output_idx] == kNullOp || req[output_idx] == kWriteTo ||
        req[output_idx] == kWriteInplace)
      << "NumpyUnique only supports null and write requests";
  return req[output_idx] != kNullOp;
}

struct UniqueReturnInverseKernel {
  MSHADOW_XINLINE static void Map(dim_t i,
                                  dim_t* unique_inverse,
                                  const int32_t* prefix_sum,
                                  const dim_t* perm) {
    dim_t j           = perm[i];
    unique_inverse[j] = prefix_sum[i] - 1;
  }
};

struct UniqueReturnCountsKernel {
  MSHADOW_XINLINE static void Map(dim_t i, dim_t* unique_counts, const dim_t* idx) {
    unique_counts[i] = idx[i + 1] - idx[i];
  }
};

struct NumpyUniqueParam : public dmlc::Parameter<NumpyUniqueParam> {
  bool return_index, return_inverse, return_counts;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyUniqueParam) {
    DMLC_DECLARE_FIELD(return_index)
        .set_default(false)
        .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_inverse)
        .set_default(false)
        .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_counts)
        .set_default(false)
        .describe("If true, return the number of times each unique item appears in input.");
    DMLC_DECLARE_FIELD(axis)
        .set_default(dmlc::optional<int>())
        .describe("An integer that represents the axis to operator on.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream return_index_s, return_inverse_s, return_counts_s, axis_s;
    return_index_s << return_index;
    return_inverse_s << return_inverse;
    return_counts_s << return_counts;
    axis_s << axis;
    (*dict)["return_index"]   = return_index_s.str();
    (*dict)["return_inverse"] = return_inverse_s.str();
    (*dict)["return_counts"]  = return_counts_s.str();
    (*dict)["axis"]           = axis_s.str();
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_
