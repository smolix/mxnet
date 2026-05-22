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

#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_

#include <dmlc/logging.h>
#include <mxnet/op_attr_types.h>

namespace mxnet {
namespace op {

inline void AssignQuantizedRangeOutput(float* output,
                                       const float* input,
                                       const OpReqType req,
                                       const char* op_name) {
  switch (req) {
    case kNullOp:
      return;
    case kWriteTo:
    case kWriteInplace:
      output[0] = input[0];
      return;
    case kAddTo:
      output[0] += input[0];
      return;
    default:
      LOG(FATAL) << "Unsupported request type for " << op_name << " range output";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_RANGE_UTILS_H_
