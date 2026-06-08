# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from functools import lru_cache

import mxnet as mx
import pytest


def has_onednn():
    return mx.runtime.Features().is_enabled("ONEDNN")


@lru_cache(maxsize=None)
def has_native_onednn_bf16():
    if not has_onednn():
        return False

    # Probe with a BF16 matmul (dot), NOT a convolution. Convolution has a
    # BF16->FP32 graceful fallback ("route unsupported BF16 away from oneDNN"),
    # so a conv probe succeeds even on CPUs without native BF16 ISA and would
    # wrongly report BF16 as available. dot/batch_dot have no such fallback:
    # on a CPU with native oneDNN BF16, dot dispatches to the oneDNN BF16
    # matmul primitive and succeeds; without it, oneDNN cannot create the
    # primitive and the native dot rejects BF16 outright. That makes dot a
    # faithful probe for *native* oneDNN BF16 support, correctly gating the
    # BF16 op tests (dot/batch_dot/LRN/batch_norm/...) on non-BF16 CPUs.
    try:
        a = mx.nd.ones((4, 4), dtype="bfloat16")
        mx.nd.dot(a, a).wait_to_read()
        return True
    except mx.MXNetError as err:
        message = str(err).lower()
        expected_missing_isa = (
            "could not create a primitive descriptor" in message or
            "bf16" in message or
            "bfloat16" in message or
            "float16/float32/float64" in message or
            "only supports floating point" in message
        )
        if expected_missing_isa:
            return False
        raise


def require_native_onednn_bf16():
    if not has_native_onednn_bf16():
        pytest.skip("oneDNN native BF16 primitives are unavailable on this CPU")
