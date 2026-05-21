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

    data = mx.sym.Variable(name="data", dtype="bfloat16")
    conv = mx.sym.Convolution(data=data, name="bf16_probe",
                              kernel=(1, 1), num_filter=1, no_bias=True)
    try:
        exe = conv._simple_bind(ctx=mx.cpu(), data=(1, 1, 4, 4),
                                type_dict={"bf16_probe_weight": "bfloat16"})
        exe.arg_dict["data"][:] = mx.nd.ones((1, 1, 4, 4)).astype("bfloat16")
        exe.arg_dict["bf16_probe_weight"][:] = mx.nd.ones((1, 1, 1, 1)).astype("bfloat16")
        exe.forward()[0].wait_to_read()
        return True
    except mx.MXNetError as err:
        message = str(err).lower()
        expected_missing_isa = (
            "could not create a primitive descriptor" in message or
            "bf16" in message or
            "bfloat16" in message
        )
        if expected_missing_isa:
            return False
        raise


def require_native_onednn_bf16():
    if not has_native_onednn_bf16():
        pytest.skip("oneDNN native BF16 primitives are unavailable on this CPU")
