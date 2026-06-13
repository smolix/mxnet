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
import os
import sys
import mxnet as mx
import pytest


curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
from mxnet.test_utils import default_device, set_default_device

sys.path.insert(0, os.path.join(curr_path, 'quantization'))
from test_quantization import *
from test_quantization import _legacy_nd_semantics


@pytest.fixture(autouse=True)
def _gpu_default_device():
    prev = default_device()
    set_default_device(mx.gpu(0))
    yield
    set_default_device(prev)


# test_rnn_quantization and test_quantized_rnn (imported from test_quantization)
# now run on GPU as normal regression tests: _contrib_quantized_rnn has a GPU
# FStatefulCompute (dequantize + cuDNN float LSTM fallback) and
# _contrib_quantize_asym has a GPU FStatefulCompute with an on-device scale/shift
# path, so the GPU graph-quantization chain is fully registered.
