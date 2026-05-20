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

import mxnet as mx
from mxnet import np
import numpy as onp
import pytest

try:
    import torch
except ImportError:
    torch = None


def test_dlpack_numpy_mxnet_cpu():
    x = onp.array([1.0, 2.0, 3.0], dtype=onp.float32)
    nx = np.from_dlpack(x)
    assert nx.device == mx.cpu(0)
    assert onp.allclose(nx.asnumpy(), x)


def test_dlpack_mxnet_numpy_cpu():
    x = np.array([1.0, 2.0, 3.0], dtype="float32")
    y = onp.from_dlpack(x)
    assert onp.allclose(y, x.asnumpy())


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA is not available")
def test_dlpack_torch_mxnet_torch():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor((5,), device='cuda:0', dtype=torch.float64) + 1
    stream.synchronize()
    nx = np.from_dlpack(x)
    assert nx.device == mx.gpu(0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        z = torch.from_dlpack(nx)
    stream.synchronize()
    z += 1
    assert z == x


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA is not available")
def test_dlpack_mxnet_torch_mxnet():
    x = np.array([5], device=mx.gpu(), dtype="float64") + 1
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tx = torch.from_dlpack(x)
    stream.synchronize()
    z = np.from_dlpack(tx)
    z += 1
    assert z.device == mx.gpu(0)
    assert z == x

def test_dlpack_error_message():
    with pytest.raises(AttributeError):
        # Raise AttributeError for objects that do not implement the DLPack protocol.
        np.from_dlpack(object())

    if torch is None or not torch.cuda.is_available():
        pytest.skip("remaining error checks require CUDA")
    
    with pytest.raises(TypeError):
        # raise TypeError, Stream must be int or None
        stream = torch.cuda.Stream()
        x = np.array([5], device=mx.gpu(), dtype="float64")
        tx = torch.from_dlpack(x.__dlpack__(stream=stream))
    
    with pytest.raises(ValueError):
        # raise ValueError, CPU device has no stream
        x = np.array([5], dtype="float64")
        tx = torch.from_dlpack(x.__dlpack__(stream=0))
