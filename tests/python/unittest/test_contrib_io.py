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

import mxnet.ndarray as nd
from mxnet.gluon.data.vision.datasets import *
from mxnet.gluon.data.dataloader import *
from mxnet.contrib.io import *
from mxnet.test_utils import *

# Note: the original `mx.npx.reset_np()` at module scope here is a global
# toggle that leaks into any other test file collected in the same pytest
# invocation. Replace with an autouse fixture so the np-semantics flip is
# scoped per-test and cross-file sweeps stay clean.
import pytest as _pytest_for_reset_np_fixture
@_pytest_for_reset_np_fixture.fixture(autouse=True)
def _legacy_nd_semantics():
    _prev_arr = mx.util.is_np_array()
    _prev_shp = mx.util.is_np_shape()
    mx.npx.reset_np()
    yield
    mx.npx.set_np(shape=_prev_shp, array=_prev_arr)
def test_contrib_DataLoaderIter():
    num_examples = 10000

    def test_mnist_batches(batch_size, expected, last_batch='discard'):
        dataset = mx.gluon.data.ArrayDataset(
            nd.zeros((num_examples, 28, 28, 1)),
            nd.arange(num_examples))
        dataloader = DataLoader(dataset, batch_size, last_batch=last_batch)
        test_iter = DataLoaderIter(dataloader)
        batch = next(test_iter)
        assert batch.data[0].shape == (batch_size, 28, 28, 1)
        assert batch.label[0].shape == (batch_size,)
        count = 0
        test_iter.reset()
        for _ in test_iter:
            count += 1
        assert count == expected, "expected {} batches, given {}".format(expected, count)

    test_mnist_batches(50, num_examples // 50, 'discard')
    test_mnist_batches(31, num_examples // 31, 'discard')
    test_mnist_batches(31, num_examples // 31, 'rollover')
    test_mnist_batches(31, num_examples // 31 + 1, 'keep')


if __name__ == "__main__":
    test_contrib_DataLoaderIter()
