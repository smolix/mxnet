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

"""CPU regression tests for the fresh-review Python-frontend fixes (freshissues.md)."""
import pickle

import numpy as onp
import pytest

import mxnet as mx
from mxnet._ctypes.ndarray import NDArrayBase


# H3: __del__ must tolerate a partially-constructed object whose `handle` attribute
# was never assigned (the __reduce__/unpickle path constructs with handle=None, and
# an exception during construction can leave the attribute unset).
def test_ndarray_del_tolerates_missing_handle():
    obj = NDArrayBase.__new__(NDArrayBase)  # bypass __init__: no `handle` attr
    obj.__del__()  # must not raise / must not free a NULL handle


def test_ndarray_pickle_roundtrip():
    a = mx.nd.array([[1.0, 2.0], [3.0, 4.0]])
    b = pickle.loads(pickle.dumps(a))
    onp.testing.assert_array_equal(a.asnumpy(), b.asnumpy())


# H14: legacy NDArray has element-wise __eq__, so it must not be hashable (matches
# the numpy frontend); otherwise it silently violates the hash/eq contract.
def test_legacy_ndarray_is_unhashable():
    a = mx.nd.array([1, 2, 3])
    with pytest.raises((TypeError, NotImplementedError)):
        hash(a)
    with pytest.raises((TypeError, NotImplementedError)):
        {a: 1}  # noqa: used for its side effect (hashing)


def test_numpy_ndarray_is_unhashable():
    a = mx.np.array([1, 2, 3])
    with pytest.raises((TypeError, NotImplementedError)):
        hash(a)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
