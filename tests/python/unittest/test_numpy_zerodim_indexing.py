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

"""0-d (scalar) ndarray indexing: () / Ellipsis / newaxis match NumPy; a full
slice `:` is accepted (permissive superset of NumPy) so NumPy fallback ops that
do `a[:]`/`a[:] = v` on a 0-d array interoperate (regression for
test_np_fallback_ops::nanstd). Bad indices (int / partial slice) still raise."""

import numpy as onp
import pytest
from mxnet import np, npx

npx.set_np()


def test_zerodim_getitem_shapes_match_numpy():
    a = np.array(2.5)
    an = onp.array(2.5)
    assert a[()].shape == an[()].shape == ()
    assert a[...].shape == an[...].shape == ()
    assert a[None].shape == an[None].shape == (1,)
    assert a[None, None].shape == an[None, None].shape == (1, 1)
    assert a[..., None].shape == an[..., None].shape == (1,)
    assert a[None, ...].shape == an[None, ...].shape == (1,)


def test_zerodim_full_slice_is_accepted():
    # NumPy raises IndexError on a[:] for 0-d; we accept it as the whole scalar.
    a = np.array(2.5)
    assert a[:].shape == ()
    assert float(a[:].asnumpy()) == 2.5


@pytest.mark.parametrize('key', [(), Ellipsis, None, slice(None)])
def test_zerodim_setitem_sets_scalar(key):
    a = np.array(0.0)
    a[key] = 4.0
    assert float(a.asnumpy()) == 4.0


@pytest.mark.parametrize('bad', [0, slice(1, 2)])
def test_zerodim_bad_index_rejected(bad):
    a = np.array(1.0)
    with pytest.raises(IndexError):
        _ = a[bad]
    with pytest.raises(IndexError):
        a[bad] = 3.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
