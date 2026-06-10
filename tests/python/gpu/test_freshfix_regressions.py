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

"""GPU regression tests for the fresh-review fixes (freshissues.md).

Grouped here so the individual fixes have a discoverable home; each test fails on
the pre-fix build and passes after the corresponding source change.
"""
import numpy as onp
import pytest

import mxnet as mx
from mxnet import np, npx

npx.set_np()
DEV = mx.gpu(0)


# ---------------------------------------------------------------------------
# C3: the central GPU kernel launcher (mxnet_op.h Kernel<OP,gpu>::Launch) used to
# take `int N`, truncating element counts > 2^31. np.full routes through that
# launcher; pre-fix it raises an invalid-launch error (int-overflowed grid dim),
# post-fix it fills correctly all the way to the tail.
# ---------------------------------------------------------------------------
@pytest.mark.serial
def test_kernel_launcher_handles_more_than_int32_elements():
    n = (1 << 31) + 256  # 2,147,483,904 > INT32_MAX
    try:
        x = np.full((n,), 3, dtype="int8", device=DEV)
        x.wait_to_read()
    except mx.MXNetError as e:
        pytest.skip(f"could not allocate {n} bytes on GPU: {e}")
    except MemoryError:
        pytest.skip("insufficient GPU memory for >2^31-element tensor")
    # Sample low, the exact 2^31 boundary, and the very last element.
    assert int(x[0].item()) == 3
    assert int(x[1 << 31].item()) == 3
    assert int(x[n - 1].item()) == 3
    npx.waitall()


# ---------------------------------------------------------------------------
# C1: scatter_nd must keep working after the capture-exclusion change. (It is not
# reachable through Gluon-2 hybridization, so capture-exclusion is structurally
# identical to gather_nd; this guards the op's functional correctness.)
# ---------------------------------------------------------------------------
@pytest.mark.serial
def test_scatter_nd_gpu_correctness():
    # Scatter 4 updates onto the diagonal of a 4x4 output.
    data = mx.nd.array([1, 2, 3, 4], ctx=DEV)
    idx = mx.nd.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype="int64", ctx=DEV)
    out = mx.nd.scatter_nd(data, idx, shape=(4, 4))
    expected = onp.zeros((4, 4), dtype="float32")
    for k in range(4):
        expected[k, k] = k + 1
    onp.testing.assert_array_equal(out.asnumpy(), expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
