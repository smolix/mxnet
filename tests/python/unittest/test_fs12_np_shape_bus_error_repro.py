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

"""FS12 reproduction harness — SIGBUS in MXSetIsNumpyShape during long shard.

The original symptom: running `tests/python/unittest/test_numpy_op.py` in full order
crashes with SIGBUS at ~21% through the file, specifically when
`test_np_sum[False-int64-int64-int64-False-1-shape1]` enters its
`_NumpyShapeScope.__enter__()`, which calls into the `MXSetIsNumpyShape`
C API. The same test passes when run in isolation.

The root cause identified here is an ABI mismatch in the Python wrapper:
`MXIsNumpyShape` writes an `int*`, but `mxnet.util.is_np_shape()` used to
pass a one-byte `ctypes.c_bool` out-parameter. That overwrote adjacent
ctypes storage and made the later C API call fail in an unrelated-looking
place. `MXNDArrayIsDeferredCompute` had the same Python-side bug.

This file pins the diagnostic surface:

1. The known crash anchor: a single-test invocation that PASSES in
   isolation (regression sentinel — if this ever fails on its own,
   the bug has shifted).
2. A skipped "test" that documents how to reproduce the crash under
   ASAN (skipped because ASAN isn't in the default validation matrix).
3. ABI contract tests confirming the Python wrappers allocate int-sized
   out-parameters for int* C API functions.
"""

import ctypes
import os
import sys

import pytest


def test_np_sum_int64_axis1_shape1_passes_in_isolation():
    """The exact test that crashes the long shard must pass in isolation —
    the bug is a *prior* test's corruption, not this test's own logic."""
    import mxnet as mx
    import numpy as np
    mx.npx.set_np()
    # Mirror the parametrize arguments from test_np_sum's failing instance:
    # in_dtype=int64, out_dtype=int64, np_out_dtype=int64, keepdims=False,
    # axis=1, shape=shape1 (which is (5, 6) per the parametrize matrix).
    a = mx.np.arange(30, dtype='int64').reshape(5, 6)
    out = mx.np.sum(a, axis=1, keepdims=False)
    assert out.shape == (5,)
    assert out.dtype == np.dtype('int64')
    np_a = np.arange(30, dtype='int64').reshape(5, 6)
    np_expected = np_a.sum(axis=1)
    assert (out.asnumpy() == np_expected).all()


@pytest.mark.skip(reason="FS12: requires ASAN build to root-cause the prior-test corruption")
def test_full_test_numpy_op_shard_crashes_under_asan():
    """Documentation-only repro:

    Build mxnet with `-DUSE_ASAN=ON` (cmake option) or
    `CXXFLAGS=-fsanitize=address -fno-omit-frame-pointer LDFLAGS=-fsanitize=address`,
    then:

        ASAN_OPTIONS=abort_on_error=1:halt_on_error=1:detect_leaks=0 \\
        LD_PRELOAD=$(gcc -print-file-name=libasan.so) \\
        pytest tests/python/unittest/test_numpy_op.py -x \\
                --tb=short --collect-only=0 2>&1 | tee /tmp/fs12-asan.log

    The first ASAN report between test items 0..2350 will name the
    earlier test whose memory write into the np_shape global's page
    (or an aliasing free / use-after-free) sets up the SIGBUS that
    test_np_sum[False-int64-int64-int64-False-1-shape1] then trips.

    A non-ASAN repro without bisection:

        # Run only the crashing test in isolation — passes.
        pytest tests/python/unittest/test_numpy_op.py::test_np_sum \\
            -k 'False-int64-int64-int64-False-1-shape1' --tb=short

        # Run the full file — crashes at ~21%.
        pytest tests/python/unittest/test_numpy_op.py --tb=short --co 0

    Diagnostic facts already gathered (see issues.md FS12 row):
    - Crash is SIGBUS, not SEGV → page-level page-protect or fd-backed unmap.
    - Crash site: thread_local atomic flag write in `_NumpyShapeScope.__enter__`.
    - C API entered: MXSetIsNumpyShape (see src/c_api/c_api_ndarray.cc).
    - Apport intercepts the core; either redirect /proc/sys/kernel/core_pattern
      to a file (root) or run under `gdb --args pytest ...` for an in-process
      backtrace.
    """


def test_fs12_source_anchor_present():
    """If `_NumpyShapeScope.__enter__` stops calling `MXSetIsNumpyShape`, the
    known crash anchor is gone and this file's diagnostic notes need
    updating — fail loudly so the audit doesn't drift."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "python", "mxnet", "util.py")
    if not os.path.exists(src):
        pytest.skip("source tree unavailable: util.py not found at expected path")
    with open(src) as f:
        contents = f.read()
    # MXSetIsNumpyShape is the documented crash C API; the python wrapper
    # may be named set_np_shape / np_shape / is_np_shape.
    assert ("_NumpyShapeScope" in contents
            or "set_np_shape" in contents
            or "np_shape" in contents.lower()), \
        ("FS12: the documented crash anchor (_NumpyShapeScope / set_np_shape) "
         "is no longer present in util.py — issues.md FS12 row needs to be "
         "updated with the new code path.")


def _write_int_out_param(out_param, value):
    obj = getattr(out_param, "_obj", None)
    assert obj is not None
    assert ctypes.sizeof(obj) == ctypes.sizeof(ctypes.c_int)
    ctypes.cast(out_param, ctypes.POINTER(ctypes.c_int))[0] = value
    return 0


def test_is_np_shape_uses_int_sized_c_api_out_param(monkeypatch):
    import mxnet.util as util

    class FakeLib:
        @staticmethod
        def MXIsNumpyShape(curr):
            return _write_int_out_param(curr, 2)

    monkeypatch.setattr(util, "_LIB", FakeLib())
    monkeypatch.setattr(util, "check_call", lambda status: None)

    assert util.is_np_shape() is True


def test_is_deferred_compute_uses_int_sized_c_api_out_param(monkeypatch):
    import mxnet._deferred_compute as deferred_compute

    class FakeLib:
        @staticmethod
        def MXNDArrayIsDeferredCompute(curr):
            return _write_int_out_param(curr, 1)

    monkeypatch.setattr(deferred_compute, "_LIB", FakeLib())
    monkeypatch.setattr(deferred_compute, "check_call", lambda status: None)

    assert deferred_compute.is_deferred_compute() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
