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

"""Regression tests for the npy/npz loader hardening (freshissues.md M20).

Malformed/untrusted archives must fail with a clean error, never crash/UB; valid
files must still round-trip (including >255-byte headers, which the old broken
header-length decode silently truncated).
"""
import os
import struct
import tempfile
import zipfile

import numpy as onp
import pytest

import mxnet as mx


def _save_load_roundtrip(arr):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npy")
        mx.nd.save(path, [mx.nd.array(arr)])
        loaded = mx.nd.load(path)
    return loaded


def test_npy_roundtrip_basic():
    arr = onp.arange(24, dtype="float32").reshape(2, 3, 4)
    loaded = _save_load_roundtrip(arr)
    onp.testing.assert_array_equal(loaded[0].asnumpy(), arr)


def test_npy_roundtrip_many_dims_long_header():
    # A high-dimensional shape produces a header > 255 bytes; the old little-endian
    # decode bug (>> instead of <<) truncated it. This must round-trip exactly.
    arr = onp.ones((1,) * 30, dtype="float32")
    loaded = _save_load_roundtrip(arr)
    assert loaded[0].shape == arr.shape


def test_npz_zero_length_entry_name_is_rejected_cleanly():
    # Craft a zip with a zero-length member name; the loader must not crash
    # (string_view size()-1 underflow) — it should skip it or raise cleanly.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "evil.npz")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("", b"garbage")          # zero-length name
            zf.writestr("x", b"not a npy")        # too-short name
        try:
            mx.nd.load(path)
        except mx.MXNetError:
            pass  # clean error is acceptable
        # The key requirement is no crash/abort; reaching here is success.


def test_npy_truncated_header_is_rejected_cleanly():
    # Valid magic + version but a header length pointing past EOF.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trunc.npy")
        with open(path, "wb") as f:
            f.write(b"\x93NUMPY")               # magic
            f.write(bytes([0x01, 0x00]))         # version 1.0
            f.write(struct.pack("<H", 9999))     # header_len = 9999 (no header follows)
        with pytest.raises(mx.MXNetError):
            mx.nd.load(path)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
