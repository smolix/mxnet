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

"""Regression for the missing-OpenCV crash that hit ~29 d2l notebooks
against the 2026-05-22 cleanup wheel.

The first cleanup wheel was built with USE_OPENCV=OFF, so any d2l notebook
that touched image io (`mx.image.imread`, `mx.image.imdecode`,
`mx.image.imresize`, `gluon.data.vision.transforms.Resize`, etc.) raised
`MXNetError: Build with USE_OPENCV=1 for image io / image resize operator`.

The fix is to ship OpenCV-enabled wheels and bundle the libopencv_*.so
files into the wheel so the loader resolves them next to libmxnet.so.
This test enforces that an OpenCV-on wheel actually exposes the image
operators, and that a tiny in-memory JPEG / PNG round-trip works.

Skips cleanly when the wheel was built with USE_OPENCV=OFF so legacy
hosts (or a future OpenCV-off variant) don't see spurious failures.
"""

import io
import struct
import zlib

import numpy as np
import pytest

import mxnet as mx


# Minimal in-memory PNG that doesn't rely on PIL/OpenCV being importable
# from Python.  4x4 RGB, all pixels (10, 20, 30).
def _make_tiny_png():
    width, height = 4, 4
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # filter byte
        for _ in range(width):
            raw.extend(b"\x0a\x14\x1e")  # RGB(10,20,30)
    compressed = zlib.compress(bytes(raw), 9)
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b""))


def _opencv_enabled():
    return mx.runtime.Features().is_enabled("OPENCV")


@pytest.mark.skipif(not _opencv_enabled(),
                    reason="wheel built with USE_OPENCV=OFF")
def test_imdecode_roundtrip():
    # mx.image.imdecode is what `mx.image.imread` ultimately routes through
    # in d2l's data-loading code paths.  Must return a real ndarray, not
    # raise "Build with USE_OPENCV=1 for image io.".
    buf = _make_tiny_png()
    img = mx.image.imdecode(buf)
    assert img is not None
    assert img.shape == (4, 4, 3)
    # PNG is 8-bit; channel ordering is BGR per OpenCV convention.  Compare
    # the values regardless of order.
    pix = img.asnumpy()[0, 0].tolist()
    assert sorted(pix) == [10, 20, 30]


@pytest.mark.skipif(not _opencv_enabled(),
                    reason="wheel built with USE_OPENCV=OFF")
def test_imresize_does_not_crash():
    # mx.image.imresize wraps the C++ image-resize operator; the d2l
    # vision notebooks call it via Resize / RandomResizedCrop transforms.
    buf = _make_tiny_png()
    img = mx.image.imdecode(buf)
    out = mx.image.imresize(img, 8, 8)
    assert out.shape == (8, 8, 3)


@pytest.mark.skipif(not _opencv_enabled(),
                    reason="wheel built with USE_OPENCV=OFF")
def test_image_resize_transform():
    # gluon.data.vision.transforms.Resize is what the d2l vision chapters
    # use end-to-end in their dataset pipelines.
    from mxnet.gluon.data.vision import transforms
    buf = _make_tiny_png()
    img = mx.image.imdecode(buf)
    out = transforms.Resize((8, 8))(img)
    assert out.shape == (8, 8, 3)


def test_features_opencv_flag_is_consistent_with_imdecode():
    # Tie the runtime feature flag to actual behavior so a future packaging
    # regression (e.g. opencv-on build but no bundled libopencv_*.so visible
    # at runtime) shows up here instead of silently in a user's notebook.
    enabled = _opencv_enabled()
    buf = _make_tiny_png()
    if enabled:
        # Must not raise.
        img = mx.image.imdecode(buf)
        assert img.shape == (4, 4, 3)
    else:
        with pytest.raises(Exception) as exc_info:
            mx.image.imdecode(buf)
        # The error message should mention USE_OPENCV to point the user at
        # the right wheel variant.
        assert "USE_OPENCV" in str(exc_info.value)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
