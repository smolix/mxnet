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

import importlib
import sys
import types

import mxnet as mx


def import_opencv_plugin(monkeypatch):
    if 'cv2' not in sys.modules:
        monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(
            INTER_LINEAR=1,
            INTER_CUBIC=2,
            BORDER_CONSTANT=0,
        ))
    return importlib.import_module('plugin.opencv.opencv')


def test_opencv_imdecode_passes_bytes_to_c_api(monkeypatch):
    opencv = import_opencv_plugin(monkeypatch)
    captured = {}

    class FakeLib(object):
        def MXCVImdecode(self, buf, size, flag, out):
            captured['buf'] = buf.value
            captured['size'] = size.value
            captured['flag'] = flag
            return 0

    monkeypatch.setattr(opencv, '_LIB', FakeLib())
    monkeypatch.setattr(opencv.mx.nd, 'NDArray', lambda handle: handle)

    opencv.imdecode(bytearray(b'\xff\xd8data'), flag=7)

    assert captured == {'buf': b'\xff\xd8data', 'size': 6, 'flag': 7}


def test_opencv_image_list_iter_reads_images_as_bytes(tmp_path, monkeypatch):
    opencv = import_opencv_plugin(monkeypatch)
    img_path = tmp_path / 'sample.jpg'
    img_path.write_bytes(b'\xff\xd8data')
    list_path = tmp_path / 'images.lst'
    list_path.write_text('sample\n')

    def fake_imdecode(buf, flag):
        assert isinstance(buf, bytes)
        assert buf == b'\xff\xd8data'
        assert flag == 1
        return mx.nd.zeros((2, 2, 3))

    monkeypatch.setattr(opencv, 'imdecode', fake_imdecode)
    monkeypatch.setattr(opencv, 'random_crop', lambda img, size: (img, (0, 0, size[0], size[1])))

    iterator = opencv.ImageListIter(str(tmp_path) + '/', str(list_path), 1, (2, 2))
    batch = iterator.next()

    assert batch.data[0].shape == (1, 3, 2, 2)
