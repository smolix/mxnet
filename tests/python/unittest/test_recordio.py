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

# pylint: skip-file
import sys
import mxnet as mx
import numpy as np
import pytest
import random
import string
import warnings

def test_recordio(tmpdir):
    frec = tmpdir.join('rec')
    N = 255

    writer = mx.recordio.MXRecordIO(str(frec), 'w')
    for i in range(N):
        writer.write(bytes(str(chr(i)), 'utf-8'))
    del writer

    reader = mx.recordio.MXRecordIO(str(frec), 'r')
    for i in range(N):
        res = reader.read()
        assert res == bytes(str(chr(i)), 'utf-8')

def test_indexed_recordio(tmpdir):
    fidx = tmpdir.join('idx')
    frec = tmpdir.join('rec')
    N = 255

    writer = mx.recordio.MXIndexedRecordIO(str(fidx), str(frec), 'w')
    for i in range(N):
        writer.write_idx(i, bytes(str(chr(i)), 'utf-8'))
    del writer

    reader = mx.recordio.MXIndexedRecordIO(str(fidx), str(frec), 'r')
    keys = reader.keys
    assert sorted(keys) == [i for i in range(N)]
    random.shuffle(keys)
    for i in keys:
        res = reader.read_idx(i)
        assert res == bytes(str(chr(i)), 'utf-8')

def test_indexed_recordio_closes_handles_when_index_load_fails(monkeypatch):
    events = []

    class FakeLib:
        def MXRecordIOReaderCreate(self, uri, handle):
            events.append(('create', uri.value.decode('utf-8')))
            handle._obj.value = 123
            return 0

        def MXRecordIOReaderFree(self, handle):
            events.append(('free', handle.value))
            return 0

    class FakeIndexFile:
        def __init__(self):
            self.read = False

        def readline(self):
            if self.read:
                return ''
            self.read = True
            events.append(('readline',))
            return 'not-an-int\t0\n'

        def close(self):
            events.append(('close-index',))

    def fake_open(path, flag):
        events.append(('open', path, flag))
        return FakeIndexFile()

    monkeypatch.setattr(mx.recordio, '_LIB', FakeLib())
    monkeypatch.setattr(mx.recordio, 'check_call', lambda ret: None)
    monkeypatch.setattr(mx.recordio, 'open', fake_open, raising=False)

    record = object.__new__(mx.recordio.MXIndexedRecordIO)
    record.idx_path = 'broken.idx'
    record.idx = {}
    record.keys = []
    record.key_type = int
    record.fidx = None
    record.uri = mx.recordio.c_str('data.rec')
    record.handle = mx.recordio.RecordIOHandle()
    record.flag = 'r'
    record.pid = None
    record.is_open = False

    with pytest.raises(ValueError):
        record.open()

    assert events == [
        ('create', 'data.rec'),
        ('open', 'broken.idx', 'r'),
        ('readline',),
        ('free', 123),
        ('close-index',),
    ]
    assert record.is_open is False
    assert record.fidx is None

def test_recordio_pack_label():
    N = 255

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        for i in range(1, N):
            for j in range(N):
                content = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(j))
                content = content.encode('utf-8')
                label = np.random.uniform(size=i).astype(np.float32)
                header = (0, label, 0, 0)
                s = mx.recordio.pack(header, content)
                rheader, rcontent = mx.recordio.unpack(s)
                assert (label == rheader.label).all()
                assert content == rcontent
