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

import os
import io

import mxnet as mx
import requests

import pytest

try:
    from unittest import mock
except ImportError:
    import mock


class MockResponse(requests.Response):
    def __init__(self, status_code, content):
        super(MockResponse, self).__init__()
        self.status_code = status_code
        self.raw = io.BytesIO(content)


@mock.patch(
    'requests.get', mock.Mock(side_effect=requests.exceptions.ConnectionError))
def test_download_retries():
    with pytest.raises(Exception):
        mx.test_utils.download("http://doesnotexist.notfound")


@mock.patch(
    'requests.get',
    mock.Mock(side_effect=lambda *args, **kwargs: MockResponse(200, b'MOCK CONTENT' * 100)))
def test_download_successful(tmpdir):
    tmp = str(tmpdir)
    tmpfile = os.path.join(tmp, 'README.md')
    mx.test_utils.download("https://raw.githubusercontent.com/apache/mxnet/master/README.md",
                           fname=tmpfile)
    assert os.path.getsize(tmpfile) > 100


def test_download_stream_error_closes_response_and_removes_partial(monkeypatch, tmp_path):
    closed = []
    target = tmp_path / 'partial.bin'

    class FakeResponse:
        status_code = 200

        def iter_content(self, chunk_size=1024):
            yield b'partial'
            raise RuntimeError('stream failed')

        def close(self):
            closed.append(True)

    monkeypatch.setattr(mx.test_utils.requests, 'get',
                        lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match='stream failed'):
        mx.test_utils.download('https://example.invalid/partial.bin',
                               fname=str(target),
                               retries=0)

    assert closed == [True]
    assert not target.exists()
    assert list(tmp_path.glob('partial.bin.*')) == []
