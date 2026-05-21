#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from unittest.mock import Mock

import pytest

from ci import util


class FakeResponse:
    def __init__(self, status_code, chunks=()):
        self.status_code = status_code
        self._chunks = chunks

    def iter_content(self, chunk_size):
        return iter(self._chunks)


def test_download_file_streams_with_timeout(monkeypatch, tmp_path):
    get = Mock(return_value=FakeResponse(200, [b"abc", b"", b"def"]))
    monkeypatch.setattr(util.requests, "get", get)

    result = util.download_file("https://example.test/path/archive.zip?sig=ignored", tmp_path)

    assert result == str(tmp_path / "archive.zip")
    assert (tmp_path / "archive.zip").read_bytes() == b"abcdef"
    get.assert_called_once_with(
        "https://example.test/path/archive.zip?sig=ignored",
        stream=True,
        timeout=util.DOWNLOAD_TIMEOUT_SECONDS,
    )


def test_download_file_rejects_non_success_response(monkeypatch, tmp_path):
    monkeypatch.setattr(util.requests, "get", Mock(return_value=FakeResponse(500, [b"error"])))

    with pytest.raises(RuntimeError, match="returned status code 500"):
        util.download_file("https://example.test/archive.zip", tmp_path)

    assert not (tmp_path / "archive.zip").exists()


def test_download_file_keeps_legacy_404_return(monkeypatch, tmp_path):
    monkeypatch.setattr(util.requests, "get", Mock(return_value=FakeResponse(404, [b"missing"])))

    assert util.download_file("https://example.test/missing.zip", tmp_path) == 404
    assert not (tmp_path / "missing.zip").exists()
