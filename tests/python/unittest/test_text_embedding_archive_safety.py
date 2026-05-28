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

import hashlib
import os
import tarfile
import zipfile

import pytest

from mxnet.contrib.text import embedding
from mxnet.gluon import utils as gluon_utils


class _TestEmbedding(embedding._TokenEmbedding):
    pretrained_file_name_sha1 = {}
    pretrained_archive_name_sha1 = {}

    @classmethod
    def _get_pretrained_file_url(cls, pretrained_file_name):
        return "https://example.com/archive" + cls.archive_ext


def _configure_embedding(archive_ext, file_bytes):
    _TestEmbedding.archive_ext = archive_ext
    _TestEmbedding.pretrained_file_name_sha1 = {
        "vec.txt": hashlib.sha1(file_bytes).hexdigest()
    }
    _TestEmbedding.pretrained_archive_name_sha1 = {
        "archive" + archive_ext: "ignored-by-test"
    }


def test_pretrained_embedding_zip_rejects_path_traversal(monkeypatch, tmp_path):
    file_bytes = b"token 0.1 0.2\n"
    _configure_embedding(".zip", file_bytes)

    def fake_download(url, path, sha1_hash=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("vec.txt", file_bytes)
            zf.writestr("../escaped", b"pwned")

    monkeypatch.setattr(gluon_utils, "download", fake_download)

    with pytest.raises(ValueError, match="outside target directory"):
        _TestEmbedding._get_pretrained_file(str(tmp_path), "vec.txt")

    assert not (tmp_path / "escaped").exists()


def test_pretrained_embedding_tar_rejects_path_traversal(monkeypatch, tmp_path):
    file_bytes = b"token 0.1 0.2\n"
    _configure_embedding(".tar.gz", file_bytes)

    def fake_download(url, path, sha1_hash=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            good = tmp_path / "vec.txt"
            good.write_bytes(file_bytes)
            tar.add(str(good), arcname="vec.txt")
            escaped = tmp_path / "escaped-source"
            escaped.write_bytes(b"pwned")
            tar.add(str(escaped), arcname="../escaped")

    monkeypatch.setattr(gluon_utils, "download", fake_download)

    with pytest.raises(ValueError, match="outside target directory"):
        _TestEmbedding._get_pretrained_file(str(tmp_path), "vec.txt")

    assert not (tmp_path / "escaped").exists()
