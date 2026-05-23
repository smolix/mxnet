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

"""GH6 tail: dataset / transform edge cases.

The prior GH6 sweep landed RecordFileDataset reader reset, ImageFolderDataset
class= ordering, dynamic separator selection, MultiBoxPrior coverage, and
RandomRotation skip-path label preservation. Three concrete edge cases were
identified afterwards in `gluon.data.vision.datasets`:

1. `ImageListDataset` crashed on a trailing blank line in the imglist file
   (`int("")` → ValueError) instead of skipping it.
2. `ImageListDataset` used different key types for the file-loaded path
   (int) vs the list path (str), so the same dataset accessed two
   different keyspaces depending on its origin.
3. `MNIST` silently produced an empty 0-shaped data tensor if the label
   file was empty or truncated.

This file pins each fix.
"""

import gzip
import io
import os
import struct
import tempfile

import numpy as np
import pytest

import mxnet as mx
from mxnet.gluon.data.vision import datasets


def test_image_list_dataset_skips_blank_lines():
    """A trailing newline / blank line in the imglist file must not raise
    ValueError from int('') — silently skip it."""
    with tempfile.TemporaryDirectory() as tmp:
        list_path = os.path.join(tmp, "imglist.lst")
        # One valid entry, then a blank line, then another valid entry.
        # Also create dummy image files to avoid downstream failures.
        for i in (1, 2):
            with open(os.path.join(tmp, f"img{i}.bin"), "wb") as f:
                f.write(b"x")
        with open(list_path, "w") as f:
            f.write("1\t0.0\timg1.bin\n")
            f.write("\n")  # blank line — used to crash
            f.write("2\t0.0\timg2.bin\n")
        ds = datasets.ImageListDataset(root=tmp, imglist="imglist.lst")
        # Two entries should be present despite the blank line.
        assert len(ds._imgkeys) == 2
        assert ds._imgkeys == [1, 2]


def test_image_list_dataset_key_type_consistent():
    """File-loaded and list-loaded ImageListDatasets must use the same key
    type so downstream lookups behave consistently."""
    with tempfile.TemporaryDirectory() as tmp:
        # File path: use a single entry.
        for i in (1,):
            with open(os.path.join(tmp, f"img{i}.bin"), "wb") as f:
                f.write(b"x")
        list_path = os.path.join(tmp, "imglist.lst")
        with open(list_path, "w") as f:
            f.write("1\t0.0\timg1.bin\n")
        ds_file = datasets.ImageListDataset(root=tmp, imglist="imglist.lst")
        # List path: same one entry, this time as a Python list.
        ds_list = datasets.ImageListDataset(
            root=tmp, imglist=[[0.0, "img1.bin"]])
        # Both keys must have the same Python type. (Previously file→int,
        # list→str; this regressed lookups by integer index for list-built
        # datasets.)
        key_types_file = {type(k) for k in ds_file._imgkeys}
        key_types_list = {type(k) for k in ds_list._imgkeys}
        assert key_types_file == key_types_list, \
            f"GH6: ImageListDataset key types differ — file={key_types_file}, list={key_types_list}"


def test_mnist_raises_on_empty_label_file(monkeypatch):
    """An empty / truncated MNIST label file must raise ValueError rather
    than produce a silently-empty dataset that would propagate as a
    length-0 tensor downstream."""
    with tempfile.TemporaryDirectory() as tmp:
        # Build a fake MNIST label file with the standard 8-byte header
        # but ZERO labels following.
        label_path = os.path.join(tmp, "train-labels-idx1-ubyte.gz")
        with gzip.open(label_path, "wb") as f:
            f.write(struct.pack(">II", 0x801, 0))  # magic + count=0
            # No label bytes follow.
        # Build a matching empty data file.
        data_path = os.path.join(tmp, "train-images-idx3-ubyte.gz")
        with gzip.open(data_path, "wb") as f:
            f.write(struct.pack(">IIII", 0x803, 0, 28, 28))
        # Patch the download function so it returns our paths instead of
        # trying to fetch from the network. The real `_get_data` runs.
        def fake_download(url, path=None, sha1_hash=None):
            name = url.rsplit("/", 1)[-1]
            return label_path if "labels" in name else data_path
        monkeypatch.setattr(datasets, "download", fake_download)
        with pytest.raises(ValueError) as excinfo:
            datasets.MNIST(root=tmp, train=True)
        msg = str(excinfo.value)
        assert "0 labels" in msg or "empty" in msg.lower() or "truncated" in msg.lower(), \
            f"GH6: MNIST empty-label error message regressed: {msg!r}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
