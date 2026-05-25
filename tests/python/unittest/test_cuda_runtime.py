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

import importlib.util
import os
from pathlib import Path


def _load_cuda_runtime_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "python" / "mxnet" / "_cuda_runtime.py"
    spec = importlib.util.spec_from_file_location("_mxnet_cuda_runtime_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_nvidia_cudnn_soname_aliases_creates_versioned_links(tmp_path, monkeypatch):
    cuda_runtime = _load_cuda_runtime_module()
    cudnn_lib_dir = tmp_path / "nvidia" / "cudnn" / "lib"
    cudnn_lib_dir.mkdir(parents=True)
    for name in ("libcudnn_ops.so.9", "libcudnn_graph.so.9"):
        (cudnn_lib_dir / name).write_text("test", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MXNET_CUDNN_ALIAS_REPAIR", "1")
    monkeypatch.setattr(cuda_runtime, "_cudnn_package_version", lambda: "9.22.0.52")

    created = cuda_runtime.repair_nvidia_cudnn_soname_aliases()

    assert set(Path(path).name for path in created) == {
        "libcudnn_ops.so.9.22.0",
        "libcudnn_graph.so.9.22.0",
    }
    for name in ("libcudnn_ops.so.9", "libcudnn_graph.so.9"):
        alias = cudnn_lib_dir / "{}.22.0".format(name)
        assert alias.is_symlink()
        assert os.readlink(str(alias)) == name


def test_repair_nvidia_cudnn_soname_aliases_can_be_disabled(tmp_path, monkeypatch):
    cuda_runtime = _load_cuda_runtime_module()
    cudnn_lib_dir = tmp_path / "nvidia" / "cudnn" / "lib"
    cudnn_lib_dir.mkdir(parents=True)
    (cudnn_lib_dir / "libcudnn_ops.so.9").write_text("test", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MXNET_CUDNN_ALIAS_REPAIR", "0")
    monkeypatch.setattr(cuda_runtime, "_cudnn_package_version", lambda: "9.22.0.52")

    assert cuda_runtime.repair_nvidia_cudnn_soname_aliases() == []
    assert not (cudnn_lib_dir / "libcudnn_ops.so.9.22.0").exists()
