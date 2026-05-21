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

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_CUDA_SM_TO_ARCH = {
    "sm_80": "8.0",
    "sm_86": "8.6",
    "sm_89": "8.9",
    "sm_90": "9.0",
    "sm_120": "12.0+PTX",
}


def _cmake_lists_text():
    return (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def _mxnet_cuda_arch_default():
    match = re.search(
        r'set\(\s*MXNET_CUDA_ARCH\s+"(?P<value>[^"]+)"\s+'
        r'CACHE\s+STRING\s+"(?P<help>.*?)"\s*\)',
        _cmake_lists_text(),
        re.DOTALL,
    )
    assert match is not None, "CMakeLists.txt must define MXNET_CUDA_ARCH"
    value = tuple(token for token in match.group("value").split(";") if token)
    return value, match.group("help")


def _cuda_arch_to_sm(arch):
    numeric_arch = arch.removesuffix("+PTX")
    return "sm_" + numeric_arch.replace(".", "")


def test_release_cuda_arch_matrix_keeps_explicit_ampere_to_blackwell_coverage():
    arch_tokens, help_text = _mxnet_cuda_arch_default()

    unsupported_selectors = {"Auto", "Common", "All"}
    assert unsupported_selectors.isdisjoint(arch_tokens)

    missing_arches = {
        sm: arch
        for sm, arch in RELEASE_CUDA_SM_TO_ARCH.items()
        if arch not in arch_tokens
    }
    assert not missing_arches

    resolved_sms = {_cuda_arch_to_sm(arch) for arch in arch_tokens}
    assert set(RELEASE_CUDA_SM_TO_ARCH) <= resolved_sms

    for sm in RELEASE_CUDA_SM_TO_ARCH:
        assert sm in help_text


def test_cmake_cuda_architectures_defers_to_mxnet_release_arch_matrix():
    text = _cmake_lists_text()
    assert re.search(
        r"if\(NOT DEFINED CMAKE_CUDA_ARCHITECTURES\)\s+"
        r"set\(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING\s+"
        r'"MXNet sets CUDA -gencode flags from MXNET_CUDA_ARCH"\)',
        text,
    )
