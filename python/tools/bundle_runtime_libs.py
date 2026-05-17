#!/usr/bin/env python3
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
"""Bundle CUDA / cuDNN / NCCL runtime libraries into python/mxnet/lib/ so the
produced wheel is self-contained.

After running this script:
  - python/mxnet/lib/ contains the runtime .so files and symlinks
  - python/mxnet/libmxnet.so has RUNPATH set to $ORIGIN/lib

Requires:
  - patchelf in PATH
  - The CUDA toolkit and cuDNN/NCCL installed somewhere readable

Usage:
  python tools/bundle_runtime_libs.py
"""
import os
import shutil
import subprocess
import sys


# Files we bundle. For each, the destination filename is the SONAME the
# dynamic linker actually looks for (matches DT_NEEDED entries in libmxnet.so).
# We copy the on-disk file (which has the full version in its name) but rename
# it to the soname, avoiding shipping both the symlink and the target (wheels
# can't represent symlinks, so doing so would double our storage cost).
# Mapping: source filename in SEARCH_DIRS -> SONAME we package as.
BUNDLED = [
    # cuDNN 9.x (from local wheel cudnn_local/unpacked/.../lib — filenames are
    # already the SONAME `.so.9`, so source == destination here).
    ('libcudnn.so.9', 'libcudnn.so.9'),
    ('libcudnn_adv.so.9', 'libcudnn_adv.so.9'),
    ('libcudnn_cnn.so.9', 'libcudnn_cnn.so.9'),
    ('libcudnn_engines_precompiled.so.9', 'libcudnn_engines_precompiled.so.9'),
    ('libcudnn_engines_runtime_compiled.so.9', 'libcudnn_engines_runtime_compiled.so.9'),
    ('libcudnn_engines_tensor_ir.so.9', 'libcudnn_engines_tensor_ir.so.9'),
    ('libcudnn_graph.so.9', 'libcudnn_graph.so.9'),
    ('libcudnn_heuristic.so.9', 'libcudnn_heuristic.so.9'),
    ('libcudnn_ops.so.9', 'libcudnn_ops.so.9'),
    # NCCL 2.x
    ('libnccl.so.2.28.3', 'libnccl.so.2'),
    # CUDA 13 runtime + math libs
    ('libcudart.so.13.0.96', 'libcudart.so.13'),
    ('libcublas.so.13.1.0.3', 'libcublas.so.13'),
    ('libcublasLt.so.13.1.0.3', 'libcublasLt.so.13'),
    ('libcufft.so.12.0.0.61', 'libcufft.so.12'),
    ('libcusolver.so.12.0.4.66', 'libcusolver.so.12'),
    ('libcurand.so.10.4.0.35', 'libcurand.so.10'),
    ('libcusparse.so.12.6.3.3', 'libcusparse.so.12'),
    ('libnvrtc.so.13.0.88', 'libnvrtc.so.13'),
    ('libnvJitLink.so.13.0.88', 'libnvJitLink.so.13'),
    # libnvrtc-builtins is dlopen()ed by libnvrtc at runtime with the major+minor
    # SONAME (libnvrtc-builtins.so.13.0), not just the major.
    ('libnvrtc-builtins.so.13.0.88', 'libnvrtc-builtins.so.13.0'),
    # Fortran ABI used by OpenBLAS
    ('libgfortran.so.5', 'libgfortran.so.5'),
]

SEARCH_DIRS = [
    # Prefer the local cuDNN wheel (9.22) over the system one (9.14).
    '/workspace/mxnet/cudnn_local/unpacked/nvidia/cudnn/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/local/cuda/lib64',
    '/usr/local/cuda/targets/x86_64-linux/lib',
]


def find_lib(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {name} in {SEARCH_DIRS}')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.normpath(os.path.join(here, '..', 'mxnet'))
    lib_dir = os.path.join(pkg_root, 'lib')
    libmxnet = os.path.join(pkg_root, 'libmxnet.so')

    if not os.path.exists(libmxnet):
        sys.exit(f'libmxnet.so not found at {libmxnet}; build first.')

    shutil.rmtree(lib_dir, ignore_errors=True)
    os.makedirs(lib_dir, exist_ok=True)

    for srcname, dstname in BUNDLED:
        src = find_lib(srcname)
        dst = os.path.join(lib_dir, dstname)
        shutil.copy(src, dst)

    # Set RUNPATH=$ORIGIN/lib on libmxnet.so
    subprocess.check_call(['patchelf', '--set-rpath', '$ORIGIN/lib', libmxnet])
    # Set RUNPATH=$ORIGIN on each bundled lib so they find each other
    for fn in os.listdir(lib_dir):
        p = os.path.join(lib_dir, fn)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        try:
            subprocess.check_call(['patchelf', '--set-rpath', '$ORIGIN', p])
        except subprocess.CalledProcessError:
            # Some files (e.g. linker scripts) are not ELF; skip.
            pass

    print(f'Bundled libraries into {lib_dir}')
    print('libmxnet.so RUNPATH:')
    subprocess.check_call(['patchelf', '--print-rpath', libmxnet])


if __name__ == '__main__':
    main()
