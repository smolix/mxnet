#!/usr/bin/env bash

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

# This is a convenience script for calling the build scripts of all dependency libraries.
# Environment variables should be set beforehand.
#
# GH1 NOTE (2026-05-23): This is the LEGACY static-build dependency path
# (tools/staticbuild/build.sh).  The current Ampere-through-Blackwell
# Linux/CUDA wheel build uses tools/build_cleanup_wheel.sh instead, which
# bundles dependencies via cmake + pip-installed cuDNN/NCCL and does not
# go through this script.
#
# The individual `tools/dependencies/*.sh` scripts download dependency
# tarballs via the `download` function below.  They do NOT verify
# checksums; if you reactivate this path for a new CD pipeline, pin
# SHA256 hashes per dependency before any production use.  The
# Python builders (`build_openmp.py`, `build_opencv.py`,
# `build_libturbojpeg.py`) already do verify checksums via
# `tools/dependencies/download_checksums.json` — match that policy
# before exposing the .sh path again.

set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

download () {
    local URL=$1
    local OUT_FILE=$2

    if [[ -f "${OUT_FILE}" ]]; then
        echo "File ${OUT_FILE} already downloaded."
        return 0
    fi

    echo "Downloading ${URL} ..."
    local CURL_OPTIONS=(
        --connect-timeout 60
        --max-time 300
        --retry-delay 30
        --retry 5
        --location
        --silent
        --show-error
        --fail
    )
    if ! curl "${CURL_OPTIONS[@]}" "${URL}" -o "${OUT_FILE}"; then
        rm -f "${OUT_FILE}"
        echo "File ${URL} couldn't be downloaded!"
        exit 1
    fi

    if [[ ! -f "${OUT_FILE}" ]]; then
        echo "File ${URL} couldn't be downloaded!"
        exit 1
    fi
}

if [[ ! $PLATFORM == 'darwin' ]] && [[ ! $BLAS == 'mkl' ]]; then
    source "${DIR}/openblas.sh"
fi
source "${DIR}/libz.sh"
source "${DIR}/libturbojpeg.sh"
source "${DIR}/libpng.sh"
source "${DIR}/libtiff.sh"
source "${DIR}/openssl.sh"
source "${DIR}/curl.sh"
source "${DIR}/eigen.sh"
source "${DIR}/opencv.sh"
source "${DIR}/protobuf.sh"
source "${DIR}/cityhash.sh"
source "${DIR}/zmq.sh"
source "${DIR}/lz4.sh"
if [[ $BLAS == 'mkl' ]]; then
    source "${DIR}/mkl.sh"
    source "${DIR}/numpy_mkl.sh"
    if [[ $PLATFORM == 'darwin' ]]; then
        # export this path to find iomp5 needed by MKL according to Intel Link Line Advisor
        export DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/oneapi/compiler/${INTEL_MKL}/mac/compiler
    fi
fi


export LIBRARY_PATH=${LIBRARY_PATH}:$(dirname $(find $DEPS_PATH -type f -name 'libprotoc*' | grep protobuf | head -n 1)):$DEPS_PATH/lib:$DEPS_PATH/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(dirname $(find $DEPS_PATH -type f -name 'libprotoc*' | grep protobuf | head -n 1)):$DEPS_PATH/lib:$DEPS_PATH/lib64
