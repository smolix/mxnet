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

"""Run pytest with mx.gpu(0) remapped to a selected physical GPU.

MXNet currently reports CUDA error 304 on this host when CUDA_VISIBLE_DEVICES is
set.  Most legacy GPU tests hard-code mx.gpu(0), so this runner lets multiple
pytest processes exercise different physical GPUs without relying on CUDA
visibility masking.
"""

import argparse
import os
import sys


def _patch_mxnet_gpu(lane):
    import mxnet as mx
    import mxnet.context as mx_context
    import mxnet.device as mx_device

    original_gpu = mx_device.gpu

    def lane_gpu(device_id=0):
        if device_id == 0:
            device_id = lane
        return original_gpu(device_id)

    mx.gpu = lane_gpu
    mx_context.gpu = lane_gpu
    mx_device.gpu = lane_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-lane", type=int, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.gpu_lane < 0:
        parser.error("--gpu-lane must be non-negative")

    _patch_mxnet_gpu(args.gpu_lane)

    import pytest

    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
