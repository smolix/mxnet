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

LIMITATION: this remapping only confines ``mx.gpu(0)`` to the lane.  A test that
explicitly requests ``mx.gpu(1)`` (or higher) addresses a *physical* device the
lane does not own, so two concurrent lanes can collide on the same card and
produce flaky results or false passes.  By default this runner therefore raises
when a test asks for any device other than the lane, turning a silent collision
into a deterministic failure.  Pass ``--allow-multi-gpu`` to restore the old
permissive behaviour (e.g. when running a single lane over a genuinely
multi-device test).  CUDA_VISIBLE_DEVICES masking remains the correct long-term
mechanism once the error-304 issue is root-caused.
"""

import argparse
import os
import sys


def _patch_mxnet_gpu(lane, allow_multi_gpu):
    import mxnet as mx
    import mxnet.context as mx_context
    import mxnet.device as mx_device

    original_gpu = mx_device.gpu

    def lane_gpu(device_id=0):
        if device_id == 0:
            device_id = lane
        elif device_id != lane and not allow_multi_gpu:
            raise RuntimeError(
                f"run_pytest_gpu_lane: test requested mx.gpu({device_id}) but this "
                f"lane only owns physical GPU {lane}. Concurrent lanes would collide "
                f"on a shared device. Skip/serialize multi-GPU tests, or pass "
                f"--allow-multi-gpu to opt out of this guard.")
        return original_gpu(device_id)

    mx.gpu = lane_gpu
    mx_context.gpu = lane_gpu
    mx_device.gpu = lane_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-lane", type=int, required=True)
    parser.add_argument("--allow-multi-gpu", action="store_true",
                        help="permit tests to address GPUs other than the lane "
                             "(disables the collision guard)")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.gpu_lane < 0:
        parser.error("--gpu-lane must be non-negative")

    _patch_mxnet_gpu(args.gpu_lane, args.allow_multi_gpu)

    import pytest

    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
