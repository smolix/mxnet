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

"""Run pytest with BLAS/OpenMP thread pools capped per worker."""

import os
import sys


def _enable_cpu_only_collection():
    import mxnet as mx
    import mxnet.context as mx_context
    import mxnet.device as mx_device

    mx_device.num_gpus = lambda: 0
    mx_context.num_gpus = lambda: 0
    mx.device.num_gpus = lambda: 0
    mx.context.num_gpus = lambda: 0
    mx.test_utils.set_default_device(mx.cpu())


def main():
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    args = sys.argv[1:]
    if "--cpu-only" in args:
        args.remove("--cpu-only")
        _enable_cpu_only_collection()

    import pytest

    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
