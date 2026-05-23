#!/usr/bin/env python3
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
