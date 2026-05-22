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

import os
import subprocess
import sys
import textwrap

import pytest


def _run_optimized_python(source):
    env = os.environ.copy()
    repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python'))
    env['PYTHONPATH'] = repo_python + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, '-O', '-c', textwrap.dedent(source)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('source', [
    """
    import mxnet as mx
    record = object.__new__(mx.recordio.MXRecordIO)
    record.writable = False
    try:
        record.write(b'data')
    except RuntimeError as err:
        if 'not open for writing' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('MXRecordIO.write accepted read-only handle')
    """,
    """
    from mxnet.kvstore.base import _ctype_key_value
    try:
        _ctype_key_value(1.5, [])
    except TypeError as err:
        if 'unexpected type for keys' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('_ctype_key_value accepted invalid key type')
    """,
    """
    from mxnet import amp
    try:
        amp.convert_symbol('not-a-symbol', {}, {}, 'float16')
    except TypeError as err:
        if 'convert_symbol should be a Symbol' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('amp.convert_symbol accepted invalid symbol')
    """,
    """
    import numpy as np
    import mxnet as mx
    from mxnet.rtc import CudaKernel

    CudaKernel.__del__ = lambda self: None
    kernel = object.__new__(CudaKernel)
    kernel._name = 'test_kernel'
    kernel._is_ndarray = [False]
    kernel._dtypes = [np.float32]
    try:
        kernel.launch(['not-a-number'], mx.gpu(0), (1, 1, 1), (1, 1, 1))
    except TypeError as err:
        if 'expected to be a number' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('CudaKernel.launch accepted invalid scalar argument')
    """,
    """
    from mxnet.lr_scheduler import LRScheduler
    try:
        LRScheduler(warmup_steps=1.5)
    except TypeError as err:
        if 'warmup_steps must be an int' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('LRScheduler accepted non-integer warmup_steps')
    """,
    """
    from mxnet.lr_scheduler import MultiFactorScheduler
    try:
        MultiFactorScheduler(step=())
    except TypeError as err:
        if 'Schedule step must be a list of integers' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('MultiFactorScheduler accepted non-list step')
    """,
    """
    from mxnet.lr_scheduler import PolyScheduler
    try:
        PolyScheduler(max_update=1.5)
    except TypeError as err:
        if 'max_update must be an int' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('PolyScheduler accepted non-integer max_update')
    """,
    """
    from mxnet.lr_scheduler import CosineScheduler
    try:
        CosineScheduler(max_update=1.5)
    except TypeError as err:
        if 'max_update must be an int' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('CosineScheduler accepted non-integer max_update')
    """,
    """
    from mxnet.lr_scheduler import LRScheduler
    scheduler = LRScheduler(warmup_steps=1)
    try:
        scheduler.get_warmup_lr(1)
    except ValueError as err:
        if 'num_update must be smaller than warmup_steps' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('get_warmup_lr accepted num_update outside warmup range')
    """,
    """
    from mxnet.gluon.data import ArrayDataset
    try:
        ArrayDataset([1], [2, 3])
    except ValueError as err:
        if 'same length' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('ArrayDataset accepted mismatched lengths')
    """,
    """
    from mxnet.gluon.data import ArrayDataset
    try:
        ArrayDataset([1, 2, 3]).shard(0, 0)
    except ValueError as err:
        if 'Number of shards' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Dataset.shard accepted zero shards')
    """,
    """
    from mxnet.gluon.data.batchify import Group, Stack
    try:
        Group([Stack()], Stack())
    except ValueError as err:
        if 'Input pattern not understood' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Group accepted mixed constructor pattern')
    """,
    """
    from mxnet.gluon.data.batchify import Group
    try:
        Group(1)
    except TypeError as err:
        if 'Batchify functions must be callable' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Group accepted non-callable batchify function')
    """,
])
def test_user_validation_survives_optimized_python(source):
    _run_optimized_python(source)
