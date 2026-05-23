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
    """
    from mxnet.gluon import nn
    try:
        nn.LeakyReLU(-0.1)
    except ValueError as err:
        if 'Slope coefficient for LeakyReLU' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('LeakyReLU accepted negative alpha')
    """,
    """
    from mxnet.gluon import nn
    try:
        nn.Conv2D(1, 3, layout='BAD')
    except ValueError as err:
        if "Only supports 'NCHW' and 'NHWC' layout" not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Conv2D accepted invalid layout')
    """,
    """
    from mxnet.gluon import nn
    try:
        nn.Conv2D(1, (3, 3, 3))
    except ValueError as err:
        if 'kernel_size must be a number or a list of 2 ints' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Conv2D accepted invalid kernel_size rank')
    """,
    """
    from mxnet.gluon import nn
    try:
        nn.Conv2DTranspose(1, 3, output_padding=(0, 0, 0))
    except ValueError as err:
        if 'output_padding must be a number or a list of 2 ints' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Conv2DTranspose accepted invalid output_padding rank')
    """,
    """
    from mxnet.gluon import nn
    try:
        nn.MaxPool2D(layout='BAD')
    except ValueError as err:
        if 'Only NCHW and NHWC layouts are valid for 2D Pooling' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('MaxPool2D accepted invalid layout')
    """,
    """
    from mxnet.gluon import nn
    try:
        nn.MaxPool2D(pool_size=(2, 2, 2))
    except ValueError as err:
        if 'pool_size must be a number or a list of 2 ints' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('MaxPool2D accepted invalid pool_size rank')
    """,
    """
    from mxnet.gluon import Parameter
    try:
        Parameter('w', shape=(2,), grad_req='bogus')
    except ValueError as err:
        if 'grad_req must be one of' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Parameter accepted invalid grad_req')
    """,
    """
    from mxnet.gluon import Parameter
    try:
        Parameter('w', shape=(2,), stype='bogus')
    except ValueError as err:
        if 'stype for Parameter must be' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Parameter accepted invalid stype')
    """,
    """
    from mxnet.gluon import Parameter
    try:
        Parameter('w', shape=(2,), grad_stype='bogus')
    except ValueError as err:
        if 'grad_stype for Parameter must be' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Parameter accepted invalid grad_stype')
    """,
    """
    from mxnet.optimizer import SGD
    try:
        SGD(lazy_update=True, use_fused_step=False)
    except ValueError as err:
        if 'lazy_update has to be turned off' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('SGD accepted lazy_update with use_fused_step=False')
    """,
    """
    from mxnet.optimizer import SGD
    try:
        SGD(lazy_update=True, multi_precision=True)
    except ValueError as err:
        if 'multi_precision has be turned off' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('SGD accepted lazy_update with multi_precision')
    """,
    """
    from mxnet.optimizer import Adam
    try:
        Adam(lazy_update=True, use_fused_step=False)
    except ValueError as err:
        if 'lazy_update has to be turned off' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Adam accepted lazy_update with use_fused_step=False')
    """,
    """
    from mxnet.optimizer import LANS
    try:
        LANS(aggregate_num=46)
    except ValueError as err:
        if 'aggregate_num <= 45' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('LANS accepted aggregate_num=46')
    """,
    """
    from mxnet.optimizer import LAMB
    try:
        LAMB(aggregate_num=46)
    except ValueError as err:
        if 'aggregate_num <= 45' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('LAMB accepted aggregate_num=46')
    """,
    """
    from mxnet.contrib.text.vocab import Vocabulary
    from collections import Counter
    try:
        Vocabulary(counter=Counter({'a': 1}), min_freq=0)
    except ValueError as err:
        if 'min_freq' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Vocabulary accepted min_freq=0')
    """,
    """
    from mxnet.kvstore.kvstore import _get_kvstore_server_command_type
    try:
        _get_kvstore_server_command_type('kBogus')
    except ValueError as err:
        if 'Unknown command type' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('_get_kvstore_server_command_type accepted unknown command')
    """,
    """
    from mxnet.kvstore.kvstore import KVStore
    try:
        KVStore('not-a-handle')
    except TypeError as err:
        if 'KVStoreHandle' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('KVStore accepted non-handle')
    """,
    """
    from mxnet.optimizer import Optimizer
    try:
        Optimizer(param_idx2name='not-a-dict')
    except TypeError as err:
        if 'param_idx2name' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Optimizer accepted non-dict param_idx2name')
    """,
    """
    # XOP22 second wave: KVStore.save_optimizer_states must still reject the
    # 'no updater' case under python -O.
    from mxnet.kvstore.kvstore import KVStore
    kv = object.__new__(KVStore)
    kv._updater = None
    try:
        kv.save_optimizer_states('/tmp/should-not-be-written')
    except RuntimeError as err:
        if 'distributed training' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('save_optimizer_states accepted None updater')
    """,
    """
    # XOP22 second wave: KVStore.load_optimizer_states symmetric guard.
    from mxnet.kvstore.kvstore import KVStore
    kv = object.__new__(KVStore)
    kv._updater = None
    try:
        kv.load_optimizer_states('/tmp/should-not-be-read')
    except RuntimeError as err:
        if 'distributed training' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('load_optimizer_states accepted None updater')
    """,
    """
    # XOP22 second wave: BytePS KVStore key type must still be checked under -O.
    from mxnet.kvstore.byteps import BytePS
    kv = object.__new__(BytePS)
    try:
        kv.pushpull(1.5, None)
    except TypeError as err:
        if 'key must be str or int' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('BytePS pushpull accepted float key')
    """,
    """
    # XOP22 second wave: Gluon Parameter.set_data without prior init must
    # still raise instead of silently corrupting state under -O.
    from mxnet.gluon.parameter import Parameter
    p = Parameter('test_p', shape=(2, 3))
    # Don't initialize; deferred_init is empty by default.
    p._deferred_init = ()
    import mxnet as mx
    try:
        p.set_data(mx.nd.zeros((2, 3)))
    except RuntimeError as err:
        if 'has not been initialized' not in str(err):
            raise AssertionError(str(err))
    else:
        raise AssertionError('Parameter.set_data accepted uninitialized parameter')
    """,
])
def test_user_validation_survives_optimized_python(source):
    _run_optimized_python(source)
