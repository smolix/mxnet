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

import ctypes
import mxnet as mx
from numpy.testing import assert_equal
import mxnet.test_utils as test_utils
from mxnet import base
from mxnet.base import data_dir
from mxnet.test_utils import environment
from mxnet.util import getenv
from common import with_environment, retry
import os
import logging
import os.path as op
import platform
import pytest

@pytest.mark.garbage_expected
def test_environment():
    name1 = 'MXNET_TEST_ENV_VAR_1'
    name2 = 'MXNET_TEST_ENV_VAR_2'

    # Test that a variable can be set in the python and backend environment
    with environment(name1, '42'):
        assert_equal(os.environ.get(name1), '42')
        assert_equal(getenv(name1), '42')

    # Test dict form of invocation
    env_var_dict = {name1: '1', name2: '2'}
    with environment(env_var_dict):
        for key, value in env_var_dict.items():
            assert_equal(os.environ.get(key), value)
            assert_equal(getenv(key), value)

    # Further testing in 'test_with_environment()'

@with_environment({'MXNET_TEST_ENV_VAR_1': '10', 'MXNET_TEST_ENV_VAR_2': None})
def test_with_environment():
    name1 = 'MXNET_TEST_ENV_VAR_1'
    name2 = 'MXNET_TEST_ENV_VAR_2'
    def check_background_values():
        assert_equal(os.environ.get(name1), '10')
        assert_equal(getenv(name1), '10')
        assert_equal(os.environ.get(name2), None)
        assert_equal(getenv(name2), None)

    check_background_values()

    # This completes the testing of with_environment(), but since we have
    # an environment with a couple of known settings, lets use it to test if
    # 'with environment()' properly restores to these settings in all cases.

    class OnPurposeError(Exception):
        """A class for exceptions thrown by this test"""
        pass

    # Enter an environment with one variable set and check it appears
    # to both python and the backend.  Then, outside the 'with' block,
    # make sure the background environment is seen, regardless of whether
    # the 'with' block raised an exception.
    def test_one_var(name, value, raise_exception=False):
        try:
            with environment(name, value):
                assert_equal(os.environ.get(name), value)
                assert_equal(getenv(name), value)
                if raise_exception:
                    raise OnPurposeError
        except OnPurposeError:
            pass
        finally:
            check_background_values()

    # Test various combinations of set and unset env vars.
    # Test that the background setting is restored in the presense of exceptions.
    for raise_exception in [False, True]:
        # name1 is initially set in the environment
        test_one_var(name1, '42', raise_exception)
        test_one_var(name1, None, raise_exception)
        # name2 is initially not set in the environment
        test_one_var(name2, '42', raise_exception)
        test_one_var(name2, None, raise_exception)


def test_data_dir():
    prev_data_dir = data_dir()
    system = platform.system()
    # Test that data_dir() returns the proper default value when MXNET_HOME is not set
    with environment('MXNET_HOME', None):
        if system == 'Windows':
            assert_equal(data_dir(), op.join(os.environ.get('APPDATA'), 'mxnet'))
        else:
            assert_equal(data_dir(), op.join(op.expanduser('~'), '.mxnet'))
    # Test that data_dir() responds to an explicit setting of MXNET_HOME
    with environment('MXNET_HOME', '/tmp/mxnet_data'):
        assert_equal(data_dir(), '/tmp/mxnet_data')
    # Test that this test has not disturbed the MXNET_HOME value existing before the test
    assert_equal(data_dir(), prev_data_dir)


def test_environment_preserves_primary_exception_when_waitall_fails(monkeypatch):
    name = 'MXNET_TEST_ENV_VAR_PRIMARY_EXCEPTION'

    def fail_waitall():
        raise RuntimeError('async cleanup failure')

    monkeypatch.setattr(mx.nd, 'waitall', fail_waitall)
    with pytest.raises(AssertionError, match='primary failure'):
        with environment(name, 'set'):
            raise AssertionError('primary failure')
    assert os.environ.get(name) is None


def test_retry_reports_cleanup_error_with_original_assertion(monkeypatch):
    def fail_waitall():
        raise mx.base.MXNetError('async cleanup failure')

    monkeypatch.setattr(mx.nd, 'waitall', fail_waitall)

    @retry(2)
    def flaky():
        raise AssertionError('primary assertion')

    with pytest.raises(AssertionError, match='primary assertion.*async cleanup failure'):
        flaky()


def test_download_passes_timeout_to_requests(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        status_code = 200

        def iter_content(self, chunk_size=1024):
            yield b'ok'

    def fake_get(url, stream, timeout):
        calls.append((url, stream, timeout))
        return FakeResponse()

    monkeypatch.setattr(test_utils.requests, 'get', fake_get)
    fname = test_utils.download('https://example.invalid/file.bin',
                                dirname=str(tmp_path),
                                retries=1,
                                timeout=7)

    assert calls == [('https://example.invalid/file.bin', True, 7)]
    with open(fname, 'rb') as downloaded:
        assert downloaded.read() == b'ok'


def test_generate_op_module_signature_closes_files_on_codegen_failure(monkeypatch):
    opened_files = []

    class FakeLib:
        def __init__(self):
            self._op_names = (ctypes.c_char_p * 1)(b'broken_op')

        def MXListAllOpNames(self, size, plist):
            ctypes.cast(size, ctypes.POINTER(ctypes.c_uint))[0] = 1
            ctypes.cast(plist, ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)))[0] = (
                ctypes.cast(self._op_names, ctypes.POINTER(ctypes.c_char_p)))
            return 0

        def NNGetOpHandle(self, name, handle):
            return 0

    class FakeFile:
        def __init__(self, path):
            self.path = path
            self.closed = False
            self.writes = []

        def write(self, data):
            assert not self.closed
            self.writes.append(data)

        def close(self):
            assert not self.closed
            self.closed = True

    def fake_open(path, mode, encoding=None):
        assert mode == 'w'
        assert encoding == 'utf-8'
        opened_file = FakeFile(path)
        opened_files.append(opened_file)
        return opened_file

    def raise_codegen_error(handle, name, func_name, signature_only):
        raise RuntimeError('codegen failed')

    monkeypatch.setattr(base, '_LIB', FakeLib())
    monkeypatch.setattr(base, 'check_call', lambda ret: None)
    monkeypatch.setattr(base, 'open', fake_open, raising=False)

    with pytest.raises(RuntimeError, match='codegen failed'):
        base._generate_op_module_signature('mxnet', 'ndarray', raise_codegen_error)

    assert opened_files
    assert all(opened_file.closed for opened_file in opened_files)
