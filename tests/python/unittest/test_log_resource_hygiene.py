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

import logging
import uuid

import pytest

from mxnet.log import get_logger


def _fresh_logger():
    name = 'mxnet-test-log-resource-' + uuid.uuid4().hex
    logger = logging.getLogger(name)
    logger.propagate = False
    return name, logger


def _reset_logger(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if hasattr(logger, '_init_done'):
        delattr(logger, '_init_done')
    logger.propagate = True


def test_get_logger_removes_file_handler_when_setup_fails(tmp_path):
    name, logger = _fresh_logger()
    log_path = tmp_path / 'mxnet.log'
    try:
        for _ in range(2):
            with pytest.raises(TypeError):
                get_logger(name, filename=str(log_path), level={})
            assert logger.handlers == []
            assert not getattr(logger, '_init_done', False)

        configured = get_logger(name, filename=str(log_path), level=logging.INFO)
        assert configured is logger
        assert len(logger.handlers) == 1
    finally:
        _reset_logger(logger)
