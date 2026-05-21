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

from mxnet.log import get_logger


def _fresh_logger():
    name = 'mxnet-test-log-' + uuid.uuid4().hex
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


def test_get_logger_file_handler_closes_stream_between_records(tmp_path):
    name, logger = _fresh_logger()
    log_path = tmp_path / 'mxnet.log'
    rotated_path = tmp_path / 'mxnet.log.1'
    try:
        configured = get_logger(name, filename=str(log_path), filemode='w', level=logging.INFO)

        assert configured is logger
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        assert logger.handlers[0].stream is None
        assert log_path.exists()

        logger.warning('first message')
        assert logger.handlers[0].stream is None

        log_path.rename(rotated_path)
        logger.warning('second message')

        assert 'first message' in rotated_path.read_text()
        assert 'second message' in log_path.read_text()
        assert logger.handlers[0].stream is None
    finally:
        _reset_logger(logger)


def test_get_logger_repeated_call_does_not_duplicate_file_handler(tmp_path):
    name, logger = _fresh_logger()
    log_path = tmp_path / 'mxnet.log'
    try:
        first = get_logger(name, filename=str(log_path), level=logging.INFO)
        second = get_logger(name, filename=str(log_path), level=logging.DEBUG)

        assert first is second
        assert len(logger.handlers) == 1

        logger.warning('only once')
        assert log_path.read_text().count('only once') == 1
    finally:
        _reset_logger(logger)
