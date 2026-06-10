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

"""Pytest plugin: keep only every Nth collected test, for GPU sharding.

Select the shard via environment variables:

* ``MXNET_TEST_SHARD_ID``   -- 0-based index of this shard (default 0)
* ``MXNET_TEST_NUM_SHARDS`` -- total number of shards (default 1, i.e. no-op)

Each shard process then runs a disjoint ~1/N slice of the same target files.
Sharding by collection index (rather than by passing parametrized node ids,
which can contain spaces/brackets) keeps the launcher robust.

Enable with ``pytest -p pytest_gpu_shard_plugin`` (ensure this file's directory
is on ``PYTHONPATH``). ``tools/run_gpu_shards.sh`` wires this up automatically.

For backward compatibility the legacy ``SHARD_ID`` / ``NUM_SHARDS`` variables are
still honoured when the ``MXNET_TEST_*`` ones are unset.
"""

import os


def _env_int(*names, default=0):
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return int(value)
    return default


def pytest_collection_modifyitems(config, items):
    num = _env_int("MXNET_TEST_NUM_SHARDS", "NUM_SHARDS", default=1)
    sid = _env_int("MXNET_TEST_SHARD_ID", "SHARD_ID", default=0)
    if num <= 1:
        return
    if not 0 <= sid < num:
        raise ValueError(
            f"shard id {sid} out of range for {num} shards")
    selected, deselected = [], []
    for idx, item in enumerate(items):
        (selected if idx % num == sid else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
