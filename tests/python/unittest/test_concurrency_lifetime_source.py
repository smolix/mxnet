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

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def _read(rel):
    return (REPO / rel).read_text()


def test_legacy_ndarray_op_owns_callback_wrappers():
    contents = _read("src/operator/custom/ndarray_op.cc")

    assert "std::vector<std::unique_ptr<NDArray>> nd_wrappers" in contents
    assert "new NDArray(blob, ndctx.dev_id)" in contents
    assert "reinterpret_cast<void*>(new NDArray" not in contents


def test_threaded_engine_wait_ops_run_during_shutdown():
    header = _read("src/engine/threaded_engine.h")
    source = _read("src/engine/threaded_engine.cc")

    assert "!shutdown_phase_ || threaded_opr->wait" in header
    assert "std::make_shared<std::atomic<bool>>(false)" in source
    assert "[this, &done]" not in source


def test_nccl_kvstore_drains_engine_before_teardown():
    contents = _read("src/kvstore/kvstore_nccl.h")

    assert "Engine::Get()->WaitForAll();" in contents
