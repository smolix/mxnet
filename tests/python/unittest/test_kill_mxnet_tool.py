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

import importlib.util
import shlex
from pathlib import Path
from types import SimpleNamespace


def _load_tool():
    module_path = Path(__file__).resolve().parents[3] / "tools" / "kill-mxnet.py"
    spec = importlib.util.spec_from_file_location("kill_mxnet", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kill_local_uses_ps_and_kill_argument_lists(monkeypatch):
    module = _load_tool()
    calls = []
    ps_output = """alice 101 python train.py
bob 202 python train.py
alice 303 bash
alice 404 mxnet worker
alice 999 python kill-mxnet.py
"""

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[:2] == ["ps", "-eo"]:
            return SimpleNamespace(stdout=ps_output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.os, "getpid", lambda: 999)

    module._kill_local("alice", "python")

    assert calls[0][0] == ["ps", "-eo", "user=,pid=,args="]
    assert calls[1][0] == ["kill", "-9", "101"]
    assert all(not isinstance(call[0], str) for call in calls)


def test_remote_command_quotes_user_and_program_arguments():
    module = _load_tool()
    command = module._remote_command("alice; touch user-pwned", "train.py' ; touch prog-pwned")

    argv = shlex.split(command)

    assert argv[0:3] == ["sh", "-c", module.REMOTE_SCRIPT]
    assert argv[3:] == ["sh", "alice; touch user-pwned", "train.py' ; touch prog-pwned"]
    assert "ps aux |" not in command
    assert "xargs kill" not in command


def test_main_starts_remote_kills_and_local_kill(monkeypatch, tmp_path):
    module = _load_tool()
    host_file = tmp_path / "hosts"
    host_file.write_text("worker1:2222\n\nworker2\n")
    popen_calls = []
    local_calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            popen_calls.append((argv, kwargs))
            self.returncode = 0

        def communicate(self, timeout=None):
            assert timeout == module.REMOTE_TIMEOUT_SECONDS
            return b"", b""

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(module, "_kill_local", lambda user, prog: local_calls.append((user, prog)))

    assert module.main(["kill-mxnet.py", str(host_file), "alice", "mxnet-worker"]) == 0

    assert [call[0][2] for call in popen_calls] == ["worker1", "worker2"]
    assert all(call[0][0:2] == ["ssh", "-oStrictHostKeyChecking=no"] for call in popen_calls)
    assert all(call[1]["shell"] is False for call in popen_calls)
    assert local_calls == [("alice", "mxnet-worker")]
