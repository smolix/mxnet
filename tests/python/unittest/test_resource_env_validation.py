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

"""Process-isolated validation tests for resource-manager env vars."""

import os
import subprocess
import sys
import textwrap


def _expect_resource_env_rejected(env_name, value, body):
    code = f"""
import sys
import mxnet as mx
import numpy as np

try:
{textwrap.indent(body, '    ')}
except mx.base.MXNetError as err:
    message = str(err)
    if "{env_name}" in message and "positive" in message:
        sys.exit(0)
    print("wrong MXNetError: " + message)
    sys.exit(2)
except Exception as err:
    print(type(err).__name__ + ": " + str(err))
    sys.exit(3)

print("operation unexpectedly succeeded")
sys.exit(4)
"""
    env = os.environ.copy()
    env[env_name] = value
    env.setdefault("MXNET_ENGINE_TYPE", "NaiveEngine")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (
        f"{env_name}={value} was not rejected cleanly\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_cpu_temp_copy_zero_is_rejected_before_temp_space_use():
    _expect_resource_env_rejected(
        "MXNET_CPU_TEMP_COPY",
        "0",
        """
a = mx.nd.ones((8,))
b = mx.nd.ones((8,))
mx.nd.contrib.allclose(a, b).wait_to_read()
""",
    )


def test_cpu_temp_copy_negative_is_rejected_before_temp_space_use():
    _expect_resource_env_rejected(
        "MXNET_CPU_TEMP_COPY",
        "-1",
        """
a = mx.nd.ones((8,))
b = mx.nd.ones((8,))
mx.nd.contrib.allclose(a, b).wait_to_read()
""",
    )


def test_cpu_parallel_rand_copy_zero_is_rejected_before_random_resource_use():
    _expect_resource_env_rejected(
        "MXNET_CPU_PARALLEL_RAND_COPY",
        "0",
        """
x = mx.nd.ones((8,))
mx.nd.LeakyReLU(x, act_type='rrelu').wait_to_read()
""",
    )


def test_cpu_parallel_rand_copy_negative_is_rejected_before_random_resource_use():
    _expect_resource_env_rejected(
        "MXNET_CPU_PARALLEL_RAND_COPY",
        "-1",
        """
x = mx.nd.ones((8,))
mx.nd.LeakyReLU(x, act_type='rrelu').wait_to_read()
""",
    )
