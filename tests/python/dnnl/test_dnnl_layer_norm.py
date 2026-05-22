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

import numpy as np
import mxnet as mx
from mxnet.test_utils import assert_almost_equal


def _layer_norm_grad(data, gamma, out_grad, eps):
    mean = data.mean(axis=-1, keepdims=True)
    var = data.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = (data - mean) / std
    gamma_grad = (normalized * out_grad).sum(axis=0)
    beta_grad = out_grad.sum(axis=0)
    data_grad_scale = out_grad * gamma.reshape((1, -1)) / std
    data_grad = (data_grad_scale - data_grad_scale.mean(axis=-1, keepdims=True) -
                 normalized * (data_grad_scale * normalized).mean(axis=-1, keepdims=True))
    return data_grad, gamma_grad, beta_grad


def test_dnnl_layer_norm_output_mean_std_visible():
    shape = (1024, 1024)
    eps = 1e-3
    rng = np.random.default_rng(1234)
    data_np = rng.normal(size=shape).astype("float32")
    gamma_np = rng.normal(size=(shape[-1],)).astype("float32")
    beta_np = rng.normal(size=(shape[-1],)).astype("float32")

    data = mx.sym.Variable("data")
    gamma = mx.sym.Variable("gamma")
    beta = mx.sym.Variable("beta")
    sym = mx.sym.LayerNorm(data, gamma, beta, axis=-1, eps=eps, output_mean_var=True)
    exe = sym._simple_bind(mx.cpu(), data=shape, grad_req="null")
    exe.arg_dict["data"][:] = data_np
    exe.arg_dict["gamma"][:] = gamma_np
    exe.arg_dict["beta"][:] = beta_np
    out, mean, std = exe.forward(is_train=False)

    mean_np = data_np.mean(axis=-1, keepdims=True)
    std_np = np.sqrt(data_np.var(axis=-1, keepdims=True) + eps)
    out_np = (data_np - mean_np) / std_np * gamma_np.reshape((1, -1)) + beta_np.reshape((1, -1))

    assert_almost_equal(out.asnumpy(), out_np, rtol=1e-5, atol=1e-5)
    assert_almost_equal(mean.asnumpy(), mean_np, rtol=1e-5, atol=1e-5)
    assert_almost_equal(std.asnumpy(), std_np, rtol=1e-5, atol=1e-5)


def test_dnnl_layer_norm_gamma_beta_grad_req_add():
    shape = (1024, 1024)
    eps = 1e-3
    rng = np.random.default_rng(5678)
    data_np = rng.normal(size=shape).astype("float32")
    gamma_np = rng.normal(size=(shape[-1],)).astype("float32")
    beta_np = rng.normal(size=(shape[-1],)).astype("float32")
    out_grad_np = rng.normal(size=shape).astype("float32")
    init_data_grad = rng.normal(size=shape).astype("float32")
    init_gamma_grad = rng.normal(size=(shape[-1],)).astype("float32")
    init_beta_grad = rng.normal(size=(shape[-1],)).astype("float32")

    data = mx.sym.Variable("data")
    gamma = mx.sym.Variable("gamma")
    beta = mx.sym.Variable("beta")
    sym = mx.sym.LayerNorm(data, gamma, beta, axis=-1, eps=eps)
    exe = sym._simple_bind(mx.cpu(), data=shape, grad_req="add")
    exe.arg_dict["data"][:] = data_np
    exe.arg_dict["gamma"][:] = gamma_np
    exe.arg_dict["beta"][:] = beta_np
    exe.grad_dict["data"][:] = init_data_grad
    exe.grad_dict["gamma"][:] = init_gamma_grad
    exe.grad_dict["beta"][:] = init_beta_grad
    exe.forward(is_train=True)
    exe.backward([mx.nd.array(out_grad_np)])

    data_grad_np, gamma_grad_np, beta_grad_np = _layer_norm_grad(data_np, gamma_np, out_grad_np, eps)
    assert_almost_equal(exe.grad_dict["data"].asnumpy(), data_grad_np + init_data_grad,
                        rtol=1e-4, atol=1e-4)
    assert_almost_equal(exe.grad_dict["gamma"].asnumpy(), gamma_grad_np + init_gamma_grad,
                        rtol=1e-4, atol=1e-4)
    assert_almost_equal(exe.grad_dict["beta"].asnumpy(), beta_grad_np + init_beta_grad,
                        rtol=1e-4, atol=1e-4)
