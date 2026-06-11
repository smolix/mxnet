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

"""Executable repros for still-open apache/mxnet issue reports.

Each test below encodes the behavior MXNet should have after the issue is
fixed.  The tests are xfailed while the corresponding issue is still known to
reproduce on the current wheel.  Running this file with ``pytest --runxfail``
turns these into hard failures and is the quickest way to verify that the bug
still exists.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def issue_xfail(issue, reason):
    return pytest.mark.xfail(
        strict=True,
        reason="apache/mxnet#{}: {}".format(issue, reason),
    )


def run_python(code, timeout=30, extra_env=None):
    env = os.environ.copy()
    env.setdefault("MXNET_CUDNN_LIB_CHECKING", "0")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-c", textwrap.dedent(code)]
    try:
        return subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        stdout = err.stdout or ""
        stderr = err.stderr or ""
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout,
            stderr + "\nsubprocess timed out after {} seconds".format(timeout),
        )


def require_gpus(count):
    import mxnet as mx

    try:
        available = mx.context.num_gpus()
    except Exception as err:
        pytest.skip("GPU discovery failed: {}".format(err))
    if available < count:
        pytest.skip("requires {} GPUs, found {}".format(count, available))


def assert_subprocess_ok(proc):
    assert proc.returncode == 0, (
        "returncode={}\nstdout:\n{}\nstderr:\n{}".format(
            proc.returncode, proc.stdout, proc.stderr
        )
    )


def repo_root():
    return Path(__file__).resolve().parents[3]


@issue_xfail(21217, "Horovod KVStore API has no barrier method")
def test_pr_21217_horovod_kvstore_exposes_barrier():
    from mxnet.kvstore.horovod import Horovod

    assert hasattr(Horovod, "_barrier")


@issue_xfail(21176, "CPU Conv2D dispatch accepts NHWC and then oneDNN rejects it")
def test_issue_21176_conv2d_nhwc_cpu_runs():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    net = mx.gluon.nn.Conv2D(channels=32, kernel_size=(3, 3), layout="NHWC")
    net.initialize()
    out = net(mxnp.ones((10, 28, 28, 1)))
    out.wait_to_read()
    assert out.shape == (10, 26, 26, 32)


@issue_xfail(21044, "SymbolBlock drops user-specified symbol parameter attributes")
def test_pr_21044_symbolblock_preserves_symbol_parameter_attrs():
    import mxnet as mx
    import mxnet.symbol as mxs

    lr_mult = 0.555
    wd_mult = 0.444
    data = mxs.var("x", shape=(1, 256), dtype=np.float32)
    weight = mxs.var("W", shape=(256, 192), lr_mult=lr_mult, wd_mult=wd_mult, dtype=np.float32)
    bias = mxs.var("b", shape=(1, 192), dtype=np.float32, init=mx.init.Zero())
    out = mxs.linalg.gemm(data, weight, bias)
    block = mx.gluon.SymbolBlock([out], [data])
    block.initialize()
    block(mx.nd.random_uniform(-1.0, 1.0, shape=(1, 256), dtype=np.float32)).wait_to_read()
    params = block.collect_params()
    assert params["W"].lr_mult == lr_mult
    assert params["W"].wd_mult == wd_mult
    assert not params["b"].data().asnumpy().any()


@issue_xfail(21119, "cross-GPU NumPy binary ops hang instead of copying or rejecting contexts")
def test_issue_21119_cross_gpu_binary_op_does_not_hang():
    require_gpus(2)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        npx.set_np()
        left = mxnp.ones((1,), ctx=mx.gpu(0))
        right = mxnp.ones((1,), ctx=mx.gpu(1))
        try:
            out = left + right
            out.wait_to_read()
            onp.testing.assert_allclose(out.asnumpy(), onp.array([2.0]))
        except Exception as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        """,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


@issue_xfail(21111, "cuDNN BatchNorm CachedOp mutates state during forward-only train mode")
def test_issue_21111_cudnn_batchnorm_cachedop_forward_only_train_mode_is_stateless():
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx
        from mxnet import autograd
        from mxnet import np as mxnp
        from mxnet import npx

        npx.set_np()
        ctx = mx.gpu(0)
        data = mx.sym.var("data")
        gamma = mx.sym.var("gamma")
        beta = mx.sym.var("beta")
        moving_mean = mx.sym.var("moving_mean")
        moving_var = mx.sym.var("moving_var")
        sym = mx.sym.BatchNorm(
            data=data,
            gamma=gamma,
            beta=beta,
            moving_mean=moving_mean,
            moving_var=moving_var,
            fix_gamma=False,
            use_global_stats=False,
            cudnn_off=False,
            momentum=0.0,
            eps=1e-5,
        )
        op = mx.nd.CachedOp(sym)
        x = mxnp.random.uniform(size=(1, 6, 1), ctx=ctx)
        scale = mxnp.ones((6,), ctx=ctx)
        offset = mxnp.zeros((6,), ctx=ctx)
        mean = mxnp.zeros((6,), ctx=ctx)
        var = mxnp.ones((6,), ctx=ctx)
        with autograd.predict_mode():
            before = op(x, scale, offset, mean, var)
        before.wait_to_read()
        with autograd.train_mode():
            train_out = op(x, scale, offset, mean, var)
        train_out.wait_to_read()
        with autograd.predict_mode():
            after = op(x, scale, offset, mean, var)
        after.wait_to_read()
        assert onp.isfinite(var.asnumpy()).all()
        onp.testing.assert_allclose(after.asnumpy(), before.asnumpy(), rtol=1e-6, atol=1e-6)
        """,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


@issue_xfail(21156, "RecordIO shutdown path depends on module globals used by super()")
def test_issue_21156_indexed_recordio_close_survives_module_teardown():
    import mxnet.recordio as recordio

    cls = recordio.MXIndexedRecordIO
    obj = cls.__new__(cls)
    obj.is_open = True
    obj.fidx = None

    # Interpreter shutdown can clear module globals before __del__ calls close().
    try:
        recordio.MXIndexedRecordIO = None
        obj.close()
    finally:
        recordio.MXIndexedRecordIO = cls
        obj.is_open = False


@issue_xfail(21146, "GRU use_sequence_length is passed positionally during deferred init")
def test_issue_21146_gru_deferred_init_with_sequence_length_runs():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx
    from mxnet.gluon import rnn

    npx.set_np()
    net = rnn.GRU(
        hidden_size=1,
        num_layers=2,
        bidirectional=True,
        layout="NTC",
        use_sequence_length=True,
    )
    net.initialize()
    out, state = net(mxnp.random.uniform(size=(2, 3, 2)), sequence_length=mxnp.array([1, 2]))
    out.wait_to_read()
    assert out.shape == (2, 3, 2)
    assert state.shape == (4, 2, 1)


@issue_xfail(20936, "wheel does not include headers reachable by find_include_path()")
def test_issue_20936_wheel_exposes_include_path():
    import mxnet as mx

    paths = mx.libinfo.find_include_path()
    assert paths
    assert any(os.path.exists(os.path.join(path, "mxnet", "base.h")) for path in paths)


@issue_xfail(20657, "find_conf_path returns a string for MXNET_CONF_PATH")
def test_issue_20657_find_conf_path_env_override_is_sequence(tmp_path, monkeypatch):
    from mxnet.libinfo import find_conf_path

    conf = tmp_path / "tvmop.conf"
    conf.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MXNET_CONF_PATH", str(conf))
    paths = find_conf_path("tvmop")
    assert isinstance(paths, list)
    assert paths == [str(conf)]


@issue_xfail(20605, "CSRNDArray gradients densify the sparse pattern")
def test_issue_20605_csr_gradient_preserves_sparse_pattern():
    import mxnet as mx
    from mxnet import autograd

    scipy_sparse = pytest.importorskip("scipy.sparse")
    source = scipy_sparse.diags([1, 2, 3], dtype="float64", format="csr")
    data = mx.nd.sparse.array(source, dtype="float64")
    vector = mx.nd.ones((3, 1), dtype="float64")
    data.attach_grad()
    vector.attach_grad()
    with autograd.record():
        loss = mx.nd.sparse.dot(data, vector).sum()
    loss.backward()
    assert data.grad.stype == "csr"
    np.testing.assert_allclose(data.grad.asnumpy(), np.eye(3))


@issue_xfail(20577, "SymbolBlock export reads missing _cached_op_args")
def test_issue_20577_symbolblock_export_succeeds_without_cached_op_args(tmp_path):
    import mxnet as mx

    data = mx.sym.Variable("data")
    block = mx.gluon.SymbolBlock([mx.sym.relu(data)], [data])
    prefix = str(tmp_path / "symbolblock")
    block.export(prefix)
    assert (tmp_path / "symbolblock-symbol.json").exists()


@issue_xfail(20391, "NumPy/Gluon 2.0 rejects row_sparse gradients")
def test_issue_20391_numpy_gluon_allows_row_sparse_gradients():
    from mxnet import npx
    from mxnet.gluon import Parameter

    npx.set_np()
    param = Parameter(
        "embed_weight",
        shape=(10, 3),
        stype="row_sparse",
        grad_stype="row_sparse",
    )
    param.initialize()
    assert param.grad().stype == "row_sparse"


@issue_xfail(20491, "C++ Symbol API lacks OptimizeForBackend support")
def test_pr_20491_cpp_symbol_exposes_optimize_for_backend():
    symbol_header = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.h"
    symbol_impl = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.hpp"
    text = symbol_header.read_text(encoding="utf-8") + symbol_impl.read_text(encoding="utf-8")
    assert "OptimizeForBackend" in text


@issue_xfail(20037, "scalar RecordIO labels are packed as float32 and lose integer precision")
def test_issue_20037_recordio_preserves_large_integer_label():
    from mxnet.recordio import IRHeader, pack, unpack

    header = IRHeader(0, 17672687.0, 1, 0)
    round_tripped = unpack(pack(header, b"x"))[0]
    assert round_tripped.label == header.label


@issue_xfail(20180, "box_encode with zero reference boxes fails with an internal TBlob error")
def test_issue_20180_box_encode_zero_refs_is_validated_or_empty():
    proc = run_python(
        """
        import mxnet as mx

        try:
            anchors = mx.nd.ones((1, 1, 4))
            refs = mx.nd.empty((1, 0, 4))
            out = mx.nd.contrib.box_encode(anchors, refs, samples=mx.nd.ones((1, 1)))
            out.wait_to_read()
            assert out.shape[0] == 1
        except Exception as err:
            msg = str(err).lower()
            assert "refs" in msg and ("empty" in msg or "zero" in msg or "shape" in msg)
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(20076, "SequenceMask accepts sequence_length values that overflow indexing")
def test_issue_20076_sequence_mask_rejects_huge_lengths_cleanly():
    proc = run_python(
        """
        import mxnet as mx

        try:
            data = mx.nd.ones((1, 2))
            seq = mx.nd.array([2147483647, 2147483647], dtype="int64")
            out = mx.nd.SequenceMask(data, sequence_length=seq, use_sequence_length=True)
            out.wait_to_read()
        except Exception as err:
            msg = str(err).lower()
            assert "sequence" in msg and ("length" in msg or "range" in msg)
        else:
            raise AssertionError("huge sequence_length should be rejected")
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(20046, "image.resize forwards invalid interpolation ids into OpenCV")
def test_issue_20046_image_resize_invalid_interp_has_mxnet_validation():
    import mxnet as mx

    try:
        mx.nd.image.resize(mx.nd.ones((4, 4, 3), dtype="uint8"), size=(2, 2), interp=10).wait_to_read()
    except Exception as err:
        msg = str(err).lower()
        assert "interp" in msg
        assert "opencv" not in msg
    else:
        raise AssertionError("invalid interpolation id should be rejected")


@issue_xfail(20044, "boolean_mask with empty inputs and out= crashes asynchronously")
def test_issue_20044_boolean_mask_empty_out_is_safe():
    proc = run_python(
        """
        import mxnet as mx

        try:
            data = mx.nd.ones((0, 2))
            index = mx.nd.empty((0,), dtype="bool")
            out = mx.nd.empty((0, 2))
            mx.nd.contrib.boolean_mask(data, index, out=out).wait_to_read()
            assert out.shape == (0, 2)
        except Exception as err:
            msg = str(err).lower()
            assert "empty" in msg or "zero" in msg or "shape" in msg
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(19860, "Swish with very negative beta returns NaN for zero input")
def test_issue_19860_swish_negative_beta_zero_input_is_finite():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    net = mx.gluon.nn.Swish(beta=-1e307)
    net.initialize()
    out = net(mxnp.zeros((1,)))
    out.wait_to_read()
    assert np.isfinite(out.asnumpy()).all()
    np.testing.assert_allclose(out.asnumpy(), np.array([0.0]))


@issue_xfail(19852, "InstanceNorm overflows finite large inputs into NaN outputs")
def test_issue_19852_instancenorm_large_finite_input_is_finite():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    layer = mx.gluon.nn.InstanceNorm()
    layer.initialize()
    data = mxnp.array([[[[1.4918449e38], [9.0072335e37], [-1.3146734e38], [3.0568930e38]]]])
    out = layer(data)
    out.wait_to_read()
    assert np.isfinite(out.asnumpy()).all()


@issue_xfail(19785, "GroupNorm num_groups=0 aborts the process with SIGFPE")
def test_issue_19785_groupnorm_zero_groups_is_python_error_not_abort():
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np, npx
        from mxnet.gluon.nn import GroupNorm

        npx.set_np()
        try:
            net = GroupNorm(num_groups=0)
            net.initialize()
            net(np.ones((1, 4, 2))).wait_to_read()
        except Exception as err:
            assert "num_groups" in str(err) or "groups" in str(err)
        else:
            raise AssertionError("num_groups=0 should be rejected")
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(19753, "TopK ret_typ='both' returns float indices")
def test_issue_19753_topk_indices_are_integer_typed():
    import mxnet as mx

    values, indices = mx.nd.topk(mx.nd.array([3, 1, 2]), ret_typ="both")
    values.wait_to_read()
    indices.wait_to_read()
    assert np.issubdtype(indices.asnumpy().dtype, np.integer)


@issue_xfail(19628, "GPU CTCLoss rejects FP16 predictions through an internal dtype mismatch")
def test_issue_19628_gpu_ctcloss_accepts_fp16_predictions():
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        npx.set_np()
        ctx = mx.gpu(0)
        loss = mx.gluon.loss.CTCLoss(layout="NTC", label_layout="NT")
        pred = mxnp.random.uniform(size=(2, 4, 5), ctx=ctx, dtype="float16")
        label = mxnp.array([[1, 2], [2, 1]], ctx=ctx, dtype="float32")
        pred_lengths = mxnp.array([4, 4], ctx=ctx, dtype="float32")
        label_lengths = mxnp.array([2, 2], ctx=ctx, dtype="float32")
        out = loss(pred, label, pred_lengths, label_lengths)
        out.wait_to_read()
        assert out.shape == (2,)
        assert onp.isfinite(out.asnumpy()).all()
        """,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


@issue_xfail(19659, "hybridized boolean_mask backward lacks required backward inputs")
def test_issue_19659_hybrid_boolean_mask_backward_runs():
    import mxnet as mx
    from mxnet import nd
    from mxnet import np as mxnp
    from mxnet import npx
    from mxnet.gluon import HybridBlock

    npx.set_np()

    class Foo(HybridBlock):
        def forward(self, data, indices):
            mask = indices < 3
            data = npx.reshape(data, (-1, -2), reverse=True)
            mask = mxnp.reshape(mask, (-1,))
            return nd.np._internal.boolean_mask(data, mask)

    data = mxnp.random.normal(0, 1, (5, 5, 5, 5, 16))
    indices = mxnp.random.randint(0, 5, (5, 5, 5, 5))
    data.attach_grad()
    indices.attach_grad()
    block = Foo()
    block.hybridize()
    with mx.autograd.record():
        out = block(data, indices)
    out.backward()
    out.wait_to_read()
    assert np.isfinite(data.grad.asnumpy()).all()


@issue_xfail(19686, "interleaved_matmul_selfatt_qk divides by heads without validation")
def test_issue_19686_selfatt_qk_rejects_zero_heads_cleanly():
    proc = run_python(
        """
        import mxnet as mx

        try:
            out = mx.nd.contrib.interleaved_matmul_selfatt_qk(mx.nd.ones((2, 1, 3)), heads=0)
            out.wait_to_read()
        except Exception as err:
            assert "heads" in str(err).lower()
        else:
            raise AssertionError("heads=0 should be rejected")
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(19683, "arange_like repeat=0 aborts instead of producing/validating empty output")
def test_issue_19683_arange_like_repeat_zero_is_safe():
    proc = run_python(
        """
        import mxnet as mx

        try:
            out = mx.nd.contrib.arange_like(mx.nd.ones((1,)), repeat=0)
            out.wait_to_read()
            assert out.size == 0
        except Exception as err:
            assert "repeat" in str(err).lower()
        """
    )
    assert_subprocess_ok(proc)


@issue_xfail(19647, "Symbol.optimize_for logs missing backends but still returns a symbol")
def test_issue_19647_optimize_for_missing_backend_raises():
    import mxnet as mx

    sym = mx.sym.relu(mx.sym.var("data"))
    try:
        sym.optimize_for("definitely_missing_backend")
    except Exception as err:
        assert "backend" in str(err).lower()
    else:
        raise AssertionError("missing backend should raise")


@issue_xfail(19423, "choice(size=n, replace=False) leaves the range unshuffled")
def test_issue_19423_choice_full_without_replacement_is_permutation():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    saw_non_identity = False
    for seed in range(5):
        mx.random.seed(seed)
        sample = mxnp.random.choice(5, size=5, replace=False)
        sample.wait_to_read()
        saw_non_identity |= not np.array_equal(sample.asnumpy(), np.arange(5))
    assert saw_non_identity


@issue_xfail(19458, "tensordot backward mishandles scalar input with explicit empty axes")
def test_issue_19458_tensordot_scalar_empty_axes_backward():
    from mxnet import autograd
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    left = mxnp.array(np.array(2.0, dtype="float32"))
    right = mxnp.array(np.arange(1, 513, dtype="float32"))
    left.attach_grad()
    right.attach_grad()
    with autograd.record():
        loss = mxnp.tensordot(left, right, axes=([], [])).sum()
    loss.backward()
    left.grad.wait_to_read()
    right.grad.wait_to_read()
    np.testing.assert_allclose(left.grad.asnumpy(), np.array(131328.0, dtype=np.float32))
    np.testing.assert_allclose(right.grad.asnumpy(), np.full((512,), 2.0, dtype=np.float32))


@issue_xfail(19422, "NumPy ndarray iteration yields MXNet scalar arrays instead of Python scalars")
def test_issue_19422_numpy_array_iteration_yields_python_scalars():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    items = list(mxnp.arange(3, dtype="int64"))
    assert all(isinstance(item, (int, np.integer)) for item in items)
    assert items == [0, 1, 2]


@issue_xfail(19170, "stepped NumPy slicing returns a copy instead of a view")
def test_issue_19170_stepped_slice_shares_storage():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    base = mxnp.ones((6,))
    view = base[:5:2]
    base[:] = 0
    view.wait_to_read()
    np.testing.assert_allclose(view.asnumpy(), np.zeros((3,)))


@issue_xfail(18583, "C++ Symbol API does not expose partial shape inference")
def test_pr_18583_cpp_symbol_exposes_partial_shape_inference():
    symbol_header = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.h"
    symbol_impl = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.hpp"
    text = symbol_header.read_text(encoding="utf-8") + symbol_impl.read_text(encoding="utf-8")
    assert "InferShapePartial" in text


@issue_xfail(19021, "backward accepts head gradients with shapes incompatible with the output")
def test_issue_19021_backward_rejects_mismatched_head_gradient_shape():
    from mxnet import autograd
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    left = mxnp.array([[2.0, 3.0], [5.0, 6.0]])
    right = mxnp.array([[3.0, 4.0], [7.0, 8.0]])
    left.attach_grad()
    right.attach_grad()
    with autograd.record():
        out = left * right
    try:
        out.backward(mxnp.array([1.5]))
    except Exception as err:
        assert "shape" in str(err).lower()
    else:
        raise AssertionError("mismatched head gradient shape should be rejected")


@issue_xfail(18919, "NumPy advanced indexing does not broadcast mixed index arrays")
def test_issue_18919_numpy_advanced_indexing_matches_numpy():
    import mxnet as mx
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    batch_size = 2
    seq_length = 4
    num_positions = 2
    data = mxnp.arange(batch_size * seq_length * 3).reshape((batch_size, seq_length, 3))
    positions = mxnp.array([[0, 2], [1, 3]], dtype="int32")
    batch = mxnp.expand_dims(mx.npx.arange_like(data, axis=0).astype("int32"), axis=1)
    actual = data[batch, positions]
    expected = data.asnumpy()[np.expand_dims(np.arange(batch_size, dtype=np.int32), axis=1),
                               positions.asnumpy()]
    np.testing.assert_allclose(actual.asnumpy(), expected)


@issue_xfail(18770, "NumPy byte order is silently discarded instead of preserved or rejected")
def test_issue_18770_non_native_byte_order_is_not_silently_lost():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    source = np.array([1, 2, 3], dtype=np.dtype(">i4"))
    try:
        out = mxnp.array(source)
    except (TypeError, ValueError):
        return
    assert out.asnumpy().dtype.byteorder == ">"


@issue_xfail(18792, "sort and argsort do not support float16 tensors")
def test_pr_18792_sort_and_argsort_support_float16():
    import mxnet as mx

    data = mx.nd.array([3, 1, 2], dtype="float16")
    sorted_data = mx.nd.sort(data)
    indices = mx.nd.argsort(data)
    sorted_data.wait_to_read()
    indices.wait_to_read()
    np.testing.assert_allclose(sorted_data.asnumpy(), np.array([1, 2, 3], dtype=np.float16))
    np.testing.assert_allclose(indices.asnumpy(), np.array([1, 2, 0]))


@issue_xfail(18669, "ZoneoutCell returns output inconsistent with the first recurrent state")
def test_issue_18669_zoneout_output_matches_new_state():
    import mxnet as mx
    from mxnet import autograd
    from mxnet.gluon.rnn import GRUCell, ZoneoutCell

    mx.random.seed(7)
    cell = ZoneoutCell(
        GRUCell(hidden_size=4, input_size=5),
        zoneout_outputs=0.5,
        zoneout_states=0.0,
    )
    cell.initialize()
    inputs = mx.nd.ones((2, 5))
    states = cell.begin_state(batch_size=2)
    with autograd.record(train_mode=True):
        out, new_states = cell(inputs, states)
    out.wait_to_read()
    np.testing.assert_allclose(out.asnumpy(), new_states[0].asnumpy())


@issue_xfail(18563, "max/min backward gives full gradient to every tied extremum")
def test_issue_18563_max_backward_splits_tied_gradient():
    import mxnet as mx
    from mxnet import autograd

    data = mx.nd.array([2.0, 2.0])
    data.attach_grad()
    with autograd.record():
        out = mx.nd.max(data)
    out.backward()
    np.testing.assert_allclose(data.grad.asnumpy(), np.array([0.5, 0.5]))


@issue_xfail(18078, "prod backward computes 0/0 for multiple zero inputs")
def test_issue_18078_prod_backward_multiple_zeros_is_finite():
    import mxnet as mx
    from mxnet import autograd

    data = mx.nd.array([0.0, 0.0])
    data.attach_grad()
    with autograd.record():
        out = mx.nd.prod(data)
    out.backward()
    np.testing.assert_allclose(data.grad.asnumpy(), np.array([0.0, 0.0]))


@issue_xfail(11774, "BatchNorm(scale=False, center=False) backward loses the graph")
def test_issue_11774_batchnorm_without_scale_or_center_trains():
    import mxnet as mx
    from mxnet import autograd
    from mxnet import gluon
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    data = mxnp.ones((2, 3, 4, 4))
    net = gluon.nn.Sequential()
    net.add(gluon.nn.BatchNorm(scale=False, center=False))
    net.initialize(mx.init.Normal(sigma=0.1), ctx=mx.cpu())
    loss_fn = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), "sgd")
    with autograd.record():
        loss = loss_fn(net(data[0]), data[0])
    loss.backward(retain_graph=True)
    trainer.step(32)


@issue_xfail(18300, "mxnet.numpy.prod rejects shape tuples")
def test_issue_18300_numpy_prod_accepts_shape_tuple():
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    data = mx.np.ones((5,))
    result = mx.np.prod(data.shape)
    assert int(result) == 5


@issue_xfail(17209, "Gluon Parameter variables still encode a fixed dtype")
def test_pr_17209_parameter_symbol_var_omits_dtype_attribute():
    from mxnet.gluon import Parameter

    param = Parameter("weight", shape=(2, 2), dtype="float32")
    attrs = next(iter(param.var().attr_dict().values()))
    assert "__dtype__" not in attrs


@issue_xfail(17936, "gammaln keeps integer output dtype and truncates lgamma")
def test_issue_17936_gammaln_promotes_integer_input():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    out = npx.gammaln(mxnp.array([1, 2, 3], dtype="int32"))
    out.wait_to_read()
    assert np.issubdtype(out.asnumpy().dtype, np.floating)
    np.testing.assert_allclose(out.asnumpy(), np.array([0.0, 0.0, np.log(2.0)]), rtol=1e-6)


@issue_xfail(17698, "split_and_load first materializes the full NumPy input on ctx_list[0]")
def test_issue_17698_split_and_load_does_not_materialize_full_input_first(monkeypatch):
    import mxnet as mx
    from mxnet.gluon import utils

    calls = []
    original_array = utils._mx_np.array

    def spy_array(data, *args, **kwargs):
        calls.append(tuple(getattr(data, "shape", np.asarray(data).shape)))
        return original_array(data, *args, **kwargs)

    monkeypatch.setattr(utils._mx_np, "array", spy_array)
    utils.split_and_load(np.arange(8), [mx.cpu(0), mx.cpu(0)], even_split=True)
    assert (8,) not in calls


@issue_xfail(16402, "legacy NDArray.dtype returns a NumPy scalar type instead of numpy.dtype")
def test_issue_16402_legacy_ndarray_dtype_is_numpy_dtype_object():
    import mxnet as mx

    data = mx.nd.array([1], dtype="float32")
    assert isinstance(data.dtype, np.dtype)


@issue_xfail(16427, "recordio.pack concatenates Python 3 str with bytes")
def test_issue_16427_recordio_pack_accepts_python3_string_payload():
    from mxnet import recordio

    header = recordio.IRHeader(0, 4, 2574, 0)
    packed = recordio.pack(header, "")
    unpacked_header, payload = recordio.unpack(packed)
    assert unpacked_header.id == 2574
    assert payload == b""


@issue_xfail(13953, "UpSampling symbolic wrapper does not accept the data keyword")
def test_issue_13953_upsampling_accepts_data_keyword():
    import mxnet as mx

    data = mx.symbol.Variable("data")
    sym = mx.symbol.UpSampling(data=data, scale=16, sample_type="nearest")
    assert sym.list_arguments() == ["data"]


@issue_xfail(13945, "MXIndexedRecordIO read_idx is not thread-safe on shared readers")
def test_issue_13945_indexed_recordio_shared_reader_is_thread_safe(tmp_path):
    import concurrent.futures
    import random

    from mxnet import recordio

    idx_path = str(tmp_path / "threaded.idx")
    rec_path = str(tmp_path / "threaded.rec")
    writer = recordio.MXIndexedRecordIO(idx_path, rec_path, "w")
    expected = {}
    try:
        for idx in range(1000):
            payload = ("record-%04d-" % idx).encode("ascii") + bytes([idx % 251]) * 128
            expected[idx] = payload
            writer.write_idx(idx, payload)
    finally:
        writer.close()

    reader = recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
    keys = list(expected)

    def read_one(key):
        return key, reader.read_idx(key)

    try:
        for attempt in range(20):
            random.Random(attempt).shuffle(keys)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                for key, payload in pool.map(read_one, keys):
                    assert payload == expected[key]
    finally:
        reader.close()


@issue_xfail(13193, "sparse elemwise_mul leaves CSR payload unordered and overallocated")
def test_issue_13193_sparse_elemwise_mul_has_canonical_csr_payload():
    import mxnet as mx

    left = mx.nd.array([[1, 2, 3, 0, 0, 0, 0]]).tostype("csr")
    right = mx.nd.array([[0, 1, 2, 3, 4, 0, 0]]).tostype("csr")
    out = mx.nd.sparse.elemwise_mul(left, right)
    nnz = int(out.indptr.asnumpy()[-1])
    assert out.data.shape[0] == nnz
    np.testing.assert_allclose(out.data.asnumpy(), np.array([2.0, 6.0]))
    np.testing.assert_array_equal(out.indices.asnumpy(), np.array([1, 2]))
    np.testing.assert_array_equal(out.indptr.asnumpy(), np.array([0, 2]))


@issue_xfail(8430, "NDArrayIter does not preserve NumPy label dtype")
def test_issue_8430_ndarrayiter_preserves_integer_label_dtype():
    import mxnet as mx

    data = np.zeros((2, 1), dtype=np.float32)
    labels = np.array([2 ** 40 + 1, 2 ** 40 + 3], dtype=np.int64)
    for shuffle in (False, True):
        iterator = mx.io.NDArrayIter(data=data, label=labels, batch_size=2, shuffle=shuffle)
        batch = next(iter(iterator))
        actual = batch.label[0].asnumpy()
        assert actual.dtype == labels.dtype
        np.testing.assert_array_equal(np.sort(actual), np.sort(labels))


@issue_xfail(12286, "generated NDArray wrappers accept invalid calls until backend invocation")
def test_issue_12286_ndarray_wrapper_raises_python_typeerror_for_missing_inputs():
    import mxnet as mx

    with pytest.raises(TypeError):
        mx.nd.softmax()


@issue_xfail(8817, "sparse zeros does not accept one-dimensional integer shapes")
def test_issue_8817_sparse_zeros_accepts_integer_shape():
    import mxnet as mx

    data = mx.nd.sparse.zeros("csr", shape=10, ctx=mx.cpu())
    assert data.shape == (10,)
    assert data.stype == "csr"


@issue_xfail(14695, "single-output legacy NDArray result can be unpacked as multiple values")
def test_issue_14695_single_output_ndarray_is_not_tuple_unpackable():
    import mxnet as mx

    result = mx.nd.split(mx.nd.ones((2, 1)), num_outputs=1, squeeze_axis=True)
    first, *rest = result
    first.wait_to_read()
    assert rest == []
