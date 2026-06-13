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




def similar_bug_xfail(name, reason):
    return pytest.mark.xfail(
        strict=True,
        reason="similar bug sweep {}: {}".format(name, reason),
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


def require_nccl():
    import mxnet as mx

    if not mx.runtime.Features().is_enabled("NCCL"):
        pytest.skip("MXNet was built without NCCL")


def assert_subprocess_ok(proc):
    assert proc.returncode == 0, (
        "returncode={}\nstdout:\n{}\nstderr:\n{}".format(
            proc.returncode, proc.stdout, proc.stderr
        )
    )


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_pr_21217_horovod_kvstore_exposes_barrier():
    from mxnet.kvstore.horovod import Horovod

    assert hasattr(Horovod, "_barrier")


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
        except ValueError as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type") from err
        """,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


def test_similar_nccl_updater_cpu_gradient_matches_gpu_weight_context():
    require_nccl()
    require_gpus(1)
    proc = run_python(
        """
        import mxnet as mx

        seen = []
        mismatch = []

        def updater(key, grad, weight):
            contexts = (str(grad.context), str(weight.context))
            seen.append(contexts)
            if grad.context != weight.context:
                mismatch.append(contexts)
            weight[:] = weight

        kv = mx.kv.create("nccl")
        key = 1
        shape = (4,)

        kv.init(key, mx.nd.zeros(shape, mx.gpu(0)))
        kv.push(key, [mx.nd.ones(shape, mx.gpu(0))])
        mx.nd.waitall()

        kv._set_updater(updater)
        kv.push(key, [mx.nd.ones(shape, mx.cpu())])
        mx.nd.waitall()

        assert seen, "updater did not run"
        assert not mismatch, "updater contexts: {}".format(seen)
        """,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


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


def test_issue_20936_wheel_exposes_include_path():
    import mxnet as mx

    paths = mx.libinfo.find_include_path()
    assert paths
    assert any(os.path.exists(os.path.join(path, "mxnet", "base.h")) for path in paths)


def test_issue_20657_find_conf_path_env_override_is_sequence(tmp_path, monkeypatch):
    from mxnet.libinfo import find_conf_path

    conf = tmp_path / "tvmop.conf"
    conf.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MXNET_CONF_PATH", str(conf))
    paths = find_conf_path("tvmop")
    assert isinstance(paths, list)
    assert paths == [str(conf)]


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


def test_issue_20577_symbolblock_export_succeeds_without_cached_op_args(tmp_path):
    import mxnet as mx

    data = mx.sym.Variable("data")
    block = mx.gluon.SymbolBlock([mx.sym.relu(data)], [data])
    prefix = str(tmp_path / "symbolblock")
    block.export(prefix)
    assert (tmp_path / "symbolblock-symbol.json").exists()


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


def test_pr_20491_cpp_symbol_exposes_optimize_for_backend():
    symbol_header = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.h"
    symbol_impl = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.hpp"
    text = symbol_header.read_text(encoding="utf-8") + symbol_impl.read_text(encoding="utf-8")
    assert "OptimizeForBackend" in text


def test_issue_20037_recordio_preserves_large_integer_label():
    from mxnet.recordio import IRHeader, pack, unpack

    header = IRHeader(0, 17672687.0, 1, 0)
    round_tripped = unpack(pack(header, b"x"))[0]
    assert round_tripped.label == header.label


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


def test_issue_19753_topk_indices_are_integer_typed():
    import mxnet as mx

    values, indices = mx.nd.topk(mx.nd.array([3, 1, 2]), ret_typ="both")
    values.wait_to_read()
    indices.wait_to_read()
    assert np.issubdtype(indices.asnumpy().dtype, np.integer)


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


def test_issue_19647_optimize_for_missing_backend_raises():
    import mxnet as mx

    sym = mx.sym.relu(mx.sym.var("data"))
    try:
        sym.optimize_for("definitely_missing_backend")
    except Exception as err:
        assert "backend" in str(err).lower()
    else:
        raise AssertionError("missing backend should raise")


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


def test_issue_19422_numpy_array_iteration_yields_python_scalars():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    items = list(mxnp.arange(3, dtype="int64"))
    assert all(isinstance(item, (int, np.integer)) for item in items)
    assert items == [0, 1, 2]


@issue_xfail(19170, "stepped NumPy slicing needs backend stride metadata; current slice op materializes a copy")
def test_issue_19170_stepped_slice_shares_storage():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    base = mxnp.ones((6,))
    view = base[:5:2]
    base[:] = 0
    view.wait_to_read()
    np.testing.assert_allclose(view.asnumpy(), np.zeros((3,)))


def test_pr_18583_cpp_symbol_exposes_partial_shape_inference():
    symbol_header = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.h"
    symbol_impl = repo_root() / "cpp-package" / "include" / "mxnet-cpp" / "symbol.hpp"
    text = symbol_header.read_text(encoding="utf-8") + symbol_impl.read_text(encoding="utf-8")
    assert "InferShapePartial" in text


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


def test_pr_18792_sort_and_argsort_support_float16():
    import mxnet as mx

    data = mx.nd.array([3, 1, 2], dtype="float16")
    sorted_data = mx.nd.sort(data)
    indices = mx.nd.argsort(data)
    sorted_data.wait_to_read()
    indices.wait_to_read()
    np.testing.assert_allclose(sorted_data.asnumpy(), np.array([1, 2, 3], dtype=np.float16))
    np.testing.assert_allclose(indices.asnumpy(), np.array([1, 2, 0]))


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


def test_issue_18563_max_backward_splits_tied_gradient():
    import mxnet as mx
    from mxnet import autograd

    data = mx.nd.array([2.0, 2.0])
    data.attach_grad()
    with autograd.record():
        out = mx.nd.max(data)
    out.backward()
    np.testing.assert_allclose(data.grad.asnumpy(), np.array([0.5, 0.5]))


def test_issue_18078_prod_backward_multiple_zeros_is_finite():
    import mxnet as mx
    from mxnet import autograd

    data = mx.nd.array([0.0, 0.0])
    data.attach_grad()
    with autograd.record():
        out = mx.nd.prod(data)
    out.backward()
    np.testing.assert_allclose(data.grad.asnumpy(), np.array([0.0, 0.0]))


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


def test_issue_18300_numpy_prod_accepts_shape_tuple():
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    data = mx.np.ones((5,))
    result = mx.np.prod(data.shape)
    assert int(result) == 5


def test_pr_17209_parameter_symbol_var_omits_dtype_attribute():
    from mxnet.gluon import Parameter

    param = Parameter("weight", shape=(2, 2), dtype="float32")
    attrs = next(iter(param.var().attr_dict().values()))
    assert "__dtype__" not in attrs


def test_issue_17936_gammaln_promotes_integer_input():
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    out = npx.gammaln(mxnp.array([1, 2, 3], dtype="int32"))
    out.wait_to_read()
    assert np.issubdtype(out.asnumpy().dtype, np.floating)
    np.testing.assert_allclose(out.asnumpy(), np.array([0.0, 0.0, np.log(2.0)]), rtol=1e-6)


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


def test_issue_16402_legacy_ndarray_dtype_is_numpy_dtype_object():
    import mxnet as mx

    data = mx.nd.array([1], dtype="float32")
    assert isinstance(data.dtype, np.dtype)


def test_issue_16427_recordio_pack_accepts_python3_string_payload():
    from mxnet import recordio

    header = recordio.IRHeader(0, 4, 2574, 0)
    packed = recordio.pack(header, "")
    unpacked_header, payload = recordio.unpack(packed)
    assert unpacked_header.id == 2574
    assert payload == b""


def test_issue_13953_upsampling_accepts_data_keyword():
    import mxnet as mx

    data = mx.symbol.Variable("data")
    sym = mx.symbol.UpSampling(data=data, scale=16, sample_type="nearest")
    assert sym.list_arguments() == ["data"]


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


def test_issue_12286_ndarray_wrapper_raises_python_typeerror_for_missing_inputs():
    import mxnet as mx

    with pytest.raises(TypeError):
        mx.nd.softmax()


def test_issue_8817_sparse_zeros_accepts_integer_shape():
    import mxnet as mx

    data = mx.nd.sparse.zeros("csr", shape=10, ctx=mx.cpu())
    assert data.shape == (10,)
    assert data.stype == "csr"


def test_issue_14695_single_output_ndarray_is_not_tuple_unpackable():
    import mxnet as mx

    result = mx.nd.split(mx.nd.ones((2, 1)), num_outputs=1, squeeze_axis=True)
    first, *rest = result
    first.wait_to_read()
    assert rest == []

SIMILAR_WRAPPER_VALIDATION_REPROS = [
    pytest.param(
        "sym_box_encode_empty_refs",
        """
        s = mx.sym.Variable("samples"); m = mx.sym.Variable("matches")
        a = mx.sym.Variable("anchors"); r = mx.sym.Variable("refs")
        means = mx.sym.Variable("means"); stds = mx.sym.Variable("stds")
        sym = mx.sym.contrib.box_encode(samples=s, matches=m, anchors=a, refs=r,
                                        means=means, stds=stds)
        exe = sym._simple_bind(ctx=mx.cpu(), samples=(1, 1), matches=(1, 1),
                               anchors=(1, 1, 4), refs=(1, 0, 4),
                               means=(4,), stds=(4,))
        exe.arg_dict["samples"][:] = 1; exe.arg_dict["matches"][:] = 0
        exe.arg_dict["anchors"][:] = 1; exe.arg_dict["means"][:] = 0
        exe.arg_dict["stds"][:] = 1
        exe.forward()[0].wait_to_read()
        """,
        ("refs", "empty"),
    ),
    pytest.param(
        "sym_sequence_mask_huge_lengths",
        """
        data = mx.sym.Variable("data"); seq = mx.sym.Variable("seq")
        sym = mx.sym.SequenceMask(data=data, sequence_length=seq,
                                  use_sequence_length=True)
        exe = sym._simple_bind(ctx=mx.cpu(), type_dict={"seq": "int64"},
                               data=(1, 2), seq=(2,))
        exe.arg_dict["data"][:] = 1
        exe.arg_dict["seq"][:] = mx.nd.array([2147483647, 2147483647], dtype="int64")
        exe.forward()[0].wait_to_read()
        """,
        ("sequence", "length"),
    ),
    pytest.param(
        "sym_arange_like_repeat_zero",
        """
        data = mx.sym.Variable("data")
        sym = mx.sym.contrib.arange_like(data=data, repeat=0)
        exe = sym._simple_bind(ctx=mx.cpu(), data=(1,))
        exe.arg_dict["data"][:] = 1
        exe.forward()[0].wait_to_read()
        """,
        ("repeat",),
    ),
    pytest.param(
        "sym_selfatt_qk_zero_heads",
        """
        qkv = mx.sym.Variable("qkv")
        sym = mx.sym.contrib.interleaved_matmul_selfatt_qk(
            queries_keys_values=qkv, heads=0)
        exe = sym._simple_bind(ctx=mx.cpu(), qkv=(2, 1, 3))
        exe.arg_dict["qkv"][:] = 1
        exe.forward()[0].wait_to_read()
        """,
        ("heads",),
    ),
    (
        "sym_image_resize_invalid_interp",
        """
        data = mx.sym.Variable("data")
        sym = mx.sym.image.resize(data, size=(2, 2), interp=10)
        exe = sym._simple_bind(ctx=mx.cpu(), data=(4, 4, 3))
        exe.arg_dict["data"][:] = 1
        exe.forward()[0].wait_to_read()
        """,
        ("interp",),
    ),
    (
        "nd_image_random_resized_crop_invalid_interp",
        """
        x = mx.nd.ones((4, 4, 3), dtype="uint8")
        mx.nd.image.random_resized_crop(x, width=2, height=2,
                                        interp=10).wait_to_read()
        """,
        ("interp",),
    ),
    (
        "nd_image_random_resized_crop_invalid_ratio",
        """
        x = mx.nd.ones((4, 4, 3), dtype="uint8")
        mx.nd.image.random_resized_crop(x, width=2, height=2,
                                        ratio=(0.0, 0.0)).wait_to_read()
        """,
        ("ratio",),
    ),
    pytest.param(
        "npx_image_random_crop_invalid_interp",
        """
        from mxnet import npx
        npx.set_np()
        x = mx.np.ones((1, 1, 3), dtype="uint8")
        npx.image.random_crop(x, (2, 2), interp=10).wait_to_read()
        """,
        ("interp",),
    ),
    pytest.param(
        "sym_image_random_crop_invalid_interp",
        """
        data = mx.sym.Variable("data")
        sym = mx.sym.image.random_crop(data, (2, 2), interp=10)
        exe = sym._simple_bind(ctx=mx.cpu(), data=(1, 1, 3))
        exe.arg_dict["data"][:] = 1
        exe.forward()[0].wait_to_read()
        """,
        ("interp",),
    ),
]


@pytest.mark.parametrize("case, body, required", SIMILAR_WRAPPER_VALIDATION_REPROS)
def test_similar_generated_wrapper_validation(case, body, required):
    proc = run_python(
        """
        import mxnet as mx

        try:
%s
        except ValueError as err:
            msg = str(err).lower()
            for needle in %r:
                assert needle in msg, msg
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should be rejected before backend execution")
        """ % (textwrap.indent(textwrap.dedent(body).strip(), "            "),
               required, case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)


SIMILAR_CROSS_GPU_NUMPY_EXPRESSIONS = [
    "mxnp.equal(left, right)",
    "mxnp.not_equal(left, right)",
    "mxnp.greater(left, right)",
    "mxnp.less(left, right)",
    "mxnp.greater_equal(left, right)",
    "mxnp.less_equal(left, right)",
    "mxnp.true_divide(left, right)",
    "mxnp.bitwise_left_shift(ileft, iright)",
    "mxnp.bitwise_right_shift(ileft, iright)",
    "mxnp.dot(left, right)",
    "mxnp.tensordot(left, right, axes=1)",
    "mxnp.kron(left, right)",
    "mxnp.where(cond, right, right)",
    "mxnp.concatenate([left, right])",
    "mxnp.stack([left, right])",
    "mxnp.vstack([left, right])",
    "mxnp.hstack([left, right])",
    "mxnp.dstack([left, right])",
    "mxnp.column_stack([left, right])",
    "mxnp.average(left, weights=right)",
    "mxnp.einsum('i,i->', left, right)",
    "mxnp.interp(left, xp, fp)",
    "mxnp.polyval(left, right)",
    "mxnp.bincount(ileft, weights=right)",
    "mxnp.ediff1d(left, to_end=right)",
]


@pytest.mark.parametrize("expr", SIMILAR_CROSS_GPU_NUMPY_EXPRESSIONS)
def test_similar_cross_gpu_numpy_public_wrappers_do_not_hang(expr):
    require_gpus(2)
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        npx.set_np()
        left = mxnp.ones((2,), dtype="float32", ctx=mx.gpu(0))
        right = mxnp.ones((2,), dtype="float32", ctx=mx.gpu(1))
        ileft = mxnp.ones((2,), dtype="int32", ctx=mx.gpu(0))
        iright = mxnp.ones((2,), dtype="int32", ctx=mx.gpu(1))
        cond = mxnp.ones((2,), dtype="bool", ctx=mx.gpu(0))
        xp = mxnp.array([0.0, 1.0], ctx=mx.gpu(1))
        fp = mxnp.array([0.0, 1.0], ctx=mx.gpu(1))
        try:
            out = %s
            out.wait_to_read()
        except ValueError as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type") from err
        """ % expr,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


SIMILAR_CROSS_GPU_LEGACY_EXPRESSIONS = [
    "mx.nd.add(left, right)",
    "mx.nd.equal(left, right)",
    "mx.nd.broadcast_add(left, right)",
]


@pytest.mark.parametrize("expr", SIMILAR_CROSS_GPU_LEGACY_EXPRESSIONS)
def test_similar_cross_gpu_legacy_ndarray_binary_wrappers_do_not_hang(expr):
    require_gpus(2)
    proc = run_python(
        """
        import mxnet as mx

        left = mx.nd.ones((2,), ctx=mx.gpu(0))
        right = mx.nd.ones((2,), ctx=mx.gpu(1))
        try:
            out = %s
            out.wait_to_read()
        except ValueError as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type") from err
        """ % expr,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


SIMILAR_NO_AFFINE_BATCHNORM_CASES = [
    ("batchnorm_hybrid", "batchnorm", True),
    ("syncbatchnorm_imperative", "syncbatchnorm", False),
    ("syncbatchnorm_hybrid", "syncbatchnorm", True),
]


@pytest.mark.parametrize("case, layer_kind, hybridize", SIMILAR_NO_AFFINE_BATCHNORM_CASES)
def test_similar_no_affine_batchnorm_backward_keeps_graph(case, layer_kind, hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    net = mx.gluon.nn.HybridSequential()
    if layer_kind == "batchnorm":
        net.add(mx.gluon.nn.BatchNorm(scale=False, center=False))
    else:
        net.add(mx.gluon.nn.SyncBatchNorm(scale=False, center=False, num_devices=1))
    net.initialize(ctx=mx.cpu())
    if hybridize:
        net.hybridize()

    data = mx.np.ones((2, 3, 4, 4), ctx=mx.cpu())
    with mx.autograd.record():
        loss = mx.np.sum(net(data))
    loss.backward(retain_graph=True)


SIMILAR_RNN_SEQUENCE_CASES = ["rnn", "lstm", "gru"]


def _make_seq_rnn(layer_kind):
    import mxnet as mx
    if layer_kind == "rnn":
        return mx.gluon.rnn.RNN(2, activation="tanh", layout="NTC", use_sequence_length=True)
    if layer_kind == "lstm":
        return mx.gluon.rnn.LSTM(2, layout="NTC", use_sequence_length=True)
    return mx.gluon.rnn.GRU(2, layout="NTC", use_sequence_length=True)


@pytest.mark.parametrize("layer_kind", SIMILAR_RNN_SEQUENCE_CASES)
def test_similar_hybrid_rnn_cpu_sequence_length_raises(layer_kind):
    # The native CPU RNN operator ignores sequence_length, so a hybridized CPU
    # layer could only mask the outputs while the returned states would still
    # reflect the full padded sequence. The layer now raises NotImplementedError
    # instead of returning silently-wrong states. The imperative (non-hybridized)
    # CPU path still honors sequence_length correctly.
    import mxnet as mx
    import numpy as onp
    from mxnet import npx

    npx.set_np()
    mx.random.seed(7)

    data = mx.np.ones((2, 4, 3), ctx=mx.cpu())
    full = mx.np.array([4, 4], dtype="int32")
    short = mx.np.array([1, 2], dtype="int32")

    hybrid_net = _make_seq_rnn(layer_kind)
    hybrid_net.initialize(ctx=mx.cpu())
    hybrid_net.hybridize()
    with pytest.raises(NotImplementedError):
        result = hybrid_net(data, sequence_length=full)
        out = result[0] if isinstance(result, (tuple, list)) else result
        out.wait_to_read()

    # The imperative CPU path remains correct: outputs past each valid length are zero.
    eager_net = _make_seq_rnn(layer_kind)
    eager_net.initialize(ctx=mx.cpu())
    result = eager_net(data, sequence_length=short)
    out = result[0] if isinstance(result, (tuple, list)) else result
    out.wait_to_read()
    onp.testing.assert_allclose(out[0, 1:].asnumpy(), onp.zeros((3, 2)), atol=0, rtol=0)
    onp.testing.assert_allclose(out[1, 2:].asnumpy(), onp.zeros((2, 2)), atol=0, rtol=0)


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_groupnorm_large_finite_input_is_finite(hybridize):
    import mxnet as mx
    import numpy as onp
    from mxnet import npx

    npx.set_np()
    layer = mx.gluon.nn.GroupNorm(num_groups=1)
    layer.initialize(ctx=mx.cpu())
    if hybridize:
        layer.hybridize()

    data = mx.np.array([[[[1.4918449e38], [9.0072335e37], [-1.3146734e38], [3.0568930e38]]]], dtype="float32")
    out = layer(data)
    out.wait_to_read()
    assert onp.isfinite(out.asnumpy()).all()


def test_similar_groupnorm_visible_stats_are_computed_without_primary_output():
    import mxnet as mx
    import numpy as onp

    data_np = onp.arange(16, dtype="float32").reshape((2, 4, 2))
    grouped = data_np.reshape((2, 2, 2, 2))
    expected_mean = grouped.mean(axis=(2, 3))
    expected_std = onp.sqrt(grouped.var(axis=(2, 3)) + 1e-5)
    data = mx.sym.var("data")
    gamma = mx.sym.var("gamma")
    beta = mx.sym.var("beta")
    outputs = mx.sym.GroupNorm(
        data=data, gamma=gamma, beta=beta, num_groups=2, output_mean_var=True
    )

    for output_index, expected in [(1, expected_mean), (2, expected_std)]:
        exe = outputs[output_index]._simple_bind(
            ctx=mx.cpu(),
            type_dict={"data": "float32", "gamma": "float32", "beta": "float32"},
            data=data_np.shape,
            gamma=(4,),
            beta=(4,),
        )
        exe.arg_dict["data"][:] = mx.nd.array(data_np)
        exe.arg_dict["gamma"][:] = 1
        exe.arg_dict["beta"][:] = 0
        exe.forward(is_train=False)[0].wait_to_read()
        onp.testing.assert_allclose(exe.outputs[0].asnumpy(), expected, rtol=1e-5, atol=1e-6)


def test_similar_gpu_groupnorm_visible_stats_are_computed_without_primary_output():
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx

        data_np = onp.arange(16, dtype="float32").reshape((2, 4, 2))
        grouped = data_np.reshape((2, 2, 2, 2))
        expected_mean = grouped.mean(axis=(2, 3))
        expected_std = onp.sqrt(grouped.var(axis=(2, 3)) + 1e-5)
        data = mx.sym.var("data")
        gamma = mx.sym.var("gamma")
        beta = mx.sym.var("beta")
        outputs = mx.sym.GroupNorm(
            data=data, gamma=gamma, beta=beta, num_groups=2, output_mean_var=True
        )
        for output_index, expected in [(1, expected_mean), (2, expected_std)]:
            exe = outputs[output_index]._simple_bind(
                ctx=mx.gpu(0),
                type_dict={"data": "float32", "gamma": "float32", "beta": "float32"},
                data=data_np.shape,
                gamma=(4,),
                beta=(4,),
            )
            exe.arg_dict["data"][:] = mx.nd.array(data_np, ctx=mx.gpu(0))
            exe.arg_dict["gamma"][:] = 1
            exe.arg_dict["beta"][:] = 0
            exe.forward(is_train=False)[0].wait_to_read()
            onp.testing.assert_allclose(exe.outputs[0].asnumpy(), expected, rtol=1e-5, atol=1e-6)
        """,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_cosine_embedding_loss_large_parallel_vectors_is_finite_zero(hybridize):
    import mxnet as mx
    import numpy as onp
    from mxnet import npx

    npx.set_np()
    loss = mx.gluon.loss.CosineEmbeddingLoss()
    if hybridize:
        loss.hybridize()

    x = mx.np.array([[1e20, 1e20]], dtype="float32")
    label = mx.np.array([1], dtype="float32")
    out = loss(x, x, label)
    out.wait_to_read()

    assert onp.isfinite(out.asnumpy()).all()
    onp.testing.assert_allclose(out.asnumpy(), onp.array([0.0]), atol=1e-6)


SIMILAR_CSR_ELEMWISE_CASES = [
    "operator_mul",
    "nd.elemwise_mul",
    "sparse.multiply",
    "operator_add",
    "nd.elemwise_add",
    "sparse.add",
    "operator_sub",
    "nd.elemwise_sub",
    "sparse.subtract",
]


@pytest.mark.parametrize("case", SIMILAR_CSR_ELEMWISE_CASES)
def test_similar_csr_elemwise_outputs_are_canonical(case):
    import mxnet as mx

    left = mx.nd.array([[1, 2, 3, 0, 0, 0, 0]], dtype="float32").tostype("csr")
    right = mx.nd.array([[0, 1, 2, 3, 4, 0, 0]], dtype="float32").tostype("csr")
    if case == "operator_mul":
        out = left * right
    elif case == "nd.elemwise_mul":
        out = mx.nd.elemwise_mul(left, right)
    elif case == "sparse.multiply":
        out = mx.nd.sparse.multiply(left, right)
    elif case == "operator_add":
        out = left + right
    elif case == "nd.elemwise_add":
        out = mx.nd.elemwise_add(left, right)
    elif case == "sparse.add":
        out = mx.nd.sparse.add(left, right)
    elif case == "operator_sub":
        out = left - right
    elif case == "nd.elemwise_sub":
        out = mx.nd.elemwise_sub(left, right)
    else:
        out = mx.nd.sparse.subtract(left, right)
    out.wait_to_read()

    assert out.stype == "csr"
    nnz = int(out.indptr.asnumpy()[-1])
    assert out.data.shape[0] == nnz, (case, out.data.shape, out.indptr.asnumpy())
    assert out.indices.shape[0] == nnz, (case, out.indices.shape, out.indptr.asnumpy())
    out.check_format()


@similar_bug_xfail("numpy_stepped_slice_view", "NumPy basic stepped slices need backend stride metadata; current slice op materializes a copy")
def test_similar_numpy_basic_stepped_slice_is_mutable_view():
    import numpy as onp
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    base = mxnp.arange(6, dtype="float32")
    view = base[:5:2]
    view[:] = 0
    onp.testing.assert_allclose(
        base.asnumpy(),
        onp.array([0, 1, 0, 3, 0, 5], dtype="float32"),
    )


SIMILAR_STEPPED_SLICE_CASES = [
    (
        (slice(None, None, 2), slice(None)),
        np.array([[0, 0, 0, 0], [4, 5, 6, 7], [0, 0, 0, 0]], dtype="float32"),
    ),
    (
        (slice(None), slice(None, None, 2)),
        np.array([[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11]], dtype="float32"),
    ),
]


@similar_bug_xfail("numpy_stepped_slice_view", "multi-axis NumPy basic stepped slices need backend stride metadata; current slice op materializes a copy")
@pytest.mark.parametrize("key, expected", SIMILAR_STEPPED_SLICE_CASES)
def test_similar_numpy_multiaxis_basic_stepped_slice_is_mutable_view(key, expected):
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    base = mxnp.arange(12, dtype="float32").reshape((3, 4))
    view = base[key]
    view[:] = 0
    np.testing.assert_allclose(base.asnumpy(), expected)

def test_similar_sparse_retain_does_not_store_absent_zero_rows():
    import mxnet as mx

    data = mx.nd.array(
        [[1, 0, 0], [0, 0, 0], [2, 3, 0], [0, 0, 0]], dtype="float32"
    ).tostype("row_sparse")
    out = mx.nd.sparse.retain(data, mx.nd.array([0, 1, 3], dtype="int64"))
    out.wait_to_read()
    assert out.stype == "row_sparse"
    np.testing.assert_array_equal(
        out.indices.asnumpy(), np.array([0], dtype=out.indices.asnumpy().dtype)
    )
    np.testing.assert_allclose(
        out.asnumpy(),
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype="float32"),
    )
    out.check_format()


def test_similar_sparse_retain_backward_canonicalizes_duplicate_rows():
    import mxnet as mx
    from mxnet import autograd

    data = mx.nd.array([[1, 0], [0, 0], [2, 3]], dtype="float32").tostype("row_sparse")
    data.attach_grad(stype="row_sparse")
    with autograd.record():
        out = mx.nd.sparse.retain(data, mx.nd.array([0, 0, 2], dtype="int64"))
        loss = out.sum()
    loss.backward()
    data.grad.wait_to_read()
    assert data.grad.stype == "row_sparse"
    np.testing.assert_array_equal(
        data.grad.indices.asnumpy(), np.array([0, 2], dtype=data.grad.indices.asnumpy().dtype)
    )
    np.testing.assert_allclose(
        data.grad.asnumpy(), np.array([[1, 1], [0, 0], [1, 1]], dtype="float32")
    )
    data.grad.check_format()


SIMILAR_NUMPY_VIEW_STRIDE_XFAIL = similar_bug_xfail(
    "numpy_view_contract_strides",
    "axis-moving views need backend stride metadata; current operators materialize copies",
)


def _numpy_moveaxis_view(a):
    from mxnet import np as mxnp

    return mxnp.moveaxis(a, 0, -1)


def _numpy_rollaxis_view(a):
    from mxnet import np as mxnp

    return mxnp.rollaxis(a, 2, 0)


SIMILAR_NUMPY_VIEW_CASES = [
    pytest.param(
        "moveaxis",
        _numpy_moveaxis_view,
        np.array([-1, -1], dtype="float32"),
        marks=SIMILAR_NUMPY_VIEW_STRIDE_XFAIL,
    ),
    pytest.param(
        "rollaxis",
        _numpy_rollaxis_view,
        np.array([-1, 1, 2], dtype="float32"),
        marks=SIMILAR_NUMPY_VIEW_STRIDE_XFAIL,
    ),
    (
        "ravel",
        lambda a: a.ravel(),
        np.array([[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype="float32"),
    ),
]


@pytest.mark.parametrize("case, make_view, expected", SIMILAR_NUMPY_VIEW_CASES)
def test_similar_numpy_view_contract_mutates_base(case, make_view, expected):
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    base = mxnp.arange(24, dtype="float32").reshape((2, 3, 4))
    view = make_view(base)
    view[0] = -1
    if case == "moveaxis":
        actual = base[:, 0, 0].asnumpy()
    elif case == "rollaxis":
        actual = base[0, 0, :3].asnumpy()
    else:
        actual = base[0].asnumpy()
    np.testing.assert_allclose(actual, expected)


def test_similar_simple_bind_dynamic_output_allocates_known_argument_arrays():
    import mxnet as mx

    data = mx.sym.var("data")
    sym = mx.sym.contrib.boolean_mask(data, mx.sym.ones_like(data) > 0)
    exe = sym._simple_bind(ctx=mx.cpu(), data=(2,))
    assert exe.arg_dict["data"] is not None
    assert exe.arg_dict["data"].shape == (2,)
    exe.arg_dict["data"][:] = mx.nd.array([1.0, 2.0])
    out = exe.forward()[0]
    out.wait_to_read()
    assert out.shape == (2,)


def test_similar_static_shape_subgraph_does_not_treat_paramless_data_as_param():
    proc = run_python(
        """
        import mxnet as mx

        data = mx.sym.var("data")
        sym = mx.sym.contrib.boolean_mask(data, mx.sym.ones_like(data) > 0)
        opt = sym.optimize_for(
            "static_shape",
            backend_opts={"input_shape": "(2,)", "param_indices": "[]"},
        )
        exe = opt._simple_bind(ctx=mx.cpu(), data=(2,))
        exe.arg_dict["data"][:] = mx.nd.array([1.0, 2.0])
        out = exe.forward()[0]
        out.wait_to_read()
        assert out.shape == (2,)
        """,
        timeout=10,
    )
    assert_subprocess_ok(proc)


def test_similar_transformer_selfatt_valatt_computes_second_input_grad_when_first_is_null():
    import mxnet as mx

    qkv = mx.sym.Variable("qkv")
    att = mx.sym.Variable("att")
    out = mx.sym.contrib.interleaved_matmul_selfatt_valatt(qkv, att, heads=1)
    exe = out._simple_bind(
        ctx=mx.cpu(),
        qkv=(2, 1, 3),
        att=(1, 2, 2),
        grad_req={"qkv": "null", "att": "write"},
        type_dict={"qkv": "float32", "att": "float32"},
    )
    exe.arg_dict["qkv"][:] = mx.nd.array([[[1.0, 2.0, 10.0]], [[3.0, 4.0, 20.0]]])
    exe.arg_dict["att"][:] = mx.nd.array([[[1.0, 0.0], [0.0, 1.0]]])
    exe.forward(is_train=True)
    exe.backward(mx.nd.ones_like(exe.outputs[0]))

    np.testing.assert_allclose(
        exe.grad_dict["att"].asnumpy(),
        np.array([[[10.0, 20.0], [10.0, 20.0]]], dtype="float32"),
    )


def test_similar_transformer_encdec_qk_computes_second_input_grad_when_first_is_null():
    import mxnet as mx

    query = mx.sym.Variable("query")
    kv = mx.sym.Variable("kv")
    out = mx.sym.contrib.interleaved_matmul_encdec_qk(query, kv, heads=1)
    exe = out._simple_bind(
        ctx=mx.cpu(),
        query=(2, 1, 1),
        kv=(2, 1, 2),
        grad_req={"query": "null", "kv": "write"},
        type_dict={"query": "float32", "kv": "float32"},
    )
    exe.arg_dict["query"][:] = mx.nd.array([[[3.0]], [[5.0]]])
    exe.arg_dict["kv"][:] = mx.nd.array([[[7.0, 100.0]], [[11.0, 200.0]]])
    exe.forward(is_train=True)
    exe.backward(mx.nd.ones_like(exe.outputs[0]))

    np.testing.assert_allclose(
        exe.grad_dict["kv"].asnumpy(),
        np.array([[[8.0, 0.0]], [[8.0, 0.0]]], dtype="float32"),
    )


def test_similar_backward_preserves_numpy_scalar_runtime_shape_in_legacy_mode():
    import mxnet as mx

    mx.npx.reset_np()
    with mx.util.np_shape(True), mx.util.np_array(True):
        x = mx.np.ones((2, 3))
        scale = mx.np.array(2.0)
        x.attach_grad()
        scale.attach_grad()
        with mx.autograd.record():
            y = x / scale
        head_grad = mx.np.ones_like(y)

    assert not mx.util.is_np_shape()
    mx.autograd.backward(y, head_grads=head_grad)
    assert scale.grad.shape == ()
    np.testing.assert_allclose(scale.grad.asnumpy(), np.array(-1.5, dtype=np.float32))


SIMILAR_SEQUENCE_LENGTH_RANGE_CASES = [
    ("nd_sequence_last", "SequenceLast", "nd"),
    ("nd_sequence_reverse", "SequenceReverse", "nd"),
    ("sym_sequence_last", "SequenceLast", "sym"),
    ("sym_sequence_reverse", "SequenceReverse", "sym"),
]


@pytest.mark.parametrize("case, op_name, api", SIMILAR_SEQUENCE_LENGTH_RANGE_CASES)
def test_similar_sequence_ops_reject_out_of_range_lengths(case, op_name, api):
    proc = run_python(
        """
        import mxnet as mx

        try:
            if %r == "nd":
                data = mx.nd.ones((2, 3, 1))
                length = mx.nd.array([1, 4, 1], dtype="int32")
                out = getattr(mx.nd, %r)(data, sequence_length=length, use_sequence_length=True)
                out.wait_to_read()
            else:
                data = mx.sym.Variable("data")
                length = mx.sym.Variable("length")
                sym = getattr(mx.sym, %r)(data, sequence_length=length, use_sequence_length=True)
                exe = sym._simple_bind(ctx=mx.cpu(), type_dict={"length": "int32"}, data=(2, 3, 1), length=(3,))
                exe.arg_dict["data"][:] = 1
                exe.arg_dict["length"][:] = mx.nd.array([1, 4, 1], dtype="int32")
                exe.forward()[0].wait_to_read()
        except ValueError as err:
            msg = str(err).lower()
            assert "sequence" in msg and "length" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should reject sequence_length larger than the time axis")
        """ % (api, op_name, op_name, case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)



def test_similar_onednn_layernorm_rank5_uses_native_fallback():
    proc = run_python(
        """
        import mxnet as mx
        import numpy as onp

        data = mx.nd.ones((1024, 1024, 1, 1, 1), dtype="float32")
        gamma = mx.nd.ones((1,), dtype="float32")
        beta = mx.nd.zeros((1,), dtype="float32")
        out = mx.nd.LayerNorm(data, gamma, beta, axis=-1)
        out.wait_to_read()
        assert out.shape == data.shape
        onp.testing.assert_allclose(out.asnumpy(), onp.zeros(data.shape, dtype="float32"), atol=1e-6)
        """,
        timeout=20,
    )
    assert_subprocess_ok(proc)


def test_similar_onednn_layernorm_zero_batch_uses_native_fallback():
    proc = run_python(
        """
        import mxnet as mx

        data = mx.nd.ones((0, 1024), dtype="float32")
        gamma = mx.nd.ones((1024,), dtype="float32")
        beta = mx.nd.zeros((1024,), dtype="float32")
        out = mx.nd.LayerNorm(data, gamma, beta, axis=-1)
        out.wait_to_read()
        assert out.shape == data.shape
        """,
        timeout=10,
    )
    assert_subprocess_ok(proc)


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_layernorm_large_finite_input_is_finite(hybridize):
    import mxnet as mx
    import numpy as onp
    from mxnet import npx

    npx.set_np()
    layer = mx.gluon.nn.LayerNorm(in_channels=4)
    layer.initialize(ctx=mx.cpu())
    if hybridize:
        layer.hybridize()
    data = mx.np.array([[1.4918449e38, 9.0072335e37, -1.3146734e38, 3.0568930e38]], dtype="float32")
    out = layer(data)
    out.wait_to_read()
    assert onp.isfinite(out.asnumpy()).all()


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_layernorm_invalid_negative_axis_is_rejected(hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    layer = mx.gluon.nn.LayerNorm(axis=-3)
    layer.initialize(ctx=mx.cpu())
    if hybridize:
        layer.hybridize()
    data = mx.np.ones((2, 3), ctx=mx.cpu())
    with pytest.raises((ValueError, mx.base.MXNetError), match="axis|Channel"):
        layer(data).wait_to_read()


def test_similar_layernorm_disabled_affine_params_stay_nondifferentiable():
    import mxnet as mx

    layer = mx.gluon.nn.LayerNorm(in_channels=3, center=False, scale=False)
    assert layer.gamma.grad_req == "null"
    assert layer.beta.grad_req == "null"
    layer.gamma.grad_req = "write"
    layer.beta.grad_req = "write"
    assert layer.gamma.grad_req == "null"
    assert layer.beta.grad_req == "null"


@pytest.mark.parametrize("axis", [-1, 0])
def test_similar_layernorm_visible_stats_are_computed_without_primary_output(axis):
    import mxnet as mx
    import numpy as onp

    data_np = onp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    data = mx.sym.var("data")
    gamma = mx.sym.var("gamma")
    beta = mx.sym.var("beta")
    outputs = mx.sym.LayerNorm(data=data, gamma=gamma, beta=beta, axis=axis, output_mean_var=True)
    expected_mean = data_np.mean(axis=axis, keepdims=True)
    expected_std = onp.sqrt(data_np.var(axis=axis, keepdims=True) + 1e-5)
    gamma_shape = (data_np.shape[axis],)

    for output_index, expected in [(1, expected_mean), (2, expected_std)]:
        exe = outputs[output_index]._simple_bind(
            ctx=mx.cpu(),
            type_dict={"data": "float32", "gamma": "float32", "beta": "float32"},
            data=data_np.shape,
            gamma=gamma_shape,
            beta=gamma_shape,
        )
        exe.arg_dict["data"][:] = mx.nd.array(data_np)
        exe.arg_dict["gamma"][:] = 1
        exe.arg_dict["beta"][:] = 0
        exe.forward(is_train=False)[0].wait_to_read()
        onp.testing.assert_allclose(exe.outputs[0].asnumpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("axis", [-1, 0])
def test_similar_gpu_layernorm_visible_stats_are_computed_without_primary_output(axis):
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx

        axis = %r
        data_np = onp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        data = mx.sym.var("data")
        gamma = mx.sym.var("gamma")
        beta = mx.sym.var("beta")
        outputs = mx.sym.LayerNorm(data=data, gamma=gamma, beta=beta, axis=axis, output_mean_var=True)
        expected_mean = data_np.mean(axis=axis, keepdims=True)
        expected_std = onp.sqrt(data_np.var(axis=axis, keepdims=True) + 1e-5)
        gamma_shape = (data_np.shape[axis],)
        for output_index, expected in [(1, expected_mean), (2, expected_std)]:
            exe = outputs[output_index]._simple_bind(
                ctx=mx.gpu(0),
                type_dict={"data": "float32", "gamma": "float32", "beta": "float32"},
                data=data_np.shape,
                gamma=gamma_shape,
                beta=gamma_shape,
            )
            exe.arg_dict["data"][:] = mx.nd.array(data_np, ctx=mx.gpu(0))
            exe.arg_dict["gamma"][:] = 1
            exe.arg_dict["beta"][:] = 0
            exe.forward(is_train=False)[0].wait_to_read()
            onp.testing.assert_allclose(exe.outputs[0].asnumpy(), expected, rtol=1e-5, atol=1e-6)
        """ % axis,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)



def test_similar_syncbatchnorm_reinitializes_shared_buffers_for_new_channel_shape():
    require_gpus(1)
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import autograd

        ctx = mx.gpu(0)
        def run(channel_count):
            data = mx.nd.ones((2, channel_count, 2, 2), ctx=ctx)
            gamma = mx.nd.ones((channel_count,), ctx=ctx)
            beta = mx.nd.zeros((channel_count,), ctx=ctx)
            moving_mean = mx.nd.zeros((channel_count,), ctx=ctx)
            moving_var = mx.nd.ones((channel_count,), ctx=ctx)
            data.attach_grad()
            gamma.attach_grad()
            beta.attach_grad()
            with autograd.record():
                out = mx.nd.contrib.SyncBatchNorm(
                    data,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    ndev=1,
                    key="shape_reuse_regression",
                    fix_gamma=False,
                )
                loss = out.sum()
            loss.backward()
            mx.nd.waitall()

        run(2)
        run(3)
        """,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


@pytest.mark.parametrize("kind", ["batchnorm", "syncbatchnorm"])
@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_batchnorm_training_large_finite_is_finite(kind, hybridize):
    import mxnet as mx
    import numpy as onp
    from mxnet import autograd
    from mxnet import npx

    npx.set_np()
    if kind == "batchnorm":
        layer = mx.gluon.nn.BatchNorm(in_channels=1)
    else:
        layer = mx.gluon.nn.SyncBatchNorm(in_channels=1, num_devices=1)
    layer.initialize(ctx=mx.cpu())
    if hybridize:
        layer.hybridize()
    data = mx.np.array([[[[1.4918449e38], [9.0072335e37], [-1.3146734e38], [3.0568930e38]]]], dtype="float32")
    data.attach_grad()
    with autograd.record(train_mode=True):
        out = layer(data)
        loss = out.sum()
    loss.backward()
    out.wait_to_read()
    assert onp.isfinite(out.asnumpy()).all()
    assert onp.isfinite(data.grad.asnumpy()).all()


@pytest.mark.parametrize("kind", ["batchnorm", "syncbatchnorm"])
def test_similar_batchnorm_large_finite_updates_unscaled_running_stats(kind):
    import mxnet as mx
    import numpy as onp
    from mxnet import autograd
    from mxnet import npx

    npx.set_np()
    if kind == "batchnorm":
        layer = mx.gluon.nn.BatchNorm(in_channels=1, momentum=0.0)
    else:
        layer = mx.gluon.nn.SyncBatchNorm(in_channels=1, momentum=0.0, num_devices=1)
    layer.initialize(ctx=mx.cpu())
    data = mx.np.array([[[[1.0e19], [2.0e19]]]], dtype="float32")
    with autograd.record(train_mode=True):
        out = layer(data)
        loss = out.sum()
    loss.backward()
    out.wait_to_read()
    onp.testing.assert_allclose(layer.running_mean.data().asnumpy(), onp.array([1.5e19], dtype="float32"))
    onp.testing.assert_allclose(layer.running_var.data().asnumpy(), onp.array([2.5e37], dtype="float32"), rtol=1e-5)


def test_similar_batchnorm_fix_gamma_does_not_mutate_gamma_input():
    import mxnet as mx
    import numpy as onp

    data = mx.nd.array([[[1.0], [2.0], [3.0]]])
    gamma = mx.nd.array([5.0, 6.0, 7.0])
    beta = mx.nd.zeros((3,))
    moving_mean = mx.nd.array([0.1, 0.2, 0.3])
    moving_var = mx.nd.array([1.0, 1.5, 2.0])
    outputs = mx.nd.BatchNorm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        fix_gamma=True,
        use_global_stats=True,
        output_mean_var=True,
        cudnn_off=True,
    )
    for out in outputs:
        out.wait_to_read()
    onp.testing.assert_allclose(gamma.asnumpy(), onp.array([5.0, 6.0, 7.0], dtype="float32"))


@pytest.mark.parametrize("cudnn_off", [False, True])
def test_similar_gpu_batchnorm_inference_output_mean_var_are_populated(cudnn_off):
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx

        ctx = mx.gpu(0)
        data = mx.nd.array([[[1.0], [2.0], [3.0]]], ctx=ctx)
        gamma = mx.nd.ones((3,), ctx=ctx)
        beta = mx.nd.zeros((3,), ctx=ctx)
        moving_mean = mx.nd.array([0.1, 0.2, 0.3], ctx=ctx)
        moving_var = mx.nd.array([1.0, 1.5, 2.0], ctx=ctx)
        out, mean, invstd = mx.nd.BatchNorm(
            data,
            gamma,
            beta,
            moving_mean,
            moving_var,
            fix_gamma=False,
            use_global_stats=True,
            output_mean_var=True,
            cudnn_off=%r,
        )
        mx.nd.waitall()
        onp.testing.assert_allclose(mean.asnumpy(), moving_mean.asnumpy(), rtol=1e-6, atol=1e-6)
        onp.testing.assert_allclose(
            invstd.asnumpy(),
            1.0 / onp.sqrt(moving_var.asnumpy() + 1e-3),
            rtol=1e-6,
            atol=1e-6,
        )
        """ % cudnn_off,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


def test_similar_gpu_batchnorm_training_large_finite_is_finite():
    # The native CUDA BatchNorm training kernel (forced via cudnn_off=True) must
    # accumulate mean/variance in double so large-finite float32 inputs do not
    # overflow into a NaN output (sum-of-data overflowing float32 -> inf mean ->
    # inf*0 = NaN). Mirrors the CPU test_similar_batchnorm_training_large_finite.
    require_gpus(1)
    proc = run_python(
        """
        import numpy as onp
        import mxnet as mx
        from mxnet import autograd

        ctx = mx.gpu(0)
        data = mx.nd.array(
            [[[1.4918449e38, 9.0072335e37, -1.3146734e38, 3.0568930e38]]],
            ctx=ctx, dtype="float32")
        gamma = mx.nd.ones((1,), ctx=ctx)
        beta = mx.nd.zeros((1,), ctx=ctx)
        moving_mean = mx.nd.zeros((1,), ctx=ctx)
        moving_var = mx.nd.ones((1,), ctx=ctx)
        data.attach_grad()
        with autograd.record(train_mode=True):
            out = mx.nd.BatchNorm(
                data, gamma, beta, moving_mean, moving_var,
                fix_gamma=False, use_global_stats=False, cudnn_off=True)
            loss = out.sum()
        loss.backward()
        mx.nd.waitall()
        assert onp.isfinite(out.asnumpy()).all(), out.asnumpy()
        assert onp.isfinite(data.grad.asnumpy()).all(), data.grad.asnumpy()
        """,
        timeout=20,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_subprocess_ok(proc)


def test_similar_syncbatchnorm_float16_is_rejected_cleanly():
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    data = mx.np.ones((2, 3, 2, 2), dtype="float16")
    gamma = mx.np.ones((3,), dtype="float32")
    beta = mx.np.zeros((3,), dtype="float32")
    moving_mean = mx.np.zeros((3,), dtype="float32")
    moving_var = mx.np.ones((3,), dtype="float32")
    with pytest.raises(mx.base.MXNetError, match="SyncBatchNorm|float16|FP16|unsupported"):
        npx.sync_batch_norm(
            data,
            gamma,
            beta,
            moving_mean,
            moving_var,
            ndev=1,
            key="fp16_reject",
            fix_gamma=False,
        ).wait_to_read()



@pytest.mark.parametrize("dtype", ["float16", "float64"])
def test_similar_instancenorm_rejects_unsupported_dtype_cleanly(dtype):
    import mxnet as mx

    data = mx.nd.ones((1, 2, 2), dtype=dtype)
    gamma = mx.nd.ones((2,), dtype=dtype)
    beta = mx.nd.zeros((2,), dtype=dtype)
    with pytest.raises(mx.base.MXNetError, match="InstanceNorm|float32"):
        mx.nd.InstanceNorm(data, gamma, beta).wait_to_read()


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_instancenorm_nondefault_axis_deferred_channels(hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    layer = mx.gluon.nn.InstanceNorm(axis=3)
    layer.initialize(ctx=mx.cpu())
    if hybridize:
        layer.hybridize()
    data = mx.np.arange(2 * 4 * 5 * 3, dtype="float32").reshape((2, 4, 5, 3))
    out = layer(data)
    out.wait_to_read()
    assert out.shape == data.shape


SIMILAR_GLUON_LOSS_NUMERIC_CASES = ["huber", "triplet", "poisson_zero", "sdml_one_hot"]


@pytest.mark.parametrize("case", SIMILAR_GLUON_LOSS_NUMERIC_CASES)
@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_gluon_loss_numeric_edges(case, hybridize):
    import mxnet as mx
    import numpy as onp
    from mxnet import npx

    npx.set_np()
    if case == "huber":
        loss = mx.gluon.loss.HuberLoss(rho=1.0)
        args = (mx.np.array([[1e20]], dtype="float32"), mx.np.array([[0.0]], dtype="float32"))
        expected = onp.array([1e20], dtype="float32")
    elif case == "triplet":
        loss = mx.gluon.loss.TripletLoss(margin=1.0)
        pred = mx.np.zeros((1, 2), dtype="float32")
        pos = mx.np.array([[1e20, -1e20]], dtype="float32")
        args = (pred, pos, pos.copy())
        expected = onp.array([1.0], dtype="float32")
    elif case == "poisson_zero":
        loss = mx.gluon.loss.PoissonNLLLoss(from_logits=False, compute_full=True)
        args = (mx.np.ones((1, 1), dtype="float32"), mx.np.zeros((1, 1), dtype="float32"))
        expected = onp.array([1.0], dtype="float32")
    else:
        loss = mx.gluon.loss.SDMLLoss(smoothing_parameter=0.0)
        data = mx.np.array([[1e20, 0.0], [0.0, 1e20]], dtype="float32")
        args = (data, data)
        expected = onp.zeros((2,), dtype="float32")
    if hybridize:
        loss.hybridize()
    out = loss(*args)
    out.wait_to_read()
    assert onp.isfinite(out.asnumpy()).all()
    onp.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-6, atol=1e-6)


SIMILAR_DYNAMIC_UNROLL_INT_VALID_LENGTH_CELLS = [
    "RNNCell",
    "LSTMCell",
    "GRUCell",
]


@pytest.mark.parametrize("cell_cls_name", SIMILAR_DYNAMIC_UNROLL_INT_VALID_LENGTH_CELLS)
def test_similar_dynamic_unroll_accepts_int32_valid_length(cell_cls_name):
    import mxnet as mx
    from mxnet import npx
    from mxnet.gluon.rnn.rnn_cell import dynamic_unroll

    npx.set_np()
    cell_cls = getattr(mx.gluon.rnn, cell_cls_name)
    cell = cell_cls(2)
    data = mx.np.ones((3, 2, 4), dtype="float32")
    cell.infer_shape(0, data[0], False)
    cell.initialize(ctx=mx.cpu())
    states = cell.begin_state(batch_size=2, device=mx.cpu())
    out, _ = dynamic_unroll(
        cell,
        data,
        states,
        layout="TNC",
        valid_length=mx.np.array([1, 2], dtype="int32"),
    )
    out.wait_to_read()
    assert out.shape == (3, 2, 2)

SIMILAR_SEQUENCE_BOUNDARY_CASES = [
    (
        "nd_last_zero",
        'mx.nd.SequenceLast(mx.nd.arange(6).reshape((2, 3, 1)), sequence_length=mx.nd.array([1, 0, 2], dtype="int32"), use_sequence_length=True).wait_to_read()',
    ),
    (
        "nd_reverse_zero",
        'mx.nd.SequenceReverse(mx.nd.arange(6).reshape((2, 3, 1)), sequence_length=mx.nd.array([1, 0, 2], dtype="int32"), use_sequence_length=True).wait_to_read()',
    ),
    (
        "nd_last_axis1_too_large",
        'mx.nd.SequenceLast(mx.nd.arange(6).reshape((2, 3, 1)), sequence_length=mx.nd.array([1, 4], dtype="int32"), use_sequence_length=True, axis=1).wait_to_read()',
    ),
    (
        "nd_last_length_shape",
        'mx.nd.SequenceLast(mx.nd.arange(6).reshape((2, 3, 1)), sequence_length=mx.nd.array([1, 2], dtype="int32"), use_sequence_length=True).wait_to_read()',
    ),
    (
        "npx_last_zero",
        'npx.set_np(); npx.sequence_last(mxnp.arange(6).reshape((2, 3, 1)), sequence_length=mxnp.array([1, 0, 2], dtype="int32"), use_sequence_length=True).wait_to_read()',
    ),
]


@pytest.mark.parametrize("case, body", SIMILAR_SEQUENCE_BOUNDARY_CASES)
def test_similar_sequence_ops_reject_invalid_boundary_lengths(case, body):
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        try:
%s
        except ValueError as err:
            msg = str(err).lower()
            assert "length" in msg or "shape" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should reject invalid sequence_length")
        """ % (textwrap.indent(body, "            "), case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)


SIMILAR_IMAGE_WRAPPER_SIZE_CASES = [
    (
        "nd_resize_zero",
        'mx.nd.image.resize(mx.nd.ones((4, 4, 3), dtype="uint8"), size=(0, 2)).wait_to_read()',
    ),
    (
        "npx_resize_zero",
        'npx.set_np(); npx.image.resize(mxnp.ones((4, 4, 3), dtype="uint8"), (0, 2)).wait_to_read()',
    ),
    (
        "sym_resize_bad_dim",
        'data = mx.sym.Variable("data"); sym = mx.sym.image.resize(data, size=(2, 2)); sym._simple_bind(ctx=mx.cpu(), data=(4, 4))',
    ),
    (
        "npx_crop_negative_x",
        'npx.set_np(); npx.image.crop(mxnp.ones((4, 4, 3), dtype="uint8"), x=-1, y=0, width=2, height=2).wait_to_read()',
    ),
    (
        "sym_random_crop_bad_xrange",
        'data = mx.sym.Variable("data"); sym = mx.sym.image.random_crop(data, xrange=(0.8, 0.2), yrange=(0, 1), width=2, height=2); sym._simple_bind(ctx=mx.cpu(), data=(4, 4, 3))',
    ),
]


@pytest.mark.parametrize("case, body", SIMILAR_IMAGE_WRAPPER_SIZE_CASES)
def test_similar_image_wrappers_reject_invalid_sizes(case, body):
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        try:
%s
        except ValueError as err:
            msg = str(err).lower()
            assert any(s in msg for s in ("size", "width", "height", "range", "dimension", "offset"))
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should reject invalid image arguments")
        """ % (textwrap.indent(body, "            "), case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)


SIMILAR_RANDOM_RESIZED_CROP_RANGE_CASES = [
    (
        "nd_zero_area",
        'mx.nd.image.random_resized_crop(mx.nd.ones((4, 4, 3), dtype="uint8"), width=2, height=2, area=(0.0, 0.0)).wait_to_read()',
    ),
    (
        "npx_negative_ratio",
        'npx.set_np(); npx.image.random_resized_crop(mxnp.ones((4, 4, 3), dtype="uint8"), width=2, height=2, ratio=(-1.0, 1.0)).wait_to_read()',
    ),
]


@pytest.mark.parametrize("case, body", SIMILAR_RANDOM_RESIZED_CROP_RANGE_CASES)
def test_similar_random_resized_crop_rejects_bad_area_ratio(case, body):
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        try:
%s
        except ValueError as err:
            msg = str(err).lower()
            assert "area" in msg or "ratio" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should reject invalid area/ratio")
        """ % (textwrap.indent(body, "            "), case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)


SIMILAR_MX_IMAGE_HIGH_LEVEL_INVALID_CASES = [
    (
        "imresize_bad_interp",
        'npx.set_np(); mx.image.imresize(mxnp.ones((4, 4, 3), dtype="uint8"), 2, 2, interp=99).wait_to_read()',
    ),
    (
        "resize_short_zero",
        'npx.set_np(); mx.image.resize_short(mxnp.ones((4, 4, 3), dtype="uint8"), 0).wait_to_read()',
    ),
    (
        "random_crop_zero_width",
        'npx.set_np(); mx.image.random_crop(mxnp.ones((4, 4, 3), dtype="uint8"), (0, 2))[0].wait_to_read()',
    ),
]


@pytest.mark.parametrize("case, body", SIMILAR_MX_IMAGE_HIGH_LEVEL_INVALID_CASES)
def test_similar_mx_image_high_level_rejects_invalid_resize_crop(case, body):
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx

        try:
%s
        except ValueError:
            pass
        except Exception as err:
            raise AssertionError("unexpected exception type for %s") from err
        else:
            raise AssertionError("%s should reject invalid mx.image argument")
        """ % (textwrap.indent(body, "            "), case, case),
        timeout=10,
    )
    assert_subprocess_ok(proc)

EXTRA_CROSS_GPU_NUMPY = [
    "mxnp.take(mat0, idx1, axis=1)",
    "mat0.take(idx1, axis=1)",
    "ndnp.take(mat0, idx1, axis=1)",
    "mxnp.percentile(mat0, q_percent)",
    "mxnp.quantile(mat0, q_quant)",
    "ndnp.percentile(mat0, q_percent)",
    "ndnp.quantile(mat0, q_quant)",
    "mxnp.cross(mat0, mat1)",
    "ndnp.cross(mat0, mat1)",
    "mat0.dot(mat1.T)",
    "mxnp.vdot(vec0, vec1)",
    "mxnp.outer(vec0, vec1)",
    "ndnp.vdot(vec0, vec1)",
    "ndnp.outer(vec0, vec1)",
    "mxnp.append(mat0, mat1, axis=0)",
    "mxnp.row_stack([mat0, mat1])",
    "mxnp.ediff1d(vec0, to_begin=vec1)",
    "left == right",
    "left > right",
]


@pytest.mark.parametrize("expr", EXTRA_CROSS_GPU_NUMPY)
def test_similar_cross_gpu_numpy_extra_wrappers_reject(expr):
    require_gpus(2)
    proc = run_python(
        """
        import mxnet as mx
        from mxnet import np as mxnp
        from mxnet import npx
        from mxnet.ndarray import numpy as ndnp

        npx.set_np()
        left = mxnp.ones((2,), ctx=mx.gpu(0))
        right = mxnp.ones((2,), ctx=mx.gpu(1))
        mat0 = mxnp.ones((2, 3), ctx=mx.gpu(0))
        mat1 = mxnp.ones((2, 3), ctx=mx.gpu(1))
        vec0 = mxnp.arange(3, ctx=mx.gpu(0))
        vec1 = mxnp.arange(3, ctx=mx.gpu(1))
        idx1 = mxnp.array([0, 2], dtype="int64", ctx=mx.gpu(1))
        q_percent = mxnp.array(50.0, ctx=mx.gpu(1))
        q_quant = mxnp.array(0.5, ctx=mx.gpu(1))
        try:
            out = %s
            if hasattr(out, "wait_to_read"):
                out.wait_to_read()
        except ValueError as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type") from err
        else:
            raise AssertionError("cross-device inputs should be rejected")
        """ % expr,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


EXTRA_CROSS_GPU_LEGACY = [
    "left - right",
    "left * right",
    "left == right",
    "mx.nd.subtract(left, right)",
    "mx.nd.multiply(left, right)",
    "mx.nd.divide(left, right)",
    "mx.nd.modulo(left, right)",
    "mx.nd.power(left, right)",
    "mx.nd.maximum(left, right)",
    "mx.nd.minimum(left, right)",
    "mx.nd.not_equal(left, right)",
    "mx.nd.greater(left, right)",
    "mx.nd.lesser(left, right)",
    "mx.nd.greater_equal(left, right)",
    "mx.nd.lesser_equal(left, right)",
    "mx.nd.logical_and(left, right)",
    "mx.nd.logical_or(left, right)",
    "mx.nd.logical_xor(left, right)",
    "mx.nd.dot(left, right)",
    "mx.nd.concat(mat0, mat1, dim=0)",
    "mx.nd.stack(mat0, mat1, axis=0)",
    "mx.nd.take(mat0, idx1, axis=1)",
    "mx.nd.pick(mat0, idx1, axis=1)",
    "mx.nd.where(cond0, right, right)",
]


@pytest.mark.parametrize("expr", EXTRA_CROSS_GPU_LEGACY)
def test_similar_cross_gpu_legacy_extra_wrappers_reject(expr):
    require_gpus(2)
    proc = run_python(
        """
        import mxnet as mx

        left = mx.nd.ones((2,), ctx=mx.gpu(0))
        right = mx.nd.ones((2,), ctx=mx.gpu(1))
        mat0 = mx.nd.ones((2, 3), ctx=mx.gpu(0))
        mat1 = mx.nd.ones((2, 3), ctx=mx.gpu(1))
        idx1 = mx.nd.array([0, 1], ctx=mx.gpu(1), dtype="int64")
        cond0 = mx.nd.ones((2,), ctx=mx.gpu(0))
        try:
            out = %s
            if hasattr(out, "wait_to_read"):
                out.wait_to_read()
        except ValueError as err:
            msg = str(err).lower()
            assert "context" in msg or "device" in msg or "gpu" in msg
        except Exception as err:
            raise AssertionError("unexpected exception type") from err
        else:
            raise AssertionError("cross-device inputs should be rejected")
        """ % expr,
        timeout=10,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert_subprocess_ok(proc)


SIMILAR_NUMPY_EXTRA_VIEW_STRIDE_XFAIL = similar_bug_xfail(
    "numpy_extra_view_contract_strides",
    "flip/fliplr/rot90 need negative-stride or axis-stride backend view metadata",
)

SIMILAR_NUMPY_EXTRA_VIEW_CASES = [
    pytest.param("flip", marks=SIMILAR_NUMPY_EXTRA_VIEW_STRIDE_XFAIL),
    pytest.param("flipud", marks=SIMILAR_NUMPY_EXTRA_VIEW_STRIDE_XFAIL),
    pytest.param("fliplr", marks=SIMILAR_NUMPY_EXTRA_VIEW_STRIDE_XFAIL),
    pytest.param("rot90", marks=SIMILAR_NUMPY_EXTRA_VIEW_STRIDE_XFAIL),
    "squeeze",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
]


@pytest.mark.parametrize("case", SIMILAR_NUMPY_EXTRA_VIEW_CASES)
def test_similar_numpy_extra_view_contract_mutates_base(case):
    from mxnet import np as mxnp
    from mxnet import npx

    npx.set_np()
    if case in ("flip", "flipud", "fliplr"):
        base = mxnp.arange(6, dtype="float32").reshape((2, 3))
        if case == "flip":
            view = mxnp.flip(base, 0)
            view[0] = 0
            actual = base[1]
        elif case == "flipud":
            view = mxnp.flipud(base)
            view[0] = 0
            actual = base[1]
        else:
            view = mxnp.fliplr(base)
            view[:, 0] = 0
            actual = base[:, 2]
    elif case == "rot90":
        base = mxnp.arange(4, dtype="float32").reshape((2, 2))
        view = mxnp.rot90(base)
        view[0] = 0
        actual = base[:, 1]
    elif case == "squeeze":
        base = mxnp.arange(3, dtype="float32").reshape((1, 3, 1))
        view = mxnp.squeeze(base)
        view[:] = 0
        actual = base.reshape((-1,))
    else:
        base = mxnp.arange(3, dtype="float32")
        view = getattr(mxnp, case)(base)
        view[:] = 0
        actual = base
    np.testing.assert_allclose(actual.asnumpy(), np.zeros(actual.shape, dtype="float32"))

def test_similar_row_sparse_elemwise_mul_prunes_zero_rows():
    import mxnet as mx

    def rsp(indices, data, shape):
        return mx.nd.sparse.row_sparse_array(
            (mx.nd.array(data, dtype="float32"), mx.nd.array(indices, dtype="int64")),
            shape=shape,
        )

    left = rsp([0, 2], [[1, 1], [2, 3]], (4, 2))
    right = rsp([1, 2], [[4, 5], [6, 7]], (4, 2))
    out = left * right
    out.wait_to_read()
    np.testing.assert_array_equal(
        out.indices.asnumpy(), np.array([2], dtype=out.indices.asnumpy().dtype)
    )
    np.testing.assert_allclose(out.data.asnumpy(), np.array([[12, 21]], dtype="float32"))
    out.check_format()


SIMILAR_CSR_DENSE_MUL_PRUNE_CASES = [
    (
        "same_shape",
        np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype="float32"),
        np.array([0, 1, 1]),
        np.array([1]),
        np.array([2], dtype="float32"),
    ),
    (
        "broadcast_row",
        np.array([[0, 1, 0, 0]], dtype="float32"),
        np.array([0, 1, 2]),
        np.array([1, 1]),
        np.array([2, 4], dtype="float32"),
    ),
]


@pytest.mark.parametrize("case, mask_np, expected_indptr, expected_indices, expected_data", SIMILAR_CSR_DENSE_MUL_PRUNE_CASES)
def test_similar_csr_dense_mul_prunes_zero_entries(case, mask_np, expected_indptr, expected_indices, expected_data):
    import mxnet as mx

    data = mx.nd.array([[1, 2, 3, 0], [0, 4, 0, 5]], dtype="float32").tostype("csr")
    mask = mx.nd.array(mask_np, dtype="float32")
    out = data * mask
    out.wait_to_read()
    np.testing.assert_array_equal(out.indptr.asnumpy(), expected_indptr.astype(out.indptr.asnumpy().dtype))
    np.testing.assert_array_equal(out.indices.asnumpy(), expected_indices.astype(out.indices.asnumpy().dtype))
    np.testing.assert_allclose(out.data.asnumpy(), expected_data)
    out.check_format()


def test_similar_sparse_relu_prunes_zero_metadata():
    import mxnet as mx

    csr = mx.nd.array([[-1, 0, -2], [0, -3, 0]], dtype="float32").tostype("csr")
    out = mx.nd.relu(csr)
    out.wait_to_read()
    np.testing.assert_array_equal(out.indptr.asnumpy(), np.array([0, 0, 0], dtype=out.indptr.asnumpy().dtype))
    assert out.indices.shape[0] == 0
    assert out.data.shape[0] == 0
    out.check_format()


def test_similar_row_sparse_scalar_zero_prunes_zero_metadata():
    import mxnet as mx

    rsp = mx.nd.array([[1, 2], [0, 0], [3, 4]], dtype="float32").tostype("row_sparse")
    out = rsp * 0
    out.wait_to_read()
    assert out.indices.shape[0] == 0
    assert out.data.shape[0] == 0
    out.check_format()


def test_similar_sparse_retain_canonicalizes_unsorted_row_ids():
    import mxnet as mx

    data = mx.nd.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype="float32").tostype("row_sparse")
    out = mx.nd.sparse.retain(data, mx.nd.array([2, 0], dtype="int64"))
    out.wait_to_read()
    np.testing.assert_array_equal(
        out.indices.asnumpy(), np.array([0, 2], dtype=out.indices.asnumpy().dtype)
    )
    np.testing.assert_allclose(
        out.asnumpy(), np.array([[1, 0], [0, 0], [3, 0], [0, 0]], dtype="float32")
    )
    out.check_format()


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_softmax_ce_dense_from_logits_zero_label_masks_ninf(hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=True)
    if hybridize:
        loss.hybridize()
    pred = mx.np.array([[0.0, float("-inf")]], dtype="float32")
    label = mx.np.array([[1.0, 0.0]], dtype="float32")
    out = loss(pred, label)
    out.wait_to_read()
    actual = out.asnumpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([0.0], dtype="float32"), rtol=1e-6, atol=1e-6)


SIMILAR_BINARY_LOGIT_LIMIT_CASES = ["sigmoid_bce", "logistic_binary"]


@pytest.mark.parametrize("case", SIMILAR_BINARY_LOGIT_LIMIT_CASES)
@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_binary_logit_losses_take_finite_limits(case, hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    if case == "sigmoid_bce":
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    else:
        loss = mx.gluon.loss.LogisticLoss(label_format="binary")
    if hybridize:
        loss.hybridize()
    pred = mx.np.array([[float("inf"), float("-inf")]], dtype="float32")
    label = mx.np.array([[1.0, 0.0]], dtype="float32")
    out = loss(pred, label)
    out.wait_to_read()
    actual = out.asnumpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([0.0], dtype="float32"), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_poisson_nll_log_rate_zero_count_is_zero(hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    loss = mx.gluon.loss.PoissonNLLLoss(from_logits=True)
    if hybridize:
        loss.hybridize()
    pred = mx.np.array([[float("-inf")]], dtype="float32")
    target = mx.np.array([[0.0]], dtype="float32")
    out = loss(pred, target)
    out.wait_to_read()
    actual = out.asnumpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([0.0], dtype="float32"), rtol=1e-6, atol=1e-6)


SIMILAR_ZERO_SAMPLE_WEIGHT_OVERFLOW_CASES = ["l2", "squared_hinge"]


@pytest.mark.parametrize("case", SIMILAR_ZERO_SAMPLE_WEIGHT_OVERFLOW_CASES)
@pytest.mark.parametrize("hybridize", [False, True])
def test_similar_zero_sample_weight_masks_large_finite_loss_overflow(case, hybridize):
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    if case == "l2":
        loss = mx.gluon.loss.L2Loss()
        args = (
            mx.np.array([[1e20]], dtype="float32"),
            mx.np.array([[0.0]], dtype="float32"),
        )
    else:
        loss = mx.gluon.loss.SquaredHingeLoss()
        args = (
            mx.np.array([[-1e20]], dtype="float32"),
            mx.np.array([[1.0]], dtype="float32"),
        )
    if hybridize:
        loss.hybridize()
    sample_weight = mx.np.array([[0.0]], dtype="float32")
    out = loss(*args, sample_weight)
    out.wait_to_read()
    actual = out.asnumpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([0.0], dtype="float32"), rtol=1e-6, atol=1e-6)


def test_similar_npx_lp_pooling_large_finite_l2_window_is_finite():
    import mxnet as mx
    from mxnet import npx

    npx.set_np()
    data = mx.np.array([1e20, 1e20], dtype="float32").reshape((1, 1, 2))
    out = npx.pooling(data, kernel=(2,), pool_type="lp", p_value=2, layout="NCW")
    out.wait_to_read()
    actual = out.asnumpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual, np.array([[[np.sqrt(2.0) * 1e20]]], dtype="float32"), rtol=1e-6
    )

