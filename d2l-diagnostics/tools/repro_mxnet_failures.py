#!/usr/bin/env python3
"""Small MXNet probes for reproducing notebook runtime failures.

The notebook logs often contain the failing MXNet kernel name, but not enough
context to reproduce it without running a full notebook.  This script keeps
those reproductions intentionally small and prints progress before each case so
dead kernels can still be localized by the last visible line.
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import json
import os
import sys
import sysconfig
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def ensure_nvidia_ld_path():
    """Re-exec with pip-installed NVIDIA library directories on LD_LIBRARY_PATH."""
    if os.environ.get("D2L_MXNET_REEXEC") == "1":
        return
    site_roots = {
        Path(sysconfig.get_paths().get("purelib", "")),
        *{
            path
            for path in (ROOT / ".venv-mxnet/lib").glob("python*/site-packages")
            if path.is_dir()
        },
    }
    lib_dirs: list[str] = []
    for site_root in site_roots:
        if not site_root:
            continue
        lib_dirs.extend(glob.glob(str(site_root / "nvidia/*/lib")))
    lib_dirs = sorted({str(Path(path).resolve()) for path in lib_dirs if Path(path).is_dir()})
    if not lib_dirs:
        return
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    missing = [item for item in lib_dirs if item not in current]
    if not missing:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(missing + current)
    env["D2L_MXNET_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def import_mxnet():
    import mxnet as mx
    from mxnet import autograd, gluon, np, npx
    from mxnet.gluon import nn

    npx.set_np()
    return mx, np, npx, autograd, gluon, nn


def gpu_context(mx):
    count = mx.context.num_gpus()
    if count < 1:
        raise RuntimeError("MXNet reports no visible GPUs")
    return mx.gpu(0)


def run_case(name, fn):
    print(f"== {name} ==", flush=True)
    try:
        value = fn()
    except Exception:
        print(f"FAIL: {name}", flush=True)
        traceback.print_exc()
        return False
    print(f"OK: {name}: {value}", flush=True)
    return True


def sync(value):
    """Force lazy MXNet execution and host transfer."""
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return value


def case_info():
    mx, _, _, _, _, _ = import_mxnet()
    print(f"mxnet={mx.__version__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"num_gpus={mx.context.num_gpus()}", flush=True)


def case_gpu_sum():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((128,), ctx=ctx)
    y = (x + 1).sum()
    return y.asnumpy()


def case_gpu_softmax():
    mx, np, npx, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((4, 10), ctx=ctx)
    y = npx.softmax(x, axis=1)
    return y.asnumpy()[0, :3]


def case_gpu_transpose():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.arange(2 * 8 * 16, ctx=ctx).reshape((2, 8, 16))
    y = x.transpose((0, 2, 1))
    return y.asnumpy()[0, 0, :4]


def case_gpu_dense_loss():
    mx, np, _, autograd, gluon, nn = import_mxnet()
    ctx = gpu_context(mx)
    net = nn.Dense(10)
    net.initialize(ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    x = np.ones((8, 32), ctx=ctx)
    y = np.zeros((8,), ctx=ctx)
    with autograd.record():
        l = loss(net(x), y)
    l.backward()
    return float(l.sum())


def case_gpu_scalar_to_host():
    mx, np, _, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    x = np.ones((32, 32), ctx=ctx)
    y = (x @ x).sum()
    return float(y)


def case_gpu_gru_scalar_to_host():
    mx, np, npx, _, _, _ = import_mxnet()
    ctx = gpu_context(mx)
    batch_size, num_steps, num_inputs, num_hiddens = 32, 8, 64, 32
    indices = np.arange(batch_size * num_steps, ctx=ctx).reshape(
        batch_size, num_steps
    ) % num_inputs
    inputs = npx.one_hot(indices.astype("int32"), num_inputs).transpose((1, 0, 2))

    def weight(*shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)

    W_xz, W_hz, b_z = weight(num_inputs, num_hiddens), weight(num_hiddens, num_hiddens), np.zeros(num_hiddens, ctx=ctx)
    W_xr, W_hr, b_r = weight(num_inputs, num_hiddens), weight(num_hiddens, num_hiddens), np.zeros(num_hiddens, ctx=ctx)
    W_xh, W_hh, b_h = weight(num_inputs, num_hiddens), weight(num_hiddens, num_hiddens), np.zeros(num_hiddens, ctx=ctx)

    H = np.zeros((batch_size, num_hiddens), ctx=ctx)
    for X in inputs:
        Z = npx.sigmoid(X @ W_xz + H @ W_hz + b_z)
        R = npx.sigmoid(X @ W_xr + H @ W_hr + b_r)
        H_tilde = np.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        H = Z * H + (1 - Z) * H_tilde
    return float(H.sum())


def case_gpu_rnn_loss_to_host():
    mx, np, npx, _, gluon, _ = import_mxnet()
    ctx = gpu_context(mx)
    batch_size, num_steps, vocab_size, num_hiddens = 1024, 32, 28, 32
    X_idx = np.arange(batch_size * num_steps, ctx=ctx).reshape(
        batch_size, num_steps
    ) % vocab_size
    Y_idx = (X_idx + 1) % vocab_size
    inputs = npx.one_hot(X_idx.astype("int32").T, vocab_size)

    def weight(*shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)

    W_xz = weight(vocab_size, num_hiddens)
    W_hz = weight(num_hiddens, num_hiddens)
    b_z = np.zeros(num_hiddens, ctx=ctx)
    W_xr = weight(vocab_size, num_hiddens)
    W_hr = weight(num_hiddens, num_hiddens)
    b_r = np.zeros(num_hiddens, ctx=ctx)
    W_xh = weight(vocab_size, num_hiddens)
    W_hh = weight(num_hiddens, num_hiddens)
    b_h = np.zeros(num_hiddens, ctx=ctx)
    W_hq = weight(num_hiddens, vocab_size)
    b_q = np.zeros(vocab_size, ctx=ctx)

    H = np.zeros((batch_size, num_hiddens), ctx=ctx)
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(X @ W_xz + H @ W_hz + b_z)
        R = npx.sigmoid(X @ W_xr + H @ W_hr + b_r)
        H_tilde = np.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H @ W_hq + b_q)
    logits = np.concatenate(outputs, axis=0)
    labels = Y_idx.T.reshape(-1)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    value = loss(logits, labels).mean()
    return float(np.exp(value))


def case_gpu_fused_gru_scalar_to_host():
    mx, np, _, _, _, _ = import_mxnet()
    from mxnet.gluon import rnn

    ctx = gpu_context(mx)
    net = rnn.GRU(32)
    net.initialize(ctx=ctx)
    X = np.ones((8, 32, 64), ctx=ctx)
    Y = net(X)
    return float(Y.sum())


def case_transformer_decoder_standalone():
    mx, np, npx, autograd, _, nn = import_mxnet()

    def masked_softmax(X, valid_lens):
        if valid_lens is None:
            return npx.softmax(X)
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = npx.sequence_mask(
            X.reshape(-1, shape[-1]),
            valid_lens,
            True,
            value=-1e6,
            axis=1,
        )
        return npx.softmax(X).reshape(shape)

    class DotProductAttention(nn.Block):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

        def forward(self, queries, keys, values, valid_lens=None):
            d = queries.shape[-1]
            scores = npx.batch_dot(queries, keys, transpose_b=True) / (d ** 0.5)
            weights = masked_softmax(scores, valid_lens)
            return npx.batch_dot(self.dropout(weights), values)

    class MultiHeadAttention(nn.Block):
        def __init__(self, num_hiddens, num_heads, dropout, use_bias=False):
            super().__init__()
            self.num_heads = num_heads
            self.attention = DotProductAttention(dropout)
            self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
            self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
            self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
            self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

        def transpose_qkv(self, X):
            X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
            X = X.transpose(0, 2, 1, 3)
            return X.reshape(-1, X.shape[2], X.shape[3])

        def transpose_output(self, X):
            X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
            X = X.transpose(0, 2, 1, 3)
            return X.reshape(X.shape[0], X.shape[1], -1)

        def forward(self, queries, keys, values, valid_lens):
            queries = self.transpose_qkv(self.W_q(queries))
            keys = self.transpose_qkv(self.W_k(keys))
            values = self.transpose_qkv(self.W_v(values))
            if valid_lens is not None:
                valid_lens = valid_lens.repeat(self.num_heads, axis=0)
            output = self.attention(queries, keys, values, valid_lens)
            return self.W_o(self.transpose_output(output))

    class PositionWiseFFN(nn.Block):
        def __init__(self, ffn_num_hiddens, ffn_num_outputs):
            super().__init__()
            self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False, activation="relu")
            self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

        def forward(self, X):
            return self.dense2(self.dense1(X))

    class AddNorm(nn.Block):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.ln = nn.LayerNorm()

        def forward(self, X, Y):
            return self.ln(self.dropout(Y) + X)

    class TransformerEncoderBlock(nn.Block):
        def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
            super().__init__()
            self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm1 = AddNorm(dropout)
            self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
            self.addnorm2 = AddNorm(dropout)

        def forward(self, X, valid_lens):
            Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
            return self.addnorm2(Y, self.ffn(Y))

    class TransformerDecoderBlock(nn.Block):
        def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
            super().__init__()
            self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm1 = AddNorm(dropout)
            self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
            self.addnorm2 = AddNorm(dropout)
            self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
            self.addnorm3 = AddNorm(dropout)

        def forward(self, X, state):
            enc_outputs, enc_valid_lens, cached = state
            key_values = X if cached is None else np.concatenate((cached, X), axis=1)
            if autograd.is_training():
                batch_size, num_steps, _ = X.shape
                dec_valid_lens = np.tile(
                    np.arange(1, num_steps + 1, ctx=X.ctx), (batch_size, 1)
                )
            else:
                dec_valid_lens = None
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
            Y = self.addnorm1(X, X2)
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
            Z = self.addnorm2(Y, Y2)
            return self.addnorm3(Z, self.ffn(Z)), [enc_outputs, enc_valid_lens, key_values]

    print("step: construct blocks", flush=True)
    encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
    decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5)
    encoder_blk.initialize()
    decoder_blk.initialize()
    print("step: make inputs", flush=True)
    X = np.ones((2, 100, 24))
    valid_lens = np.array([3, 2])
    print("step: compute encoder state", flush=True)
    enc_state = encoder_blk(X, valid_lens)
    print("step: call decoder block", flush=True)
    out, _ = decoder_blk(X, [enc_state, valid_lens, None])
    print("step: sync output", flush=True)
    return sync(out[0, 0, :4])


def strip_ipython_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def run_notebook_cells(path: Path, stop_cell: int | None):
    nb = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if stop_cell is not None and idx > stop_cell:
            break
        source = strip_ipython_magics("".join(cell.get("source", "")))
        first_line = next((line.strip() for line in source.splitlines() if line.strip()), "")
        print(f"executing cell {idx}: {first_line[:100]}", flush=True)
        exec(compile(source, str(path), "exec"), namespace)
        print(f"done cell {idx}", flush=True)
    return namespace


def case_bert_prefix(stop_cell: int | None):
    path = ROOT / "_notebooks/mxnet/chapter_natural-language-processing-pretraining/bert.ipynb"
    run_notebook_cells(path, stop_cell)


def case_transformer_prefix(stop_cell: int | None):
    path = ROOT / "_notebooks/mxnet/chapter_attention-mechanisms-and-transformers/transformer.ipynb"
    run_notebook_cells(path, stop_cell)


def case_notebook_prefix(notebook: str, stop_cell: int | None):
    if not notebook:
        raise ValueError("--notebook is required with --case notebook-prefix")
    path = ROOT / "_notebooks/mxnet" / notebook
    if path.suffix != ".ipynb":
        raise ValueError("--notebook must point to a .ipynb file")
    run_notebook_cells(path, stop_cell)


def case_transformer_decoder_block():
    path = ROOT / "_notebooks/mxnet/chapter_attention-mechanisms-and-transformers/transformer.ipynb"
    namespace = run_notebook_cells(path, 22)
    TransformerDecoderBlock = namespace["TransformerDecoderBlock"]
    encoder_blk = namespace["encoder_blk"]
    valid_lens = namespace["valid_lens"]
    np = namespace["np"]
    d2l = namespace["d2l"]

    print("step: construct decoder block", flush=True)
    decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
    print("step: initialize decoder block", flush=True)
    decoder_blk.initialize()
    print("step: make input", flush=True)
    X = np.ones((2, 100, 24))
    print("step: compute encoder state", flush=True)
    enc_state = encoder_blk(X, valid_lens)
    print("step: call decoder block", flush=True)
    out, state = decoder_blk(X, [enc_state, valid_lens, [None]])
    print(f"step: decoder returned shape={out.shape}, state_len={len(state)}", flush=True)
    d2l.check_shape(out, X.shape)


def main():
    ensure_nvidia_ld_path()
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "info",
            "gpu-sum",
            "gpu-softmax",
            "gpu-transpose",
            "gpu-dense-loss",
            "gpu-scalar-to-host",
            "gpu-gru-scalar-to-host",
            "gpu-rnn-loss-to-host",
            "gpu-fused-gru-scalar-to-host",
            "transformer-decoder-standalone",
            "gpu-all",
            "bert-prefix",
            "transformer-prefix",
            "notebook-prefix",
            "transformer-decoder-block",
        ],
        default="gpu-all",
    )
    parser.add_argument(
        "--stop-cell",
        type=int,
        default=None,
        help="Last notebook cell index to execute for *-prefix cases.",
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="MXNet notebook path relative to _notebooks/mxnet for notebook-prefix.",
    )
    args = parser.parse_args()

    if args.case == "info":
        case_info()
        return
    if args.case == "bert-prefix":
        case_bert_prefix(args.stop_cell)
        return
    if args.case == "transformer-prefix":
        case_transformer_prefix(args.stop_cell)
        return
    if args.case == "notebook-prefix":
        case_notebook_prefix(args.notebook, args.stop_cell)
        return
    if args.case == "transformer-decoder-block":
        case_transformer_decoder_block()
        return

    cases = {
        "gpu-sum": case_gpu_sum,
        "gpu-softmax": case_gpu_softmax,
        "gpu-transpose": case_gpu_transpose,
        "gpu-dense-loss": case_gpu_dense_loss,
        "gpu-scalar-to-host": case_gpu_scalar_to_host,
        "gpu-gru-scalar-to-host": case_gpu_gru_scalar_to_host,
        "gpu-rnn-loss-to-host": case_gpu_rnn_loss_to_host,
        "gpu-fused-gru-scalar-to-host": case_gpu_fused_gru_scalar_to_host,
        "transformer-decoder-standalone": case_transformer_decoder_standalone,
    }
    selected = cases.items() if args.case == "gpu-all" else [(args.case, cases[args.case])]
    ok = True
    for name, fn in selected:
        ok = run_case(name, fn) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
