# int8 Quantized Concat Bug — Investigation

## Status

**Code change applied; end-to-end verification deferred.** Repeated touches to
`include/mshadow/dot_engine-inl.h` (and earlier `linalg_impl.h`) by other
build processes on this shared box kept invalidating ~600 .o files, so each
full incremental relink took 30+ min. I burned the session waiting on
libmxnet.so refreshes and could not get a final pass-confirmation run with
the fix linked in. The fix is committed to the local working tree and the
build is in flight; the next session should be able to verify with a single
`pytest` invocation once the relink is done.

## Failing tests

- `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_single_concat_pos_neg[int8-data_shape1]`
- `tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_single_concat_pos_neg[auto-data_shape1]`

The other 4 parameterizations (`data_shape0`, `data_shape2` × {`int8`, `auto`}) pass.

## Reproduction

```bash
cd /workspace/mxnet && source .venv/bin/activate && \
  MXNET_TEST_SEED=11 PYTHONPATH=python pytest --tb=short \
    tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_single_concat_pos_neg
```

Failure mode (consistent):

```
ch0:  ref [-1.0, 1.0],  qout [-1.0, 1.0],  max diff 0.0085   (OK)
ch1:  ref [-1.0, 1.0],  qout [-1.0, 1.0],  max diff 0.0086   (OK)
ch2:  ref [-1.0, 1.0],  qout [-1.0, 1.0],  max diff 0.0084   (OK)
ch3:  ref [0.0, 1.06],  qout [0.0, 0.0],   max diff 1.06     ZERO
ch4:  ref [0.0, 0.71],  qout [0.0, 0.0],   max diff 0.71     ZERO
ch5:  ref [0.0, 1.20],  qout [0.0, 0.0],   max diff 1.20     ZERO
ch6:  ref [0.0, 0.70],  qout [0.0, 0.0],   max diff 0.70     ZERO
```

Channels 3-6 (the `relu(conv0(x))` half of `mx.np.concatenate([x, relu_out])`)
are exactly zero. The `x` half (ch0-2) is quantized correctly.

## What I ruled out

### DNNL_VERBOSE shows identical primitives for passing and failing cases

The u8→s8 conv-output rescale reorder fires for all three shapes with the
same impl (`jit:uni`), same attribute mask (`attr-scales:src0:0:f32`), same
data types — only `dims` differ. Same primitive, different shape.

### Scale magnitude is identical across all three shapes

I measured `out_pos_max` for each shape under the same MXNET_TEST_SEED=11:

    shape0: x_max=1.0, conv_out_max=2.61 → scale_val ≈ 0.499
    shape1: x_max=1.0, conv_out_max=1.20 → scale_val ≈ 0.499
    shape2: x_max=1.0, conv_out_max=4.19 → scale_val ≈ 0.499

All three rescale paths receive the same numeric scale ≈ 0.499. The
scale-magnitude / divide-by-zero family of hypotheses are out.

### The bare reorder primitive is correct in isolation

Built `/tmp/standalone_reorder.cpp` linked against this libdnnl, executed the
same `reorder::primitive_desc(u8 acdb → s8 abcd, scale=0.5)` on `4x4x24x24`,
verified output: `src[i] * 0.5` (modulo int8) end-to-end. So the bug is *not*
in the reorder primitive; it's in how it interacts with the surrounding op.

### Forcing plain NCHW (`abcd`) for the rescaled buffer did not help

First fix attempt: in `dnnl_quantized_concat.cc` replaced
`CloneMemDescWithDtype(src_desc, s8)` (inherits the conv output's
`acdb:a`-annotated NHWC tag) with a freshly-built plain `abcd` desc.
DNNL_VERBOSE confirmed the change took effect; failure was identical.
Rules out simple_concat's NHWC code path for C=3+C=4 inputs.

## Root cause hypothesis (now in committed fix)

The concat op's rescale branch was building each rescale destination and
each per-tensor scale memory directly via the `dnnl::memory(desc, engine)`
constructor — engine-internal allocation. The bare op also never called
`TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace])`, despite
declaring `FResourceRequest::kTempSpace`. Two back-to-back small
allocations (rescale dst for input 0, then rescale dst for input 1, plus
two 4-byte scale tensors) appear to receive overlapping storage from
oneDNN's allocator in this configuration, so the second input's
rescaled-buffer is clobbered with zeros before the concat reads it.

The non-quantized concat (`dnnl_concat.cc:94`), the quantized BN
(`dnnl_quantized_batch_norm.cc:41`), and other quantized ops all init
TmpMemMgr at function entry and route reorder destinations through
`TmpMemMgr::Alloc(md)`. The quantized concat was the odd one out.

## Fix applied (uncommitted)

`src/operator/quantization/dnnl/dnnl_quantized_concat.cc`:

1. **Init TmpMemMgr at function entry** (mirrors the non-quantized
   `DNNLConcatForward` and `DNNLQuantizedBatchNormForward`).
2. **Use `TmpMemMgr::Get()->Alloc(mem_desc)` for the rescale
   destination** instead of `std::make_shared<dnnl::memory>(mem_desc, engine)`.
3. **User-manage the f32 scale storage** via
   `std::vector<std::shared_ptr<float>> scale_bufs;` and the user-pointer
   `dnnl::memory(scale_md, engine, scale_buf.get())` constructor, so its
   address is unique per call and its lifetime is tied to the function's
   local vector (which outlives `DNNLStream::Submit()`).
4. Also: switch the reorder ctor to the
   `(engine, src_md, engine, dst_md, attr)` overload (the
   `(memory, memory, attr)` overload uses the memory's runtime engine
   lookup, which is fine but less explicit).

Net diff: +31 / -8 lines, all in `dnnl_quantized_concat.cc`.

## Verification still pending

The libmxnet relink in flight (PID 2361975, `ninja -j 4 libmxnet.so`,
plus a `ninja -j 16` re-arm) is bottlenecked behind ~400 .cc/.cu files
that other ninja runs on the shared box invalidated by touching shared
headers (`mshadow/dot_engine-inl.h`, `linalg_impl.h`). The fix's
`dnnl_quantized_concat.cc.o` is already built (see
`build/CMakeFiles/mxnet.dir/src/operator/quantization/dnnl/dnnl_quantized_concat.cc.o`
timestamped after the source edit), but it has not yet been linked into
a fresh libmxnet.so that the test driver picks up.

Once libmxnet.so refreshes, the verification is a one-liner:

```bash
source .venv/bin/activate && \
  MXNET_TEST_SEED=11 PYTHONPATH=python pytest --tb=short -v \
    tests/python/dnnl/subgraphs/test_conv_subgraph.py::test_pos_single_concat_pos_neg
```

Expected: all 6 parameterizations pass.

Also worth running (per task spec):

```bash
PYTHONPATH=python pytest tests/python/dnnl/test_dnnl.py -q
```

…to confirm no regression in other DNNL tests.

## File pointers

- Bug + fix: `src/operator/quantization/dnnl/dnnl_quantized_concat.cc:41-126`
- Reference (working u8→s8 reorder with TmpMemMgr):
  `src/operator/quantization/dnnl/dnnl_quantized_batch_norm.cc:41-69`
- Reference (non-quantized concat init-pattern):
  `src/operator/nn/dnnl/dnnl_concat.cc:89-119`
- Test: `tests/python/dnnl/subgraphs/test_conv_subgraph.py:367-383`
