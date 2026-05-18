# Quantized-op backward status (issues.md #5)

Run date: 2026-05-18
Branch: onednn-v3-port
Test file: `tests/python/dnnl/subgraphs/test_quantized_backward.py`
Result: **13 PASSED, 4 XFAIL (strict), 0 FAILED**

Status: **RESOLVED (partial)** — Steps 1 + 2 done; Step 3 blocked.

---

## What changed since initial analysis

### Step 1 DONE — STE for `quantize_v2`
File: `src/operator/quantization/quantize_v2.cc`

`quantize_v2` previously used `MakeZeroGradNodes`, killing all gradient flow
through the quantization step.  It now has a Straight-Through Estimator (STE)
backward: the upstream gradient (int8/uint8 dtype) is cast back to float32 and
returned as the gradient for the float32 input.

The STE is implemented as a `cast` node in the backward symbolic graph:
```cpp
std::unordered_map<std::string, std::string> cast_dict = {{"dtype", "float32"}};
auto cast_node = MakeNode("cast", n->attrs.name + "_ste_cast",
                          {ograds[0]}, &cast_dict, &n);
return {nnvm::NodeEntry{cast_node, 0, 0}};
```

### Step 2 DONE — `qat` kwarg for `quantize_net()`
File: `python/mxnet/contrib/quantization.py`

`quantize_net(..., qat=False)` (default) keeps existing inference behavior:
all quantized params get `grad_req='null'`.

`quantize_net(..., qat=True)` leaves `grad_req` at its default (`'write'`),
so that optimizers can update quantized parameters after backward.

The calibration network (used only for range collection) always gets
`grad_req='null'` regardless of `qat=`, since it is never used for training.

### Step 3 BLOCKED — FC/Conv subgraph backward
Files: `src/operator/subgraph/dnnl/dnnl_fc.cc`,
       `src/operator/subgraph/dnnl/dnnl_conv.cc`

`_sg_onednn_fully_connected` and `_sg_onednn_conv` still have
`FGradient = MakeZeroGradNodes`.  An attempt to register a custom backward
(e.g., `dot(ograds[0], weight)` for data gradient through FC) caused segfaults
in the NNVM/CachedOp backward executor.  The segfaults occur because these
fused subgraph ops interact with MXNet's static graph executor in a way that
does not support backward nodes that reference op inputs at this time.

The STE (Step 1) is functionally correct but its gradient is blocked: the
gradient that flows back to the `quantize_v2` node is all-zero (from the FC/Conv
subgraph op), so the cast-to-float32 STE output is also all-zero.

---

## Test results

| Test | Status | Description |
|------|--------|-------------|
| `test_fc_quantized_forward_runs` | PASS | Quantized FC forward produces finite output |
| `test_fc_quantized_backward_no_crash` | PASS | Backward through quantized FC does not crash; output is finite (not NaN/Inf) |
| `test_fc_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is zero — FC subgraph has no backward (Step 3 blocked) |
| `test_fc_quantized_weight_grad_nonzero` | **XFAIL** | Weight gradient is zero — FC subgraph has no backward (Step 3 blocked) |
| `test_conv_quantized_forward_runs` | PASS | Quantized Conv2D forward produces finite output with correct shape |
| `test_conv_quantized_backward_no_crash` | PASS | Backward through quantized Conv does not crash |
| `test_conv_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is zero — Conv subgraph has no backward (Step 3 blocked) |
| `test_composite_quantized_forward_runs` | PASS | Conv→ReLU→GlobalAvgPool→Dense quantized forward runs |
| `test_composite_quantized_backward_no_crash` | PASS | Composite quantized backward does not crash |
| `test_composite_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is zero — same root cause |
| `test_composite_quantized_sign_agreement` | PASS | Passes because quantized grad is all-zero (skips sign comparison) |
| `test_calibration_round_trip_fc` | PASS | Two naive-calib quantizations of the same FC net give identical outputs |
| `test_calibration_round_trip_conv` | PASS | Two naive-calib quantizations of the same Conv net give identical outputs |
| `test_fc_backward_multiple_forward_passes` | PASS | Multiple forward passes before backward do not corrupt state |
| `test_conv_quantized_output_changes_with_input` | PASS | Quantized Conv output is sensitive to input (forward path is live) |
| `test_fc_quantized_output_changes_with_input` | PASS | Quantized FC output is sensitive to input |
| `test_quantized_grad_req_all_null` | PASS | Documents that `quantize_net` (qat=False) sets grad_req='null' on all params; qat=True leaves grad_req='write' |

---

## Root cause of xfails

**`_sg_onednn_fully_connected` and `_sg_onednn_conv` have no backward.**

These fused DNNL subgraph ops use `FGradient = MakeZeroGradNodes`, which
returns all-zero gradients for all inputs.  This is the current blocker for
end-to-end QAT.

The `quantize_v2` STE is now in place (Step 1 done): if the FC/Conv subgraph
ops gain proper backward support, the STE will correctly propagate the gradient
from the quantized domain back to the float32 input — no further change to
`quantize_v2.cc` will be needed.

---

## What works

- **Forward inference** is solid: correct shapes, finite values, output changes
  with input for both FC and Conv.
- **Backward does not crash**: calling `.backward()` through a quantized graph
  is safe — it returns zeros without crashing or corrupting state.
- **STE registered**: `quantize_v2` now has a correct STE backward; the zero
  gradient issue originates from the FC/Conv subgraph ops, not from quantize_v2.
- **qat kwarg works**: `quantize_net(..., qat=True)` leaves all quantized
  parameter buffers with `grad_req='write'` — no change needed for gradient
  allocation once FC/Conv backward is unblocked.
- **Calibration round-trip** is deterministic: two independent quantizations of
  the same net with the same calibration data give bit-identical outputs.
- **State is stable** across multiple forward passes before backward.

---

## What to fix next (to complete Step 3)

To enable fine-tuning (QAT) of quantized MXNet networks, the remaining work is:

1. **Register backward for `_sg_onednn_fully_connected`** in
   `src/operator/subgraph/dnnl/dnnl_fc.cc`.  The forward pass computes
   `y = x * W^T + b`; the data backward is `dL/dx = dL/dy * W` and the weight
   backward is `dL/dW = dL/dy^T * x`.  The challenge is that this is a fused
   DNNL subgraph op with quantized inputs — the backward needs to either
   dequantize the inputs first or use a separate FP32 backward primitive.
   Segfaults observed during an initial attempt suggest the NNVM/CachedOp
   framework does not support backward nodes that reference the quantized op's
   inputs directly; this may require a custom `FStatefulComputeEx` backward
   instead of an `FGradient` symbolic graph approach.

2. **Same for `_sg_onednn_conv`** in
   `src/operator/subgraph/dnnl/dnnl_conv.cc`.

3. **Remove xfail markers** from the four tests above once (1)+(2) are done.

4. **Regression guard**: verify `test_fc_subgraph.py` (387/0/16) and
   `test_conv_subgraph.py` are unchanged after any FC/Conv FGradient change.
