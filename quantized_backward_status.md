# Quantized-op backward status (issues.md #5)

Run date: 2026-05-17  
Test file: `tests/python/dnnl/subgraphs/test_quantized_backward.py`  
Result: **13 PASSED, 4 XFAIL (strict), 0 FAILED**

---

## Test results

| Test | Status | Description |
|------|--------|-------------|
| `test_fc_quantized_forward_runs` | PASS | Quantized FC forward produces finite output |
| `test_fc_quantized_backward_no_crash` | PASS | Backward through quantized FC does not crash; output is finite (not NaN/Inf) |
| `test_fc_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is identically zero — `quantize_v2` at graph input has no backward |
| `test_fc_quantized_weight_grad_nonzero` | **XFAIL** | Weight gradient is identically zero — `grad_req='null'` set by `quantize_net` |
| `test_conv_quantized_forward_runs` | PASS | Quantized Conv2D forward produces finite output with correct shape |
| `test_conv_quantized_backward_no_crash` | PASS | Backward through quantized Conv does not crash |
| `test_conv_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is identically zero — same root cause as FC |
| `test_composite_quantized_forward_runs` | PASS | Conv→ReLU→GlobalAvgPool→Dense quantized forward runs |
| `test_composite_quantized_backward_no_crash` | PASS | Composite quantized backward does not crash |
| `test_composite_quantized_backward_nonzero_input_grad` | **XFAIL** | Input gradient is zero — same root cause |
| `test_composite_quantized_sign_agreement` | PASS | Documents sign-agreement check; passes because quantized grad is all-zero (skips comparison) |
| `test_calibration_round_trip_fc` | PASS | Two naive-calib quantizations of the same FC net give identical outputs |
| `test_calibration_round_trip_conv` | PASS | Two naive-calib quantizations of the same Conv net give identical outputs |
| `test_fc_backward_multiple_forward_passes` | PASS | Multiple forward passes before backward do not corrupt state |
| `test_conv_quantized_output_changes_with_input` | PASS | Quantized Conv output is sensitive to input (forward path is live) |
| `test_fc_quantized_output_changes_with_input` | PASS | Quantized FC output is sensitive to input |
| `test_quantized_grad_req_all_null` | PASS | Documents that `quantize_net` sets `grad_req='null'` on all quantized params |

---

## Root cause of xfails

**`quantize_v2` at graph input has no backward registered.**

`quantize_net` inserts a `quantize_v2` node immediately before each quantized
DNNL op. This node converts `float32 → int8`. Its registered backward returns
all-zeros. As a result:

1. The input gradient `x.grad` is identically zero for **all** quantized graphs.
2. All quantized weight params have `grad_req='null'` (set explicitly by
   `quantize_net` for inference safety). Even when forced to `'write'`, the
   backward returns zero because the quantized op itself has no weight-backward.

---

## What works

- **Forward inference** is solid: correct shapes, finite values, output changes
  with input for both FC and Conv.
- **Backward does not crash**: calling `.backward()` through a quantized graph
  is safe — it just returns zeros.
- **Calibration round-trip** is deterministic: two independent quantizations of
  the same net with the same calibration data give bit-identical outputs.
- **State is stable** across multiple forward passes before backward.

---

## What to fix next

To enable fine-tuning (QAT) of quantized MXNet networks:

1. **Implement straight-through estimator (STE) for `quantize_v2`.**  
   The backward of `quantize_v2` should pass gradients straight through
   (identity for inputs in `[min_range, max_range]`, zero outside).
   File to change: `src/operator/contrib/quantization.cc` — add
   `MXNET_OPERATOR_REGISTER_...` backward for `_contrib_quantize_v2`.

2. **Register weight backward for `quantized_sg_onednn_fully_connected` and
   `quantized_sg_onednn_conv`.**  These fused subgraph ops need to accumulate
   weight gradients (similar to the FP32 operators they replace).

3. **Remove `v.grad_req = 'null'` from `quantize_net`** (or make it
   conditional) so that training loops can allocate gradient buffers for
   quantized parameters.

Once (1)–(3) are done, remove the `xfail` markers from the four tests above.
