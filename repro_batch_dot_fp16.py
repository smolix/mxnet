#!/usr/bin/env python3
"""
Reproducer for apache/mxnet#18584:
batch_dot vs dot give different fp16 GPU results.
"""
import numpy as np
import mxnet as mx

ctx = mx.gpu(0)
mx.random.seed(42)

B, M, K, N = 8, 64, 64, 64

# Random fp16 inputs (fixed seed for reproducibility)
np.random.seed(42)
a_np = np.random.randn(B, M, K).astype(np.float16)
b_np = np.random.randn(B, K, N).astype(np.float16)

A = mx.nd.array(a_np, ctx=ctx, dtype=np.float16)
B_nd = mx.nd.array(b_np, ctx=ctx, dtype=np.float16)

# batch_dot result
batch_result = mx.nd.batch_dot(A, B_nd)

# Manual stacked dot result (each slice uses 2D dot)
slices = [mx.nd.dot(A[i], B_nd[i]) for i in range(A.shape[0])]
manual_result = mx.nd.stack(*slices)

# Compare
br = batch_result.asnumpy()
mr = manual_result.asnumpy()

abs_diff = np.abs(br - mr)
denom = np.abs(mr) + 1e-6
rel_err = (abs_diff / denom)
max_rel = rel_err.max()
mean_rel = rel_err.mean()

print(f"batch_dot vs manual dot (stacked 2D slices)")
print(f"  max absolute diff:  {abs_diff.max():.6f}")
print(f"  max relative error: {max_rel:.6f}")
print(f"  mean relative error:{mean_rel:.6f}")
print(f"  max |br|:  {np.abs(br).max():.4f}")
print(f"  max |mr|:  {np.abs(mr).max():.4f}")

# Check agreement at fp16 roundoff level (~5e-3)
TOLS = 5e-3
if max_rel < TOLS:
    print(f"\nPASS: max rel error {max_rel:.2e} < {TOLS}")
else:
    print(f"\nFAIL: max rel error {max_rel:.2e} >= {TOLS} (parity with dot broken)")
