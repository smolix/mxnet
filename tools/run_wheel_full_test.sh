#!/usr/bin/env bash
#
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
#
# Acceptance test for a built MXNet wheel.  Installs into a fresh venv
# (NOT the dev venv), then runs the full CPU + DNNL + GPU + quantization
# suite under controlled concurrency.  Captures per-shard logs into a
# report directory.
#
# This is intentionally separate from tools/run_fp16_remote_smoke.sh:
# that one is build-from-source on a remote target; this one is wheel
# acceptance on the build host.
#
# Usage:
#   tools/run_wheel_full_test.sh [<wheel-path>]
#
# Defaults to the newest dist/mxnet-*.whl.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WHEEL="${1:-}"
if [ -z "$WHEEL" ]; then
    WHEEL=$(ls -1t dist/mxnet-*.whl 2>/dev/null | head -n1)
fi
if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
    echo "Could not find a wheel at: $WHEEL" >&2
    echo "Usage: $0 [<wheel-path>]" >&2
    exit 2
fi
WHEEL=$(readlink -f "$WHEEL")
WHEEL_NAME=$(basename "$WHEEL")

REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/wheel-test-$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_DIR="${VENV_DIR:-$REPORT_DIR/.venv}"
mkdir -p "$REPORT_DIR/shards"
exec > >(tee -a "$REPORT_DIR/run.log") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
section() { log; log "=== $* ==="; }

# Concurrency caps — keep load under 100 on a 64-core box; ideal 64.
# xdist worker counts pick up these env vars and propagate to subprocesses.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PARALLEL_CPU=${PARALLEL_CPU:-8}
PARALLEL_DNNL=${PARALLEL_DNNL:-1}     # DNNL tests have global state
PARALLEL_GPU=${PARALLEL_GPU:-4}
PARALLEL_QUANT=${PARALLEL_QUANT:-1}
TIMEOUT_SHARD_MIN=${TIMEOUT_SHARD_MIN:-30}

# Pytest invocations should NOT install xdist via pip-install-on-the-fly.
export PYTEST_ADDOPTS="-p no:faulthandler --tb=short --color=no --durations=20"

section "Inputs"
log "Wheel: $WHEEL"
log "Wheel size: $(du -h "$WHEEL" | cut -f1)"
log "Report dir: $REPORT_DIR"
log "Venv: $VENV_DIR"

# ----------------------------------------------------------------------
# Stage 1: fresh venv + wheel install
# ----------------------------------------------------------------------
section "Create fresh venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

section "Install wheel + test dependencies (clean venv)"
# The wheel declares its install_requires (numpy, requests, graphviz,
# nvidia-cudnn-cu13, nvidia-nccl-cu13).  Pip resolves them here, ensuring the
# install path the wheel describes actually works on this host.
python -m pip install "$WHEEL"
# Test-time extras: pytest + xdist + timeout.  These are not part of the
# wheel's install_requires intentionally.
#
# scipy is consumed by several statistical-oracle nodes in test_random,
# test_operator, and gpu_operator shards; without it pytest collect-errors
# the file instead of running it, which makes acceptance summaries
# unreadable.  matplotlib covers a handful of plotting-adjacent nodes.
python -m pip install pytest pytest-xdist pytest-timeout scipy matplotlib

section "Verify wheel-installed mxnet is the one we get"
python - <<PY
import mxnet, os, sys
print("mx.__version__:", mxnet.__version__)
print("mx.__file__   :", mxnet.__file__)
# Must come from the venv site-packages, NOT the repo's editable install.
assert "$VENV_DIR" in mxnet.__file__, f"mxnet resolved outside venv: {mxnet.__file__}"
print()
import mxnet.runtime as rt
feats = rt.Features()
print(feats)
print()
print("GPU count:", mxnet.device.num_gpus())
PY
rc=$?
if [ $rc -ne 0 ]; then
    log "Wheel import failed — aborting before running any tests."
    exit 3
fi

# Save the feature snapshot.
python -c 'import mxnet, mxnet.runtime as rt; print("version:", mxnet.__version__); print(rt.Features())' \
    > "$REPORT_DIR/features.txt"

# ----------------------------------------------------------------------
# Stage 2: run shards
# ----------------------------------------------------------------------
run_shard() {
    # run_shard <id> <description> -- <pytest args>
    local id="$1"; shift
    local desc="$1"; shift
    [ "$1" = "--" ] && shift
    local logf="$REPORT_DIR/shards/${id}.log"
    section "$id — $desc"
    {
        echo "### $desc"
        echo "### pytest args: $*"
        echo "### started: $(date -u)"
        echo
    } >"$logf"
    timeout "${TIMEOUT_SHARD_MIN}m" python -m pytest "$@" >>"$logf" 2>&1
    local rc=$?
    {
        echo
        echo "### finished: $(date -u)   rc=$rc"
    } >>"$logf"
    # Print the last 5 lines + summary line so the operator can see progress.
    tail -10 "$logf" | sed "s|^|  $id: |"
    log "  $id: rc=$rc, log=$logf"
    return $rc
}

# Run from the repo's tests/ directory so relative paths in test files work,
# but ensure mxnet comes from the venv (verified above).
cd "$REPO_ROOT"

# ----- CPU / unittest lane -----
run_shard cpu_xop19 \
    "XOP19 oneDNN req regressions (new in this branch)" -- \
    tests/python/dnnl/test_xop19_onednn_req.py -v

run_shard cpu_optimized_validation \
    "Python optimized validation (XOP22)" -- \
    tests/python/unittest/test_python_optimized_validation.py -v

run_shard cpu_optimizer \
    "Optimizer suite (XOP22 fan-out)" -- \
    tests/python/unittest/test_optimizer.py -v -n "$PARALLEL_CPU"

run_shard cpu_gluon_parameter \
    "Gluon Parameter (XOP22)" -- \
    tests/python/unittest/test_gluon.py -v -k Parameter -n "$PARALLEL_CPU"

run_shard cpu_layer_norm \
    "LayerNorm parametric (XOP21 int64_t channel_size)" -- \
    tests/python/unittest/test_operator.py -v -k 'test_layer_norm' -n "$PARALLEL_CPU"

run_shard cpu_group_norm \
    "GroupNorm parametric (XOP21)" -- \
    tests/python/unittest/test_operator.py -v -k 'test_group_norm' -n "$PARALLEL_CPU"

# The big CPU unit-test lane — same shape as the existing local-validation
# matrix.  Excludes test_operator.py because it gets its own subset above and
# the monolithic file is the worst offender for xdist OOM.
run_shard cpu_unittest \
    "CPU unittest (excluding test_operator.py)" -- \
    tests/python/unittest -v -n "$PARALLEL_CPU" \
    --ignore=tests/python/unittest/test_operator.py \
    --ignore=tests/python/unittest/test_random.py

# test_operator.py — biggest CPU surface, run with stricter caps.
run_shard cpu_test_operator \
    "CPU test_operator.py (big surface)" -- \
    tests/python/unittest/test_operator.py -v -n 4

# test_random.py — separated to keep order-sensitive bus error (FS12) isolated.
run_shard cpu_test_random \
    "CPU test_random.py (FS12 isolated)" -- \
    tests/python/unittest/test_random.py -v

# ----- DNNL / oneDNN lane (serial, has global state) -----
run_shard dnnl_smoke \
    "DNNL smoke" -- \
    tests/python/dnnl/test_dnnl.py -v -n "$PARALLEL_DNNL"

run_shard dnnl_quantization \
    "DNNL quantization" -- \
    tests/python/dnnl/test_quantization_dnnl.py -v

run_shard dnnl_amp \
    "DNNL AMP / BF16" -- \
    tests/python/dnnl/test_amp.py -v

run_shard dnnl_bf16_operator \
    "DNNL BF16 operator (skips on Zen 2 EPYC)" -- \
    tests/python/dnnl/test_bf16_operator.py -v

run_shard dnnl_batchdot \
    "DNNL batch-dot attention regression" -- \
    tests/python/dnnl/test_batch_dot_attention_regression.py -v

run_shard dnnl_layer_norm \
    "DNNL LayerNorm" -- \
    tests/python/dnnl/test_dnnl_layer_norm.py -v

run_shard dnnl_subgraphs_matmul \
    "DNNL subgraph: matmul" -- \
    tests/python/dnnl/subgraphs/test_matmul_subgraph.py -v

run_shard dnnl_subgraphs_conv \
    "DNNL subgraph: conv" -- \
    tests/python/dnnl/subgraphs/test_conv_subgraph.py -v

# quantized backward — has strict xfails documenting the B4 framework block.
if [ -f tests/python/dnnl/subgraphs/test_quantized_backward.py ]; then
    run_shard dnnl_subgraphs_qat_backward \
        "DNNL subgraph: QAT backward (strict xfails per B4)" -- \
        tests/python/dnnl/subgraphs/test_quantized_backward.py -v
fi

# ----- Quantization lane -----
run_shard quant_general \
    "Quantization general" -- \
    tests/python/quantization/test_quantization.py -v -n "$PARALLEL_QUANT"

# ----- GPU lane (only if GPUs visible) -----
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 0 ]; then
    run_shard gpu_amp \
        "GPU AMP" -- \
        tests/python/gpu/test_amp.py -v -n "$PARALLEL_GPU"

    run_shard gpu_amp_init \
        "GPU AMP init" -- \
        tests/python/gpu/test_amp_init.py -v -n "$PARALLEL_GPU"

    if [ -f tests/python/gpu/test_fp16_batch_dot.py ]; then
        run_shard gpu_fp16_batchdot "GPU fp16 batch-dot parity" -- \
            tests/python/gpu/test_fp16_batch_dot.py -v
    fi

    if [ -f tests/python/gpu/test_cublaslt_fc.py ]; then
        run_shard gpu_cublaslt "cuBLASLt FC + dtype matrix" -- \
            tests/python/gpu/test_cublaslt_fc.py -v
    fi

    if [ -f tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py ]; then
        run_shard gpu_cudnn_fallback "cuDNN frontend no-plan fallback" -- \
            tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py -v
    fi

    if [ -f tests/python/gpu/test_deconv_tf32.py ]; then
        run_shard gpu_tf32_deconv "TF32 deconv parity" -- \
            tests/python/gpu/test_deconv_tf32.py -v
    fi

    if [ -f tests/python/gpu/test_reducer_regressions.py ]; then
        run_shard gpu_reducer "GPU reducer regressions" -- \
            tests/python/gpu/test_reducer_regressions.py -v
    fi

    run_shard gpu_fork_safe_dnnl \
        "GPU DNNL fork-safety DataLoader" -- \
        tests/python/gpu/test_fu4_fork_safe_dnnl.py -v

    if [ -f tests/python/gpu/test_extensions_gpu.py ]; then
        run_shard gpu_extensions "GPU extensions" -- \
            tests/python/gpu/test_extensions_gpu.py -v
    fi

    # test_operator_gpu.py is sharded by issues.md guidance — running the
    # full file in one shot is OOM-prone.  Use the same split as the local
    # validation matrix: NumPy-heavy first, then classic complement.
    run_shard gpu_operator_numpy \
        "GPU test_operator_gpu.py — numpy/linalg/reduce/broadcast subset" -- \
        tests/python/gpu/test_operator_gpu.py -v -n "$PARALLEL_GPU" \
        -k 'np_ or numpy or einsum or linalg or reduce or broadcast or elemwise or matrix'

    run_shard gpu_operator_classic \
        "GPU test_operator_gpu.py — classic complement" -- \
        tests/python/gpu/test_operator_gpu.py -v -n "$PARALLEL_GPU" \
        -k 'not (np_ or numpy or einsum or linalg or reduce or broadcast or elemwise or matrix)'

    if [ -f tests/python/test_quantization_gpu.py ]; then
        run_shard gpu_quantization "GPU quantization wrapper" -- \
            tests/python/test_quantization_gpu.py -v
    fi
fi

# ----------------------------------------------------------------------
# Stage 3: summary
# ----------------------------------------------------------------------
section "Summarize"

{
    echo "# MXNet wheel acceptance report"
    echo
    echo "Wheel: \`$WHEEL_NAME\`"
    echo "Host : $(hostname)"
    echo "When : $(date -u)"
    echo
    echo "## Wheel runtime features"
    echo
    echo '```'
    cat "$REPORT_DIR/features.txt"
    echo '```'
    echo
    echo "## Shard results"
    echo
    printf '| Shard | Result | pytest summary |\n'
    printf '|---|---|---|\n'
    for f in "$REPORT_DIR"/shards/*.log; do
        [ -f "$f" ] || continue
        id=$(basename "$f" .log)
        last=$(grep -E '=+ .*(passed|failed|error|skipped|xfailed|xpassed).* in [0-9.]+s' "$f" | tail -1)
        rc=$(grep -oE 'rc=[0-9-]+' "$f" | tail -1)
        if [ -n "$last" ]; then
            if echo "$last" | grep -q failed; then
                marker="FAIL"
            elif echo "$last" | grep -q error; then
                marker="ERROR"
            else
                marker="PASS"
            fi
            printf '| %s | %s | %s |\n' "$id" "$marker" "${last//|/\\|}"
        else
            printf '| %s | INCOMPLETE | %s |\n' "$id" "$rc"
        fi
    done
    echo
    echo "Per-shard logs: \`$REPORT_DIR/shards/\`"
} > "$REPORT_DIR/summary.md"

cat "$REPORT_DIR/summary.md"

log
log "Done. Report: $REPORT_DIR"
log "  bundle: tar czf $(basename "$REPORT_DIR").tar.gz -C $(dirname "$REPORT_DIR") $(basename "$REPORT_DIR")"
