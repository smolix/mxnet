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
# Acceptance test for the macOS arm64 CPU MXNet wheel.
#
# Installs the wheel into a FRESH venv (never the build/dev venv), verifies the
# installed package is the one under test, then runs the CPU-runnable parts of
# the Python test suite (unittest + dnnl + quantization + array-api + amp +
# profiling).  Per-shard logs land under macos_wheel_test/shards/, and every
# FAILED/ERROR line plus each shard's pytest summary is aggregated into
# macos_wheel_test/errors.log.
#
# GPU (tests/python/gpu, test_quantization_gpu.py) and ONNX (tests/python/onnx)
# are intentionally skipped: this host has no CUDA, and the ONNX path is broken
# upstream for MXNet 2.0 (errors at collection).  They are reported as SKIPPED
# lanes in the summary so the omission is explicit.
#
# Usage: tools/run_macos_wheel_full_test.sh [<wheel-path>]
# Defaults to the newest dist/mxnet-*.whl.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WHEEL="${1:-}"
[ -z "$WHEEL" ] && WHEEL="$(ls -1t dist/mxnet-*.whl 2>/dev/null | head -n1)"
if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
    echo "Could not find a wheel at: '$WHEEL'  (build one first)" >&2
    exit 2
fi
WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"

REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/macos_wheel_test}"
VENV="${VENV:-$REPORT_DIR/.venv}"
SHARDS="$REPORT_DIR/shards"
ERRLOG="$REPORT_DIR/errors.log"
RUNLOG="$REPORT_DIR/run.log"
SUMMARY="$REPORT_DIR/summary.md"
mkdir -p "$SHARDS"
: > "$ERRLOG"
: > "$SUMMARY"
exec > >(tee "$RUNLOG") 2>&1

WHEEL_TEST_PYTHON="${WHEEL_TEST_PYTHON:-python3.12}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "Wheel:   $WHEEL  ($(du -h "$WHEEL" | cut -f1))"
log "Report:  $REPORT_DIR"
log "Venv:    $VENV  (python: $WHEEL_TEST_PYTHON)"

# ---------------------------------------------------------------------------
# Fresh venv + wheel install
# ---------------------------------------------------------------------------
log "Creating fresh venv and installing the wheel"
rm -rf "$VENV"
uv venv "$VENV" --python "$WHEEL_TEST_PYTHON"
# The wheel's install_requires pulls numpy<2, requests, graphviz, packaging,
# scipy-openblas32 and (OpenCV build) opencv-python.
uv pip install --python "$VENV/bin/python" "$WHEEL"
# Test-time extras (not part of install_requires): pytest stack + scipy/matplotlib
# statistical oracles. pytest-env applies pytest.ini's `env = MXNET_HOME=...`.
uv pip install --python "$VENV/bin/python" \
    pytest pytest-xdist pytest-timeout pytest-env scipy matplotlib

PYBIN="$VENV/bin/python"

log "Verifying the installed wheel is the package under test"
"$PYBIN" - "$VENV" <<'PY'
import sys, mxnet, mxnet.runtime as rt
venv = sys.argv[1]
print("mx.__version__:", mxnet.__version__)
print("mx.__file__   :", mxnet.__file__)
assert venv in mxnet.__file__, f"mxnet resolved outside venv: {mxnet.__file__}"
feats = rt.Features()
print(feats)
assert feats.is_enabled("OPENCV"), "OPENCV feature is OFF in the installed wheel"
print("num_gpus:", mxnet.device.num_gpus())
import cv2  # bundled opencv-python; mx.image python helpers use it
print("cv2:", cv2.__version__)
PY
if [ $? -ne 0 ]; then
    log "Wheel import/verification FAILED — aborting before running tests."
    echo "FATAL: wheel import/verification failed (see run.log)" >> "$ERRLOG"
    exit 3
fi
"$PYBIN" -c 'import mxnet, mxnet.runtime as rt; print(mxnet.__version__); print(rt.Features())' \
    > "$REPORT_DIR/features.txt"

# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------
export MXNET_TEST_USE_INSTALLED_MXNET=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Per-test 300s timeout (thread method works without signals under xdist),
# short tracebacks, no faulthandler plugin, plain output.
export PYTEST_ADDOPTS="-p no:faulthandler --timeout=300 --timeout-method=thread --tb=short --color=no -ra"

run_shard() {
    # run_shard <id> <pytest args...>
    local id="$1"; shift
    local logf="$SHARDS/${id}.log"
    log "shard ${id}: pytest $*"
    {
        echo "### shard: $id"
        echo "### pytest $*"
        echo "### started: $(date -u)"
        echo
    } > "$logf"
    "$PYBIN" -m pytest "$@" --continue-on-collection-errors >> "$logf" 2>&1
    local rc=$?
    echo "### finished: $(date -u)  rc=$rc" >> "$logf"
    local summ
    summ="$(grep -E '=+ .*(passed|failed|error|skipped|xfailed|xpassed|warning).* in [0-9.]+s' "$logf" | tail -1)"
    {
        echo "##### SHARD ${id}  (rc=${rc})"
        echo "      pytest $*"
        echo "      summary: ${summ:-<none — crash/timeout/collection error>}"
        grep -E '^(FAILED|ERROR) ' "$logf" | sed 's/^/      /'
        echo
    } >> "$ERRLOG"
    log "  -> ${id}: rc=$rc  ${summ:-<no summary>}"
    printf '%s\t%s\t%s\n' "$id" "$rc" "${summ:-<no summary>}" >> "$REPORT_DIR/shard_index.tsv"
}

: > "$REPORT_DIR/shard_index.tsv"

# ----- CPU unittest lane -----------------------------------------------------
# Split the two heaviest / most crash-prone files out of the parallel lane.
run_shard unittest_main \
    tests/python/unittest -n 4 \
    --ignore=tests/python/unittest/test_operator.py \
    --ignore=tests/python/unittest/test_random.py

run_shard unittest_operator \
    tests/python/unittest/test_operator.py -n 2

run_shard unittest_random \
    tests/python/unittest/test_random.py

# ----- oneDNN lane (serial; has global state) --------------------------------
run_shard dnnl \
    tests/python/dnnl

# ----- quantization lane -----------------------------------------------------
run_shard quantization \
    tests/python/quantization

# ----- smaller CPU lanes -----------------------------------------------------
run_shard array_api \
    tests/python/array-api

run_shard amp \
    tests/python/amp -n 2

run_shard profiling \
    tests/python/profiling

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
{
    echo "# macOS arm64 CPU wheel — test summary"
    echo
    echo "Wheel: \`$(basename "$WHEEL")\`"
    echo "When : $(date -u)"
    echo
    echo "## Runtime features"
    echo '```'
    cat "$REPORT_DIR/features.txt"
    echo '```'
    echo
    echo "## Shards"
    echo
    printf '| Shard | rc | Summary |\n|---|---|---|\n'
    while IFS=$'\t' read -r id rc summ; do
        printf '| %s | %s | %s |\n' "$id" "$rc" "${summ//|/\\|}"
    done < "$REPORT_DIR/shard_index.tsv"
    echo
    echo "Skipped lanes (not runnable on this host): tests/python/gpu,"
    echo "tests/python/test_quantization_gpu.py (no CUDA); tests/python/onnx"
    echo "(broken upstream for MXNet 2.0)."
    echo
    echo "Aggregated failures/errors: \`errors.log\`. Per-shard logs: \`shards/\`."
} > "$SUMMARY"

cat "$SUMMARY"
log "Done. Report dir: $REPORT_DIR"
