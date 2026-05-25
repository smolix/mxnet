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
# FP16 / BF16 smoke test for a remote x86_64 + Ampere host.
#
# Designed for a target with:
#   - AMD Zen 4+ CPU (Ryzen 5 7000-series or newer) so that /proc/cpuinfo
#     reports avx512_bf16 and oneDNN can exercise native BF16 paths.
#     The current dev host is a Zen 2 7002 EPYC which lacks avx512_bf16, so
#     test_bf16_operator.py auto-skips there; this script's job is to confirm
#     that on a newer CPU those skips turn into passing runs.
#   - NVIDIA Ampere GPU (sm_80/86/89), e.g. RTX 3070 (sm_86).
#   - >= 32 GB RAM (this box has 64 GB).
#   - CUDA 13.x toolkit, cuDNN and NCCL via pip (nvidia-*-cu13 wheels), or
#     a system CUDA install.  We let pip-installed cuDNN/NCCL handle the
#     runtime side; the build only needs nvcc + CUDA headers from the system
#     toolkit.
#
# Output: $REPORT_DIR contains
#   env.txt         — host snapshot (CPU, GPU, libs, glibc, kernel)
#   features.txt    — mxnet runtime feature flags
#   configure.log   — cmake configure log
#   build.log       — full ninja build log (tail of)
#   build.summary   — short summary of any ERRORs
#   tests/<id>.log  — per-test-shard pytest output
#   summary.md      — combined human-readable summary
#
# Usage on the remote box:
#   bash run_fp16_remote_smoke.sh                  # full run
#   bash run_fp16_remote_smoke.sh --skip-build     # reuse existing build/
#   bash run_fp16_remote_smoke.sh --gpu-only       # only GPU FP16 tests
#   bash run_fp16_remote_smoke.sh --cpu-only       # only oneDNN BF16 tests
#   bash run_fp16_remote_smoke.sh --branch master  # alternate branch (default cleanup/p0-p1-p2-20260522)
#
set -uo pipefail

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/smolix/mxnet.git}"
BRANCH="${BRANCH:-cleanup/p0-p1-p2-20260522}"
SRC_DIR="${SRC_DIR:-$HOME/mxnet-fp16-test}"
REPORT_DIR="${REPORT_DIR:-$SRC_DIR/report-$(date -u +%Y%m%dT%H%M%SZ)}"
PARALLEL_BUILD_JOBS="${PARALLEL_BUILD_JOBS:-4}"   # 6-core Ryzen: leave 2 for the OS
PARALLEL_TEST_JOBS="${PARALLEL_TEST_JOBS:-2}"
TIMEOUT_BUILD_MIN="${TIMEOUT_BUILD_MIN:-90}"
TIMEOUT_TEST_MIN="${TIMEOUT_TEST_MIN:-30}"

# CUDA arch override: 8.6 is Ampere consumer (RTX 30xx).  Add 8.0 if you want
# A100 coverage from the same build; the wheel build uses 8.0;8.6;8.9;9.0;10.0;12.0+PTX.
CUDA_ARCH="${CUDA_ARCH:-8.6}"

# --skip-build / --gpu-only / --cpu-only / --branch <name>
SKIP_BUILD=0
GPU_ONLY=0
CPU_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --gpu-only) GPU_ONLY=1; shift ;;
        --cpu-only) CPU_ONLY=1; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$REPORT_DIR/run.log"; }
section() { log; log "=== $* ==="; }

run_capture() {
    # run_capture <logfile> <description> -- <command...>
    local logf="$1"; shift
    local desc="$1"; shift
    [ "$1" = "--" ] && shift
    log "RUN: $desc"
    log "     cmd: $*"
    log "     log: $logf"
    {
        echo "### $desc"
        echo "### cmd: $*"
        echo "### started: $(date -u)"
        echo
    } >>"$logf"
    "$@" >>"$logf" 2>&1
    local rc=$?
    {
        echo
        echo "### finished: $(date -u)   rc=$rc"
    } >>"$logf"
    log "     rc=$rc"
    return $rc
}

# ----------------------------------------------------------------------
# Stage 0: prerequisites + report dir
# ----------------------------------------------------------------------
mkdir -p "$REPORT_DIR/tests"
: >"$REPORT_DIR/run.log"

section "Preflight"

missing=()
for cmd in git cmake ninja nvcc nvidia-smi python3 patchelf; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if [ ${#missing[@]} -gt 0 ]; then
    log "Missing required tools: ${missing[*]}"
    log "Install on Ubuntu/Debian with e.g.:"
    log "  sudo apt-get install -y git cmake ninja-build gcc g++ patchelf python3 python3-venv python3-pip"
    log "  # plus CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
    exit 2
fi

# Host environment snapshot
{
    echo "## Host snapshot"
    echo
    echo "Date (UTC): $(date -u)"
    echo "Hostname:   $(hostname)"
    echo "Kernel:     $(uname -srm)"
    echo
    echo "## CPU"
    lscpu | grep -E 'Model name|Vendor ID|CPU\(s\)|Architecture|Flags' | head
    echo
    echo "### Critical CPU feature flags (look for avx512_bf16)"
    grep -m1 -oE '(avx512_bf16|avx512f|avx512vnni|amx_bf16|amx_tile|amx_int8)' /proc/cpuinfo | sort -u
    echo
    echo "## RAM"
    free -h | head -2
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    echo
    echo "## CUDA toolkit"
    nvcc --version | tail -2
    echo
    echo "## Python"
    python3 --version
    python3 -c 'import sys; print(sys.executable)'
    echo
    echo "## glibc"
    ldd --version | head -1
} >"$REPORT_DIR/env.txt"

cat "$REPORT_DIR/env.txt" | tee -a "$REPORT_DIR/run.log"

# Sanity gate: BF16 testing is meaningful only if the CPU has avx512_bf16
if ! grep -q avx512_bf16 /proc/cpuinfo; then
    log
    log "WARNING: /proc/cpuinfo has no avx512_bf16 flag."
    log "         oneDNN BF16 paths will run via fp32 emulation here, just like"
    log "         on the Zen 2 EPYC dev host, so the FP16-on-modern-CPU smoke is"
    log "         degenerate.  Continuing anyway (FP16 GPU tests are still useful)."
fi

# ----------------------------------------------------------------------
# Stage 1: clone + checkout
# ----------------------------------------------------------------------
section "Clone + checkout $BRANCH"

if [ ! -d "$SRC_DIR/.git" ]; then
    git clone --recurse-submodules --shallow-submodules --depth 1 -b "$BRANCH" "$REPO_URL" "$SRC_DIR" \
        2>&1 | tee -a "$REPORT_DIR/run.log"
else
    log "Reusing existing clone at $SRC_DIR"
    (cd "$SRC_DIR" && git fetch --depth 1 origin "$BRANCH" && git checkout "$BRANCH" && \
        git reset --hard "origin/$BRANCH" && git submodule update --init --recursive --depth 1) \
        2>&1 | tee -a "$REPORT_DIR/run.log"
fi
cd "$SRC_DIR"
log "HEAD: $(git rev-parse HEAD)"
log "  msg: $(git log -1 --format='%s')"

# ----------------------------------------------------------------------
# Stage 2: Python venv + build deps
# ----------------------------------------------------------------------
section "Python venv"

if [ ! -d .venv-fp16 ]; then
    python3 -m venv .venv-fp16
fi
# shellcheck disable=SC1091
source .venv-fp16/bin/activate
python -m pip install --upgrade pip wheel setuptools build packaging numpy requests graphviz pytest pytest-timeout pytest-xdist
# Pre-install pip-side CUDA runtime libraries that libmxnet.so RUNPATH expects.
python -m pip install --upgrade 'nvidia-cudnn-cu13>=9.22,<10' 'nvidia-nccl-cu13>=2.28,<3' || \
    log "WARNING: failed to install nvidia-cudnn-cu13/nccl-cu13; CUDA imports may fail at runtime"

# ----------------------------------------------------------------------
# Stage 3: CMake configure + build
# ----------------------------------------------------------------------
if [ $SKIP_BUILD -eq 0 ]; then
    section "Configure"
    rm -rf build
    mkdir build
    run_capture "$REPORT_DIR/configure.log" "cmake configure (CUDA $CUDA_ARCH, oneDNN, OpenCV off, NCCL on)" -- \
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DUSE_CUDA=ON \
            -DUSE_CUDNN=ON \
            -DUSE_NCCL=ON \
            -DUSE_ONEDNN=ON \
            -DUSE_OPENMP=ON \
            -DUSE_OPENCV=OFF \
            -DUSE_BLAS=open \
            -DUSE_DIST_KVSTORE=OFF \
            -DUSE_OPERATOR_TUNING=OFF \
            -DMXNET_CUDA_ARCH="$CUDA_ARCH" \
            -DBUILD_CPP_EXAMPLES=OFF
    rc=$?
    if [ $rc -ne 0 ]; then
        log "Configure failed; see $REPORT_DIR/configure.log"
        exit 3
    fi

    section "Build (timeout ${TIMEOUT_BUILD_MIN}m, -j $PARALLEL_BUILD_JOBS)"
    run_capture "$REPORT_DIR/build.log" "cmake build mxnet" -- \
        timeout "${TIMEOUT_BUILD_MIN}m" cmake --build build --target mxnet -j "$PARALLEL_BUILD_JOBS"
    rc=$?
    grep -E '(error:|warning:|undefined reference)' "$REPORT_DIR/build.log" | tail -200 > "$REPORT_DIR/build.summary"
    if [ $rc -ne 0 ]; then
        log "Build failed; see $REPORT_DIR/build.log + $REPORT_DIR/build.summary"
        exit 4
    fi

    section "Stage libmxnet.so into python/mxnet/"
    cp -v build/libmxnet.so python/mxnet/libmxnet.so
fi

# ----------------------------------------------------------------------
# Stage 4: install package, capture runtime features
# ----------------------------------------------------------------------
section "Install python package (editable)"

python -m pip install -e ./python --no-build-isolation 2>&1 | tee -a "$REPORT_DIR/run.log"

section "Runtime features"
python - <<'PY' 2>&1 | tee "$REPORT_DIR/features.txt"
import mxnet as mx
import json
print("mx.__version__:", mx.__version__)
print("native commit  :", mx.runtime.feature_list().__class__)
feats = mx.runtime.Features()
print(feats)
print()
print("Boolean flags:")
for f in ['CUDA', 'CUDNN', 'NCCL', 'ONEDNN', 'OPENMP', 'OPENCV', 'CXX14', 'F16C']:
    try:
        print(f"  {f}: {feats.is_enabled(f)}")
    except Exception as exc:
        print(f"  {f}: ERR {exc!r}")
print()
print("GPU count:", mx.device.num_gpus())
if mx.device.num_gpus() > 0:
    import mxnet.ndarray as nd
    x = nd.ones((4, 4), ctx=mx.gpu(0), dtype='float16')
    y = nd.dot(x, x)
    print("fp16 sm-up GEMM on GPU(0):", y.sum().asscalar())
PY

# ----------------------------------------------------------------------
# Stage 5: run focused tests
# ----------------------------------------------------------------------
section "Run focused FP16 / BF16 tests"

# Cap BLAS threads inside Python so xdist lanes don't oversubscribe a 6-core box.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Disable noisy faulthandler so backtraces don't drown progress on this slow box.
export PYTEST_ADDOPTS="-p no:faulthandler --tb=short --color=no --durations=20"

run_one() {
    # run_one <shard-id> <description> -- <pytest args>
    local id="$1"; shift
    local desc="$1"; shift
    [ "$1" = "--" ] && shift
    local logf="$REPORT_DIR/tests/${id}.log"
    run_capture "$logf" "pytest $desc" -- \
        timeout "${TIMEOUT_TEST_MIN}m" python -m pytest "$@"
}

# --- CPU / oneDNN BF16 lane ----------------------------------------------
if [ $GPU_ONLY -eq 0 ]; then
    # tests/python/dnnl/test_bf16_operator.py — the file that auto-skips on
    # Zen 2.  On a Zen 4+ CPU with avx512_bf16 we expect these to run.
    run_one bf16_operator "DNNL BF16 operator suite" -- \
        tests/python/dnnl/test_bf16_operator.py -v
    # tests/python/dnnl/test_amp.py — BF16 AMP coverage (subset of suite).
    run_one bf16_amp "DNNL AMP BF16" -- \
        tests/python/dnnl/test_amp.py -v
    # tests/python/dnnl/test_dnnl.py — broad oneDNN smoke (catches BF16
    # fallback regressions in conv/fc/transformer subgraph paths fixed under
    # XOP19).
    run_one dnnl_smoke "DNNL smoke" -- \
        tests/python/dnnl/test_dnnl.py -v
    # tests/python/dnnl/subgraphs/test_matmul_subgraph.py — INT8 matmul +
    # self-attention; same code path that BF16 fallback exercises when
    # avx512_bf16 is unavailable.
    run_one dnnl_matmul "DNNL INT8 matmul subgraph" -- \
        tests/python/dnnl/subgraphs/test_matmul_subgraph.py -v
    # tests/python/dnnl/test_batch_dot_attention_regression.py — covers the
    # transformer batch-dot reorder fix.
    run_one dnnl_batchdot "DNNL batch-dot attention regression" -- \
        tests/python/dnnl/test_batch_dot_attention_regression.py -v
    # XOP19 focused test (request-aware oneDNN paths).
    if [ -f tests/python/dnnl/test_xop19_onednn_req.py ]; then
        run_one xop19 "XOP19 oneDNN req/copyback regressions" -- \
            tests/python/dnnl/test_xop19_onednn_req.py -v
    fi
fi

# --- GPU FP16 lane --------------------------------------------------------
if [ $CPU_ONLY -eq 0 ] && [ "$(nvidia-smi -L | wc -l)" -gt 0 ]; then
    run_one gpu_amp "GPU AMP / fp16 cast hygiene" -- \
        tests/python/gpu/test_amp.py -v
    if [ -f tests/python/gpu/test_amp_init.py ]; then
        run_one gpu_amp_init "GPU AMP init" -- \
            tests/python/gpu/test_amp_init.py -v
    fi
    if [ -f tests/python/gpu/test_fp16_batch_dot.py ]; then
        run_one gpu_fp16_batchdot "GPU fp16 batch-dot parity" -- \
            tests/python/gpu/test_fp16_batch_dot.py -v
    fi
    if [ -f tests/python/gpu/test_cublaslt_fc.py ]; then
        run_one gpu_cublaslt "cuBLASLt FC / dtype matrix" -- \
            tests/python/gpu/test_cublaslt_fc.py -v
    fi
    if [ -f tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py ]; then
        run_one gpu_cudnn_fallback "cuDNN frontend no-plan fallback" -- \
            tests/python/gpu/test_cudnn_frontend_no_plan_fallback.py -v
    fi
    if [ -f tests/python/gpu/test_deconv_tf32.py ]; then
        run_one gpu_tf32_deconv "TF32 deconv parity" -- \
            tests/python/gpu/test_deconv_tf32.py -v
    fi
    if [ -f tests/python/gpu/test_reducer_regressions.py ]; then
        run_one gpu_reducer "GPU reducer regressions" -- \
            tests/python/gpu/test_reducer_regressions.py -v
    fi
fi

# --- Generic FP16 reductions / norm coverage (CPU + GPU) ------------------
run_one fp16_layer_norm "LayerNorm large-reduction fp16 (XOP4 fix)" -- \
    tests/python/unittest/test_operator.py -v -k 'test_layer_norm' -n "$PARALLEL_TEST_JOBS"
run_one fp16_group_norm "GroupNorm large-reduction fp16" -- \
    tests/python/unittest/test_operator.py -v -k 'test_group_norm' -n "$PARALLEL_TEST_JOBS"

# ----------------------------------------------------------------------
# Stage 6: summary
# ----------------------------------------------------------------------
section "Summarize"

{
    echo "# MXNet FP16 / BF16 smoke report"
    echo
    echo "Host: $(hostname)   $(date -u)"
    echo "Repo: $REPO_URL@$BRANCH"
    echo "Commit: $(git rev-parse HEAD)"
    echo
    echo "## Test shard outcomes"
    echo
    printf '| Shard | Result | Passed / Failed / Skipped / Errors |\n'
    printf '|---|---|---|\n'
    for f in "$REPORT_DIR"/tests/*.log; do
        [ -f "$f" ] || continue
        id=$(basename "$f" .log)
        # Pytest summary line patterns: "===== N passed, M failed, K skipped in Xs ====="
        last=$(grep -E '=+ .*(passed|failed|error|skipped|xfailed|xpassed).* in [0-9.]+s' "$f" | tail -1)
        if [ -z "$last" ]; then
            rc=$(grep -oE 'rc=[0-9-]+' "$f" | tail -1)
            printf '| %s | INCOMPLETE | %s |\n' "$id" "$rc"
        else
            printf '| %s | OK | %s |\n' "$id" "${last//|/\\|}"
        fi
    done
    echo
    echo "## env.txt excerpt"
    sed -n '1,40p' "$REPORT_DIR/env.txt"
    echo
    echo "## features.txt excerpt"
    sed -n '1,20p' "$REPORT_DIR/features.txt"
} > "$REPORT_DIR/summary.md"

log
log "Done. Report directory: $REPORT_DIR"
log "  summary.md       = combined Markdown report"
log "  env.txt          = host snapshot"
log "  features.txt     = mxnet runtime features"
log "  configure.log    = cmake configure"
log "  build.log        = full build"
log "  build.summary    = build errors/warnings only"
log "  tests/<id>.log   = per-shard pytest output"
log
log "Bundle the report:  tar czf mxnet-fp16-report-$(hostname)-$(date -u +%Y%m%d).tar.gz -C $REPORT_DIR ."
