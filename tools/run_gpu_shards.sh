#!/usr/bin/env bash
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
# Shard pytest targets across all visible GPUs: one pytest process per GPU,
# pinned via CUDA_VISIBLE_DEVICES (each process sees its physical GPU as
# gpu(0)). Sharding is done by collection index (tools/pytest_gpu_shard_plugin.py)
# so parametrized node ids with spaces/brackets are handled correctly.
#
# Usage:
#   tools/run_gpu_shards.sh tests/python/gpu/test_operator_gpu.py [more targets...]
#   tools/run_gpu_shards.sh [pytest args...] <targets...>
#
# Environment overrides (all optional):
#   NGPU                 number of shards / GPUs (default: auto-detect, else 1)
#   PYTHON               python interpreter (default: python3)
#   MXNET_LIBRARY_PATH   path to libmxnet.so (default: first found under the repo
#                        build dirs; unset => use the installed mxnet)
#   OUTDIR               where per-shard logs go (default: <repo>/.scratch/shards)
#   MXNET_TEST_USE_INSTALLED=1   do not prepend the repo's python/ to PYTHONPATH
#   PYTEST_EXTRA_ARGS    extra args appended to every pytest invocation
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <pytest targets...> [pytest args...]" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"

# Locate libmxnet.so unless the caller pinned one (or wants the installed pkg).
if [ -z "${MXNET_LIBRARY_PATH:-}" ]; then
  for cand in "$REPO_ROOT"/lib/libmxnet.so \
              "$REPO_ROOT"/build/libmxnet.so \
              "$REPO_ROOT"/build-g/libmxnet.so; do
    if [ -f "$cand" ]; then
      export MXNET_LIBRARY_PATH="$cand"
      break
    fi
  done
fi

# Use the in-tree python package unless told to use the installed mxnet.
if [ "${MXNET_TEST_USE_INSTALLED:-0}" != "1" ]; then
  export PYTHONPATH="$REPO_ROOT/python:${PYTHONPATH:-}"
fi
# Make the shard plugin importable by pytest's -p flag.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
# Fusion off by default for deterministic, fast op tests; override if needed.
export MXNET_USE_FUSION="${MXNET_USE_FUSION:-0}"

# Auto-detect GPU count (guard stderr so an nvidia-smi error line is not counted).
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    if [ "${n:-0}" -ge 1 ] 2>/dev/null; then
      echo "$n"
      return
    fi
  fi
  echo 1
}
NGPU="${NGPU:-$(detect_gpus)}"

# GPU_IDS: explicit, space-separated list of physical GPU ids to shard across
# (e.g. GPU_IDS="0 2 3" to skip a busy/wedged device). Defaults to 0..NGPU-1.
# When set, it overrides NGPU (the shard count becomes the number of ids).
if [ -n "${GPU_IDS:-}" ]; then
  # shellcheck disable=SC2206
  GPU_LIST=($GPU_IDS)
else
  GPU_LIST=()
  for i in $(seq 0 $((NGPU - 1))); do GPU_LIST+=("$i"); done
fi
NSHARDS="${#GPU_LIST[@]}"

# OpenMP hygiene: MXNet defaults OMP threads to the core count. With N shard
# processes each spawning that many threads, a many-core host is massively
# oversubscribed; with the default active wait policy the idle OMP threads
# busy-spin at barriers and test suites full of tiny ops (e.g. test_batchnorm)
# slow to a near-halt (observed: 24s at OMP=4 vs >300s at OMP=64 on a 64-core
# box). Cap per-shard threads to cores/NSHARDS and use a passive wait policy.
# Override by exporting OMP_NUM_THREADS / OMP_WAIT_POLICY before invoking.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  _cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  _per=$(( _cores / NSHARDS ))
  [ "$_per" -lt 1 ] && _per=1
  [ "$_per" -gt 8 ] && _per=8
  export OMP_NUM_THREADS="$_per"
fi
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"

OUTDIR="${OUTDIR:-$REPO_ROOT/.scratch/shards}"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/shard_*.log

# shellcheck disable=SC2206
TARGETS=("$@")
echo "Repo:    $REPO_ROOT"
echo "Lib:     ${MXNET_LIBRARY_PATH:-<installed mxnet>}"
echo "Python:  $PYTHON"
echo "GPUs:    ${GPU_LIST[*]}"
echo "Targets: ${TARGETS[*]}  (sharded across $NSHARDS GPU(s) by collection index)"

# SHARD_HARD_TIMEOUT (seconds): OS-level hard kill for a shard process. This is
# the only reliable backstop against a C-level deadlock that holds the GIL --
# pytest-timeout (both signal and thread methods) needs the GIL to fire, so a
# native deadlock (e.g. a data-dependent op's GPU sync) hangs forever and blocks
# the whole run via `wait`. With this set, `timeout -s KILL` force-kills the
# shard so the run completes and the shard is reported as a failure.
SHARD_HARD_TIMEOUT="${SHARD_HARD_TIMEOUT:-0}"
TO_PREFIX=()
if [ "$SHARD_HARD_TIMEOUT" -gt 0 ] 2>/dev/null; then
  TO_PREFIX=(timeout -s KILL "$SHARD_HARD_TIMEOUT")
fi

pids=()
for shard in $(seq 0 $((NSHARDS - 1))); do
  gpu="${GPU_LIST[$shard]}"
  (
    MXNET_TEST_SHARD_ID="$shard" MXNET_TEST_NUM_SHARDS="$NSHARDS" CUDA_VISIBLE_DEVICES="$gpu" \
      "${TO_PREFIX[@]}" "$PYTHON" -m pytest "${TARGETS[@]}" ${PYTEST_EXTRA_ARGS:-} \
      -q -p pytest_gpu_shard_plugin -p no:cacheprovider \
      >"$OUTDIR/shard_$shard.log" 2>&1
    rc=$?
    [ "$rc" = 137 ] && echo "SHARD HARD-TIMEOUT (SIGKILL after ${SHARD_HARD_TIMEOUT}s)" >>"$OUTDIR/shard_$shard.log"
    echo "SHARD_${shard}_EXIT=$rc" >>"$OUTDIR/shard_$shard.log"
  ) &
  pids+=($!)
done
echo "launched ${#pids[@]} shard(s) (pids: ${pids[*]})"
wait "${pids[@]}"

echo "=== shard summaries ==="
bad=0
for shard in $(seq 0 $((NSHARDS - 1))); do
  f="$OUTDIR/shard_$shard.log"
  [ -f "$f" ] || continue
  line="$(grep -E "passed|failed|error" "$f" | tail -1)"
  exitc="$(grep -oE "SHARD_${shard}_EXIT=[0-9]+" "$f" | tail -1)"
  echo "shard $shard (GPU ${GPU_LIST[$shard]}): ${line:-<no summary>}   [$exitc]"
  # Require a numeric prefix so "xfailed"/"xpassed"/"deselected" do not trip the
  # failure detector; only "<N> failed" / "<N> error[s]" count as real failures.
  echo "$line" | grep -qE "[0-9]+ (failed|error)" && bad=1
  [ "$exitc" != "SHARD_${shard}_EXIT=0" ] && bad=1
done
echo "=== OVERALL: $([ "$bad" -eq 0 ] && echo PASS || echo FAIL) ==="
exit "$bad"
