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
# One-command CUDA release: build -> acceptance test -> tag -> GitHub release.
#
# This is the Linux/CUDA release automation referenced by
# docs/cuda_wheel_build.md.  It runs on the build host (which has the GPU and the
# CUDA 13 toolkit) rather than in CI — a dedicated CUDA CI runner is deliberately
# out of scope (OI-24/OI-25).  It chains the existing single-purpose tools and
# gates each step, failing closed:
#
#   1. build    tools/build_cleanup_wheel.sh <version>   (runs the provenance gate)
#   2. test     tools/run_wheel_full_test.sh <wheel>      (full acceptance suite)
#   3. tag      git tag -a v<version>; git push <remote> v<version>
#   4. release  gh release create v<version> <wheel> ...
#
# Steps 3-4 are irreversible and outward-facing, so the script pauses for an
# explicit confirmation before them (skip with -y), and --dry-run stops cleanly
# after step 2 having mutated nothing remote.
#
# Usage:
#   tools/release_cuda_wheel.sh [options] [<version>]
#
# <version>   PEP 440 local version, e.g. 2.0.0+cu13.bw.20260615.1
#             Default: 2.0.0+cu13.bw.<UTC-today>.<next-build-number>, where the
#             build number is chosen to not collide with an existing tag.
#
# Options:
#   --dry-run      Build + test, then print the tag/release that WOULD be made.
#                  Creates no tag, pushes nothing, publishes nothing.
#   --skip-build   Reuse the newest dist/mxnet-*.whl instead of rebuilding.
#   --skip-tests   Skip the acceptance suite (the build's provenance gate still
#                  runs).  Not recommended for a real release.
#   -y, --yes      Publish without the interactive confirmation prompt.
#   -h, --help     Show this help.
#
# Environment:
#   MXNET_BUILD_JOBS   forwarded to the build (Ninja parallelism)
#   PYTHON             forwarded to the build / provenance interpreter
#   RELEASE_REMOTE     git remote to push the tag to (default: origin)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${RELEASE_REMOTE:-origin}"
DRY_RUN=0
SKIP_BUILD=0
SKIP_TESTS=0
ASSUME_YES=0
VERSION=""

die() { echo "ERROR: $*" >&2; exit 1; }

# Print the usage block (the comment lines from the title down to the code),
# skipping the license header.
show_help() {
    awk '/^# One-command CUDA release/{f=1} f&&!/^#/{exit} f{sub(/^# ?/,"");print}' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY_RUN=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        -h|--help)    show_help; exit 0 ;;
        --)           shift; break ;;
        -*)           die "unknown option: $1 (try --help)" ;;
        *)
            if [ -z "$VERSION" ]; then
                VERSION="$1"
            else
                die "unexpected extra argument: $1"
            fi
            ;;
    esac
    shift
done
[ -z "$VERSION" ] && [ $# -gt 0 ] && VERSION="$1"

# ----------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------
[ "$(uname -s)" = Linux ] || die "release_cuda_wheel.sh is Linux/CUDA only (host is $(uname -s)); use build_cleanup_wheel.sh for the macOS CPU wheel"
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"

HEAD_COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# The release must come from a clean, tagged checkout: the wheel embeds the commit
# hash and the provenance gate (Linux path, no --allow-dirty) refuses a dirty tree.
# Fail fast here so a ~1h build is not wasted on an un-releasable tree.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    die "working tree has uncommitted tracked changes; commit or stash before releasing"
fi

# Default version: 2.0.0+cu13.bw.<today>.<next build number>, where the build
# number is one past the highest already-tagged build for today (docs §5: always
# disambiguate same-day rebuilds so a new wheel never collides with a published tag).
if [ -z "$VERSION" ]; then
    today="$(date -u +%Y%m%d)"
    prefix="v2.0.0+cu13.bw.${today}"
    max=0
    while IFS= read -r tag; do
        [ -n "$tag" ] || continue
        rest="${tag#"$prefix"}"
        if [ -z "$rest" ]; then
            n=0                 # bare "...<today>" tag counts as build 0
        else
            n="${rest#.}"       # "...<today>.N" -> N
        fi
        case "$n" in ''|*[!0-9]*) continue ;; esac
        [ "$n" -gt "$max" ] && max="$n"
    done < <(git tag --list "$prefix" "${prefix}.*")
    VERSION="2.0.0+cu13.bw.${today}.$((max + 1))"
fi
TAG="v${VERSION}"

# Refuse to clobber an existing tag (local or remote).
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    die "tag $TAG already exists locally; choose a higher build number"
fi
if [ -n "$(git ls-remote --tags "$REMOTE" "refs/tags/$TAG" 2>/dev/null)" ]; then
    die "tag $TAG already exists on $REMOTE; choose a higher build number"
fi

# gh is only needed to actually publish.
if [ "$DRY_RUN" = 0 ]; then
    command -v gh >/dev/null 2>&1 || die "gh CLI not found (needed to create the release); install gh or pass --dry-run"
    gh auth status >/dev/null 2>&1 || die "gh is not authenticated (run 'gh auth login'); or pass --dry-run"
fi

echo "==> CUDA wheel release"
echo "    version : $VERSION"
echo "    tag     : $TAG"
echo "    commit  : $HEAD_COMMIT (branch $BRANCH)"
echo "    remote  : $REMOTE"
echo "    dry-run : $([ "$DRY_RUN" = 1 ] && echo yes || echo no)"

# ----------------------------------------------------------------------
# 1. Build
# ----------------------------------------------------------------------
echo "==> [1/4] Build wheel"
if [ "$SKIP_BUILD" = 1 ]; then
    echo "    --skip-build: reusing the newest dist/mxnet-*.whl"
else
    MXNET_PACKAGE_VERSION="$VERSION" tools/build_cleanup_wheel.sh "$VERSION"
fi

WHEEL="$(ls -1t dist/mxnet-*.whl 2>/dev/null | head -n1 || true)"
[ -n "$WHEEL" ] && [ -f "$WHEEL" ] || die "no wheel found in dist/ after build"
WHEEL="$(readlink -f "$WHEEL")"
echo "    wheel: $WHEEL"

# Independently re-assert the wheel matches the requested version and ships ONNX
# as a hard dependency (the build already ran this; re-running is cheap insurance
# against --skip-build pointing at a stale wheel).
"${PYTHON:-python3}" tools/release_provenance.py "$WHEEL" \
    --cmake-cache build/CMakeCache.txt \
    --package-version "$VERSION" \
    --expect-cuda on --expect-cudnn on --expect-nccl on \
    --expect-onednn on --expect-opencv on --expect-onnx on \
    || die "provenance check failed for $WHEEL — refusing to release"

# ----------------------------------------------------------------------
# 2. Acceptance test
# ----------------------------------------------------------------------
echo "==> [2/4] Acceptance test"
if [ "$SKIP_TESTS" = 1 ]; then
    echo "    --skip-tests: skipping tools/run_wheel_full_test.sh (NOT recommended for a real release)"
else
    if ! tools/run_wheel_full_test.sh "$WHEEL"; then
        die "acceptance suite reported failures (see the report dir above); refusing to release"
    fi
fi

# ----------------------------------------------------------------------
# 3-4. Tag + GitHub release (irreversible / outward-facing)
# ----------------------------------------------------------------------
TAG_MESSAGE="CUDA 13 Ampere→Blackwell wheel ($VERSION); OpenCV on, ONNX included."
RELEASE_NOTES="Ampere→Blackwell (sm_80/86/89/90/100/120+PTX), CUDA 13, OpenCV on, ONNX included.

Built from ${BRANCH}@${HEAD_COMMIT}."

if [ "$DRY_RUN" = 1 ]; then
    echo "==> --dry-run: stopping before tag/push/release. Would run:"
    echo "      git tag -a $TAG -m \"$TAG_MESSAGE\""
    echo "      git push $REMOTE $TAG"
    echo "      gh release create $TAG \"$WHEEL\" --title $TAG --notes \"...\""
    exit 0
fi

if [ "$ASSUME_YES" != 1 ]; then
    if [ ! -t 0 ]; then
        die "refusing to publish non-interactively; re-run with -y to confirm, or --dry-run"
    fi
    printf 'Tag %s, push to %s, and create the GitHub release now? [y/N] ' "$TAG" "$REMOTE"
    read -r reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) die "aborted by user (no tag, push, or release was created)" ;;
    esac
fi

echo "==> [3/4] Tag + push"
git tag -a "$TAG" -m "$TAG_MESSAGE"
git push "$REMOTE" "$TAG"

echo "==> [4/4] GitHub release"
gh release create "$TAG" "$WHEEL" \
    --title "$TAG" \
    --notes "$RELEASE_NOTES"

echo
echo "==> Released $TAG"
gh release view "$TAG" --json url -q .url 2>/dev/null || true
echo "    Downstreams (e.g. d2l) can now bump with tools/update_mxnet_wheel.py --source github"
