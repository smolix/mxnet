#!/usr/bin/env bash
#
# Build the cleaned-up Ampere-through-Blackwell Linux/CUDA wheel.
#
# Re-uses the already-configured build/ directory (CUDA on, oneDNN on,
# OpenCV off, NCCL off, sm_80/86/89/90/100/120+PTX).  Stages libmxnet.so
# into python/mxnet/ and invokes setup.py with MXNET_PACKAGE_VERSION so the
# wheel metadata matches the intended tag.  Runs release_provenance.py at
# the end and exits non-zero on any failure.
#
# Usage:
#   tools/build_cleanup_wheel.sh [<version>]
#
# Default version derives from today: 2.0.0+cu13.bw.YYYYMMDD.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_VERSION="2.0.0+cu13.bw.$(date -u +%Y%m%d)"
VERSION="${1:-${MXNET_PACKAGE_VERSION:-$DEFAULT_VERSION}}"

echo "==> Repo: $REPO_ROOT"
echo "==> Version: $VERSION"

if [ ! -f build/libmxnet.so ]; then
    echo "build/libmxnet.so missing — run 'cmake --build build --target mxnet -j' first" >&2
    exit 2
fi

echo "==> Staging libmxnet.so into python/mxnet/"
cp -v build/libmxnet.so python/mxnet/libmxnet.so

echo "==> Building wheel"
rm -rf dist build_wheel
mkdir -p dist
(cd python && \
    MXNET_PACKAGE_VERSION="$VERSION" \
    MXNET_SETUP_EXCLUDE_ONNX=1 \
    MXNET_SETUP_ENABLE_OPENCV_DEPS=0 \
    MXNET_SETUP_ENABLE_CUDA_DEPS=1 \
    python -m build --wheel --outdir ../dist)

WHEEL=$(ls -1 dist/*.whl | head -n1)
if [ -z "$WHEEL" ]; then
    echo "No wheel produced" >&2
    exit 3
fi
echo "==> Built: $WHEEL"
ls -lh "$WHEEL"

echo "==> Validating provenance"
python tools/release_provenance.py "$WHEEL" \
    --cmake-cache build/CMakeCache.txt \
    --package-version "$VERSION" \
    --expect-cuda on \
    --expect-opencv off

echo "==> Wheel build OK: $WHEEL"
