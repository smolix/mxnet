#!/usr/bin/env bash
#
# Publish the built API reference (docs/api/_build/html) to the `gh-pages`
# branch of this repo's `origin` remote. GitHub Pages then serves it at
# https://smolix.github.io/mxnet/ .
#
# Build first:  (cd docs/api && make html)   -- or just run `make deploy`,
# which builds and then calls this script.
#
# This is a force-deploy: gh-pages is a generated branch (one commit, no
# history) and is overwritten each time. Do not hand-edit it.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
site="$here/_build/html"
[ -f "$site/index.html" ] || { echo "No build at $site — run 'make html' first." >&2; exit 1; }

# Required so GitHub Pages serves the underscore dirs (_static/_images/...).
touch "$site/.nojekyll"

remote="$(git -C "$here" remote get-url origin)"
email="$(git -C "$here" config user.email || true)"; email="${email:-noreply@users.noreply.github.com}"
name="$(git -C "$here" config user.name || true)";  name="${name:-docs-bot}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
rsync -a --delete "$site/" "$tmp/"
git -C "$tmp" init -q -b gh-pages 2>/dev/null || { git -C "$tmp" init -q && git -C "$tmp" checkout -q -b gh-pages; }
git -C "$tmp" add -A
git -C "$tmp" -c user.email="$email" -c user.name="$name" commit -q -m "Deploy MXNet 2.0 API docs"
git -C "$tmp" push --force "$remote" gh-pages
echo "Deployed gh-pages -> https://smolix.github.io/mxnet/"
