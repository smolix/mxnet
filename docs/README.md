<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Documentation

This is the documentation index for the **`smolix/mxnet`** fork (MXNet 2.0,
CUDA 13 / Blackwell + Apple Silicon). For most users the Markdown docs at the
repo root are the place to start; the directories under `docs/` are a mix of
fork-maintained build docs and **legacy upstream documentation kept for
historical reference only**.

## Current, fork-maintained docs

| Doc | What it covers |
|-----|----------------|
| [`../README.md`](../README.md) | Overview, install, system requirements, troubleshooting |
| [`../FIXED.md`](../FIXED.md) | Everything this fork changed vs upstream |
| [`../OPEN_ISSUES.md`](../OPEN_ISSUES.md) + [`details`](../OPEN_ISSUES_DETAILS.md) | Known limitations and open work |
| [`../BUILDING.md`](../BUILDING.md) | Build from source (Linux/CUDA and Apple Silicon) |
| [`cuda_wheel_build.md`](cuda_wheel_build.md) | Authoritative, provenance-gated release-wheel pipeline |
| [`api/`](api/) | Sphinx **API reference**, generated from the installed package's docstrings |

## Legacy upstream documentation (historical — not built here)

These predate the fork, target `apache/mxnet`, and are **not currently buildable
on a modern toolchain**. They are retained for reference and for anyone who wants
to revive them:

- `python_docs/` — the old Sphinx site (Python API + tutorials). Its
  `python/scripts/conf.py` declares `needs_sphinx = '1.5.6'`, uses Sphinx APIs
  removed in 2.x (`app.add_javascript` / `add_stylesheet`), the unmaintained
  `recommonmark` Markdown bridge, the vendored `mxtheme`, and `breathe` +
  `nbsphinx` tutorial evaluation. It will not build as-is.
- `cpp_docs/` — Doxygen config for the C++ API.
- `static_site/` — the Jekyll site that produced `mxnet.apache.org` (archived).

---

# API reference (`docs/api/`)

The fork ships a **minimal, self-contained Sphinx API reference** under
[`api/`](api/). It is generated entirely from the docstrings in the installed
`mxnet` package — there is no hand-written narrative and no tutorials. This is the
low-maintenance way to give users *some* real documentation (the actual framework
API, extracted from the code) without resurrecting the full website.

## Why a fresh reference instead of reviving the old site

Reviving `python_docs/` would mean upgrading Sphinx 1.5 → 8 (rewriting `conf.py`
for removed APIs), replacing `recommonmark` with `myst-parser`, porting the
vendored `mxtheme`, wiring `breathe`/Doxygen, and re-running `nbsphinx` tutorial
evaluation (which needs a GPU and a data pipeline). That is a large, fragile effort
whose main payload — executed tutorials and the apache-branded theme — is not what
a fork needs. A clean autosummary reference is far cheaper to stand up and keep
green.

## What's here

```
docs/api/
├── conf.py                      # Sphinx config — documents the *installed* mxnet
├── index.rst                    # landing page + the autosummary module list
├── requirements.txt             # pinned doc toolchain
├── Makefile                     # `make html` / `make serve` / `make clean`
├── _templates/autosummary/      # recursive-autosummary templates (module + class)
└── .gitignore                   # ignores _build/ and the generated/ stubs
```

Tooling: Sphinx + `autosummary` (`:recursive:`) + `napoleon` (MXNet uses NumPy-style
docstrings) + `intersphinx` + `viewcode` + `myst-parser` + `sphinx-autodoc-typehints`,
with the maintained **furo** theme. Documented namespaces: `mxnet.numpy` (`np`),
`mxnet.numpy_extension` (`npx`), `ndarray`, `symbol`, `gluon` (recursive:
nn/rnn/loss/data/metric/model_zoo), `optimizer`, `lr_scheduler`, `io`, `image`,
`autograd`, `kvstore`, `device`, `profiler`, `runtime`, `contrib`.

## The one requirement: an importable `mxnet`

MXNet registers most of its operator surface dynamically from the C++ backend at
import time, so `conf.py` does `import mxnet` and the build host must have a
**working** mxnet:

- A **CPU wheel** (the macOS wheel, or a `USE_CUDA=OFF` Linux build) is enough and
  documents the full CPU operator set — no GPU needed.
- A **CUDA wheel imports only where its CUDA libraries load.** On a Linux GPU host
  it additionally surfaces GPU-only operators; on a GPU-less runner `import mxnet`
  fails, so use a CPU wheel there.

## Build it

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python dist/mxnet-*.whl -r docs/api/requirements.txt
.venv-docs/bin/sphinx-build -b html docs/api docs/api/_build/html
# open docs/api/_build/html/index.html      (or:  cd docs/api && make html)
```

A clean build emits a few hundred warnings and **succeeds** (do **not** pass `-W`).
Almost all warnings are MXNet's own auto-generated operator docstrings not being
valid reStructuredText, plus ambiguous cross-references for names that exist in
several namespaces at once (`reshape` is in `numpy`, `ndarray`, and `symbol`). They
are non-fatal; the dominant `ref.python` category is suppressed in `conf.py`. The
build produces ~200 API pages under `_build/html/generated/`.

## Deploying on a Linux + GPU box

On the GPU host, install the CUDA wheel and build the same way — the import then
documents the GPU paths too:

```bash
pip install <the cu13 release wheel> -r docs/api/requirements.txt
sphinx-build -b html docs/api docs/api/_build/html
```

Serve `docs/api/_build/html/` with any static web server (`python -m http.server`,
nginx, …). To publish via GitHub Pages instead, a minimal workflow (CPU wheel, no
GPU runner needed):

```yaml
# .github/workflows/docs.yml  (optional)
name: docs
on: { push: { branches: [master] } }
permissions: { contents: write }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install <cpu-wheel-url> -r docs/api/requirements.txt
      - run: sphinx-build -b html docs/api _site
      - uses: peaceiris/actions-gh-pages@v4
        with: { github_token: ${{ secrets.GITHUB_TOKEN }}, publish_dir: _site }
```

## Later add-ons

- **C++ API** via Doxygen + `breathe`, as a separate target.
- **Versioned docs** with `sphinx-multiversion` once more than one doc-bearing
  release exists.
