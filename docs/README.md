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

# Proposal: generating MXNet 2.0 API documentation

> **Status: recommendation only — not yet implemented.** This section describes a
> sensible, low-maintenance way to publish API docs for the fork. Nothing under
> `docs/api/` exists yet; the layout and config below are the proposed starting
> point for a follow-up change.

## Why not just fix the old site

Reviving `docs/python_docs` means simultaneously: upgrading from Sphinx 1.5 to 7+
(rewriting `conf.py` for the removed APIs), replacing the unmaintained
`recommonmark` with `myst-parser`, un-vendoring or porting `mxtheme`, wiring
`breathe`/Doxygen for C++, and re-running `nbsphinx` tutorial evaluation — which
needs a GPU and a working data pipeline. That is a large, fragile effort whose
main payload (executed tutorials, the apache-branded theme) is not what a fork
needs. A clean, minimal API reference is far cheaper to stand up and to keep
green, and it can grow later.

## Recommended approach

Stand up a **new, minimal Sphinx API reference** under `docs/api/`, built from the
**installed** `mxnet` package (the CPU wheel is enough — no GPU needed to document
the Python API), using only maintained tooling:

- **Sphinx 7+** with `sphinx.ext.autodoc` + `sphinx.ext.autosummary` (with
  `:recursive:`) to crawl the package, `sphinx.ext.napoleon` for the NumPy/Google
  docstring style MXNet uses, `sphinx.ext.intersphinx` (link to numpy/python), and
  `sphinx.ext.viewcode`.
- **`myst-parser`** for Markdown pages (the modern replacement for `recommonmark`),
  so narrative pages can be authored in Markdown alongside the autosummary API.
- **`furo`** theme (or `pydata-sphinx-theme`) — actively maintained, no vendoring.
- **No tutorial execution** for v1 (`nbsphinx` omitted). **No C++** for v1 (add
  Doxygen + `breathe` later as a separate `docs/api/cpp` target).

### Proposed layout

```
docs/api/
├── conf.py             # the skeleton below
├── index.md            # landing page (MyST Markdown)
├── requirements.txt    # pinned doc deps
├── Makefile            # `make html`
└── _templates/         # autosummary recursive templates (optional)
```

`docs/api/requirements.txt`:

```
sphinx>=7,<9
furo
myst-parser
sphinx-autodoc-typehints
```

`docs/api/conf.py` skeleton:

```python
import mxnet  # documented from the installed package

project = "MXNet 2.0 (smolix fork)"
author = "smolix/mxnet contributors"
release = mxnet.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True          # build stub pages by crawling the package
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
# If any optional import is missing in the docs venv, mock it instead of failing:
autodoc_mock_imports = []            # e.g. ["onnx"] if you document the ONNX bridge

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
```

`docs/api/index.md` (drive the recursive crawl from a single autosummary):

````markdown
# MXNet 2.0 API reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   mxnet.np
   mxnet.npx
   mxnet.gluon
   mxnet.optimizer
   mxnet.io
   mxnet.image
   mxnet.autograd
   mxnet.runtime
```
````

### Build it

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python \
  dist/mxnet-*.whl -r docs/api/requirements.txt
.venv-docs/bin/python -m sphinx -b html docs/api docs/api/_build/html
# open docs/api/_build/html/index.html
```

### Publish (GitHub Pages) — sketch

A single workflow installs the CPU wheel + doc deps, runs `sphinx-build`, and
deploys to the `gh-pages` branch:

```yaml
# .github/workflows/docs.yml
name: docs
on: { push: { branches: [master] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install dist/mxnet-*.whl -r docs/api/requirements.txt   # or the latest release wheel
      - run: python -m sphinx -b html docs/api docs/api/_build/html
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/api/_build/html
```

### Later add-ons (not in v1)

- **C++ API**: Doxygen + `breathe` as a separate target (`docs/api/cpp`).
- **Tutorials**: port a handful of the legacy `python_docs/.../tutorials` notebooks
  to MyST Markdown and include them *without* execution (`nbsphinx_execute='never'`)
  until a CI GPU runner exists.
- **Versioned docs**: `sphinx-multiversion` once there is more than one doc-bearing
  release.
