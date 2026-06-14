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
# Sphinx configuration for the MXNet 2.0 (smolix fork) API reference.
#
# This documents the *installed* mxnet package. MXNet registers most of its
# operator surface dynamically from the C++ backend at import time, so the docs
# build must be able to `import mxnet`: a CPU wheel is enough; a CUDA wheel needs
# its CUDA libraries present on the build host.

import mxnet

# -- Project information ------------------------------------------------------
project = "MXNet 2.0 (smolix fork)"
author = "smolix/mxnet contributors"
copyright = f"2026, {author}"
release = mxnet.__version__
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
root_doc = "index"

# -- autodoc / autosummary ----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
# MXNet docstrings are NumPy-style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Some submodules import optional third-party packages lazily; mock them so the
# build doesn't fail when they're absent from the docs environment.
autodoc_mock_imports = [
    "onnx",
    "onnxruntime",
    "tensorrt",
    "horovod",
    "gluoncv",
    "tvm",
]

# MXNet's C-extension signatures and uneven docstrings produce many warnings;
# keep the build resilient rather than failing on them (do not pass -W).
# The dominant category is "more than one target found for cross-reference":
# the same operator name (reshape, zeros, ...) exists in mxnet.numpy,
# mxnet.ndarray and mxnet.symbol at once, so docstring back-references are
# ambiguous-but-harmless. Suppress that (ref.python) plus autosummary stub noise.
nitpicky = False
suppress_warnings = ["autosummary", "ref.python"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = f"MXNet {release}"
html_short_title = "MXNet API"
html_static_path = ["_static"]


# -- Tidy machine-generated docstrings ----------------------------------------
# MXNet builds most operator docstrings at import time from C++ metadata, and the
# result is not always valid reStructuredText. Normalize each docstring before
# Sphinx/napoleon parse it. This is a docs-build-time transform only; it does not
# modify the installed package. (The complementary source fix lives in
# python/mxnet/base.py:build_param_doc, which re-indents multi-line argument
# descriptions so they parse cleanly.)
import re as _re
import textwrap as _textwrap

_SECTION_RE = _re.compile(
    r"\n(Parameters|Returns|Yields|Raises|Examples|Notes|References|See Also|"
    r"Attributes|Warnings|Other Parameters)\n([-=~^]+)\n"
)


def _tidy_docstring(app, what, name, obj, options, lines):
    if not lines:
        return
    text = _textwrap.dedent("\n".join(lines))
    # Guarantee a blank line before every numpydoc section header.
    text = _SECTION_RE.sub(lambda m: f"\n\n{m.group(1)}\n{m.group(2)}\n", text)
    # Pad too-short title/section underlines to the length of the title above
    # (matches how docutils already interprets them; just silences the warning).
    out, prev = [], ""
    for line in text.split("\n"):
        s = line.strip()
        if s and len(set(s)) == 1 and s[0] in "=-~^\"'`*+#" and 0 < len(s) < len(prev.strip()):
            indent = prev[: len(prev) - len(prev.lstrip())]
            line = indent + s[0] * len(prev.strip())
        out.append(line)
        prev = line
    lines[:] = out


def setup(app):
    # priority < 500 so this runs before napoleon converts the numpydoc sections.
    app.connect("autodoc-process-docstring", _tidy_docstring, priority=400)

