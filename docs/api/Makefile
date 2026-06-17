# Minimal Makefile for the MXNet 2.0 API reference.
#
# Requires an importable `mxnet` plus docs/api/requirements.txt in the active
# environment, e.g.:
#   uv venv .venv-docs --python 3.12
#   uv pip install --python .venv-docs/bin/python dist/mxnet-*.whl -r docs/api/requirements.txt
#   .venv-docs/bin/python -m make ... (or activate the venv and run `make html`)

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?=
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html linkcheck clean serve deploy

help:
	@echo "make html      - build the HTML API reference into $(BUILDDIR)/html"
	@echo "make linkcheck - validate links/cross-references in the API reference"
	@echo "make deploy    - build, then publish to the gh-pages branch (GitHub Pages)"
	@echo "make clean     - remove $(BUILDDIR) and autosummary stubs (generated/)"
	@echo "make serve     - build, then serve at http://localhost:8000"

html:
	$(SPHINXBUILD) -b html -d "$(BUILDDIR)/doctrees" $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@touch "$(BUILDDIR)/html/.nojekyll"
	@echo "Built: $(BUILDDIR)/html/index.html"

linkcheck:
	$(SPHINXBUILD) -b linkcheck -d "$(BUILDDIR)/doctrees" $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

clean:
	rm -rf "$(BUILDDIR)" generated

serve: html
	@cd "$(BUILDDIR)/html" && python -m http.server

deploy: html
	@bash deploy.sh
