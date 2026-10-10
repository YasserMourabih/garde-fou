VENV ?= .venv
PY   := $(VENV)/bin/python
GF   := $(VENV)/bin/garde-fou

# Stock macOS ships `python3` as 3.9, which this package does not support, so
# resolve the newest interpreter that actually satisfies requires-python rather
# than trusting the name. Override explicitly if you want a specific one:
#   make install PYTHON=/opt/homebrew/bin/python3.12
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3.10 python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue; \
	command -v $$p; break; \
	done)

.PHONY: help install data train cascade eval bench report all serve test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install the package with every extra
	@if [ -z "$(PYTHON)" ]; then \
		echo "No Python >= 3.10 found on PATH (this project needs one)."; \
		echo "  macOS:  brew install python@3.12"; \
		echo "  or:     make install PYTHON=/path/to/python3.12"; \
		exit 1; \
	fi
	@if [ -x "$(PY)" ] && $(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then \
		echo "reusing $(VENV) (`$(PY) -V`)"; \
	else \
		echo "creating $(VENV) with $(PYTHON) (`$(PYTHON) -V`)"; \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[neural,serve,dev]"

data:     ## download the public corpora and build the evaluation matrix
	$(GF) data build

train:    ## fit every detector
	$(GF) train all

cascade:  ## assemble and calibrate the production cascade
	$(GF) build-cascade

eval:     ## score every evaluation set and write the summary tables
	$(GF) eval --detectors heuristic_raw,heuristic,tfidf,embedding,transformer,cascade

bench:    ## measure single-request CPU latency
	$(GF) bench --detectors heuristic,tfidf,embedding,transformer,cascade

recal:    ## check whether the operating point survives real French traffic
	$(GF) recalibrate --detectors heuristic,tfidf,embedding,transformer

report:   ## regenerate the figures and reports/results.md
	$(GF) report

all: data train cascade eval bench recal report  ## the whole pipeline, start to finish

serve:    ## run the screening API on :8000
	$(GF) serve

test:     ## run the offline test suite (no downloads)
	$(PY) -m pytest -m "not network"

test-all: ## also run the tests that download corpora from the Hub
	$(PY) -m pytest

lint:     ## ruff
	$(PY) -m ruff check src tests

clean:    ## remove generated artefacts (keeps the downloaded corpora)
	rm -rf artifacts reports/figures/*.png reports/*.csv reports/*.json reports/*.parquet reports/results.md
