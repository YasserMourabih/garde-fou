VENV ?= .venv
PY   := $(VENV)/bin/python
GF   := $(VENV)/bin/garde-fou

.PHONY: help install data train cascade eval bench report all serve test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install the package with every extra
	python3 -m venv $(VENV)
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

test:     ## run the offline test suite
	$(PY) -m pytest

lint:     ## ruff
	$(PY) -m ruff check src tests

clean:    ## remove generated artefacts (keeps the downloaded corpora)
	rm -rf artifacts reports/figures/*.png reports/*.csv reports/*.json reports/*.parquet reports/results.md
