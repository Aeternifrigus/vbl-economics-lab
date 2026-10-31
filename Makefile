PY ?= python3
export PYTHONPATH := src

.PHONY: install test lint typecheck validate extract leverage ingest features backtest refit

install:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src/pwl

validate:
	$(PY) -m pwl validate

# Usage: make extract PDFS=/path/to/vbl/results
extract:
	$(PY) -m pwl extract $(PDFS)

leverage:
	$(PY) -m pwl leverage

ingest:
	$(PY) -m pwl ingest

features:
	$(PY) -m pwl features

backtest:
	$(PY) -m pwl backtest

# Validate, refresh leverage, rebuild features, train and backtest. Never forces training.
refit:
	$(PY) -m pwl refit
