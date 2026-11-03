PY ?= python3
export PYTHONPATH := src

.PHONY: install lock test lint typecheck validate extract leverage ingest features train backtest nowcast monitor release refit

# Exact, hash-verified environment (what CI uses).
install:
	pip install --require-hashes -r requirements-dev.lock
	pip install --no-deps -e .

# Regenerate both lock files after changing dependencies in pyproject.toml.
lock:
	pip install pip-tools
	pip-compile --generate-hashes --strip-extras -o requirements.lock pyproject.toml
	pip-compile --generate-hashes --strip-extras --extra dev --extra pdf -o requirements-dev.lock pyproject.toml

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

train:
	$(PY) -m pwl train

backtest:
	$(PY) -m pwl backtest

# Running estimate of the current quarter's gross margin from oil to date.
nowcast:
	$(PY) -m pwl nowcast

monitor:
	$(PY) -m pwl monitor

# Usage: make release TAG=v2026Q3
release:
	$(PY) -m pwl release --tag $(TAG)

# Validate, refresh leverage, rebuild features, train and backtest. Never forces training.
refit:
	$(PY) -m pwl refit
