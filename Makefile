PY ?= python3
export PYTHONPATH := src

.PHONY: test validate leverage ingest features refit backtest

test:
	$(PY) -m pytest -q

validate:
	$(PY) -m pwl validate

leverage:
	$(PY) -m pwl leverage

ingest:
	$(PY) -m pwl ingest

features:
	$(PY) -m pwl features

refit: validate features
	$(PY) -m pwl train
	$(PY) -m pwl backtest
