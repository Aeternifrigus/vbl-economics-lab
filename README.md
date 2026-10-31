# packaged-water-lab

Testing the "economic engine of Indian packaged water" narrative against real filings, oil, FX and weather data.

The project started from three industry reports with strong claims (a 42% Q2 volume surge, r = 0.89 between heat and volume, a 38% gross margin floor when Brent is above 90 dollars). Those claims are treated here as hypotheses. `claims/claims.csv` records each one, the evidence found, and a verdict.

## Status (read this first)

| Piece | State |
|---|---|
| Operating leverage model (opex = fixed + variable x revenue) | Trained on 6 quarters of VBL filings. Descriptive only, wide intervals. |
| Distributed-lag gross margin model (oil lags + weather) | Implemented and tested on synthetic data with known coefficients. Not trained on real data yet. |
| Rolling-origin backtest vs seasonal-naive | Implemented and tested. Needs enough real quarters to run. |
| Ingestion clients (FRED oil and FX, Open-Meteo weather) | Written with parsing tests. Never run against the live APIs, because the sandbox that built this had no access to them. First run is yours. |
| Release guard | Training refuses below 20 usable quarters unless `--force` is passed. |

Why the lag model is not trained: only six clean quarters of VBL data were available when this was built (Q1 2025 to Q2 2026), and the oil and weather snapshots were not reachable. Training a ridge model on six rows would produce numbers that look like a result and are not. The guard exists so the repo cannot publish one by accident.

## What the trained model says (operating leverage)

See `reports/operating_leverage.md`. On the six quarters, the EBITDA margin swings about 14.4 points through the opex ratio and only about 2.6 points through gross margin. Fixed cost comes out near 1,030 crore a quarter (90% interval 463 to 1,596) and variable cost near 16% of revenue (90% interval 7% to 25%). The seasonal margin swing is mostly volume spreading fixed cost, not input cost. Gross margin and the opex ratio are positively correlated across these quarters (high gross margin in the off-season), which is why the opex share of variance exceeds 100% and the covariance term is negative.

## Quickstart

```bash
pip install -e ".[dev]"
make test            # 20 tests
make validate        # checks the VBL table (identities, gaps, sources)
make leverage        # fits the operating leverage model, writes models/ and reports/
```

## Making it live (hybrid design)

Automated, monthly (GitHub Actions, `.github/workflows/refresh.yml`):

1. Add a repository secret `FRED_API_KEY` (free from fred.stlouisfed.org).
2. The workflow pulls Brent and USD/INR from FRED and population-weighted weather from Open-Meteo, writes dated CSV snapshots to `data/external/` with SHA-256 hashes in `manifest.json`, rebuilds the quarterly feature table, and commits.

Manual, about four times a year (about 15 minutes):

1. Append the new quarter to `data/raw/vbl_quarterly.csv`. Every row needs a `source_url`.
2. Run `make refit`, which validates, rebuilds features, trains, and backtests.

To reach the 20-quarter threshold, extend the CSV back through 2021 from VBL's investor presentations. `data/raw/vbl_q2_history.csv` already holds several earlier Q2 gross margin and volume points with sources.

Run `python -m pwl ingest` once locally first to confirm the three API clients work from your machine before relying on the schedule.

## Layout

```
data/raw/            VBL quarterly table and Q2 history, each row with a source URL
data/external/       API snapshots plus manifest.json (written by ingest)
claims/claims.csv    original claims vs evidence and verdicts
src/pwl/ingest/      FRED and Open-Meteo clients
src/pwl/features.py  quarterly weather, oil in INR, lags (anomalies use earlier years only)
src/pwl/models/      operating leverage (OLS) and distributed-lag ridge
src/pwl/backtest.py  rolling-origin evaluation against baselines
models/              fitted artifacts with data hashes
reports/             generated reports
tests/               parsing, leakage, parameter recovery, guards
```

## Known limits

- VBL is a soft-drink-led bottler: packaged water was 16% to 22% of its volume in the quarters checked. Results describe VBL, not the water category.
- Reported volume growth includes acquisitions (BevCo in 2024, Twizza in 2026). Use India organic growth where available.
- The population weights in `config.py` are rough assumptions.
- The EBITDA margin here is computed from the filings table and differs slightly from the company's own figure in some quarters.
