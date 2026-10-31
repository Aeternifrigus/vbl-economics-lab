# packaged-water-lab

Testing the "economic engine of Indian packaged water" narrative against real filings, oil, FX and weather data.

The project started from three industry reports with strong claims (a 42% Q2 volume surge, r = 0.89 between heat and volume, a 38% gross margin floor when Brent is above 90 dollars). Those claims are treated as hypotheses. `claims/claims.csv` records each one, the evidence found, and a verdict.

## Status

| Piece | State |
|---|---|
| Operating leverage model (opex = fixed + variable x revenue) | Fitted on 9 reconciled quarters. Descriptive only, wide intervals. |
| Distributed-lag gross margin model (oil lags + weather) | Built, tested on synthetic data with a planted signal, and run end to end. **Not trained on real data**: it refuses below 20 usable quarters and the table has 9. |
| Rolling-origin backtest vs seasonal-naive | Built and tested. Needs more than 12 usable quarters. |
| Ingestion (FRED oil and FX, Open-Meteo weather) | Working against the live APIs. Snapshots are in `data/external/`. |
| PDF extraction (`pwl extract`) | Parser tested on a real statement. Run it on your VBL results PDFs to add quarters. |

To unlock the lag model, extend `data/raw/vbl_quarterly.csv` to at least 20 usable quarters (see "Adding quarters").

## What the data says so far

See `reports/operating_leverage.md`. Across the nine reconciled quarters (2022 to 2026), the opex ratio swings about 14.4 points of margin and gross margin about 7.2. So the seasonal EBITDA swing is mostly volume spreading fixed cost (opex ratio explains about 104% of margin variance, partly offset by a negative covariance), with a separate upward drift in gross margin from about 51% in 2022 to 55% to 58% in 2025 and 2026. The fit puts fixed cost near 400 crore a quarter (90% interval 51 to 748) and variable cost near 25% of revenue (90% interval 19% to 31%). Leave-one-out error is large (about 5 points of EBITDA margin) because pooling 2022 and 2026 mixes different cost scales, so treat the fit as a description of the pattern, not a forecasting tool.

## Data

`data/raw/vbl_quarterly.csv` (all amounts in crore, consolidated):

| Column | Meaning |
|---|---|
| `quarter` | Calendar quarter, e.g. `2025Q2` (quarter ended 30 June 2025) |
| `revenue_cr` | Revenue from operations |
| `gross_profit_cr` | Revenue minus materials consumed, excise duty, purchases of stock-in-trade and the change in inventories |
| `opex_cr` | Employee benefits expense plus other expenses |
| `ebitda_cr` | `gross_profit_cr - opex_cr` |
| `volume_growth_yoy_pct` | Reported consolidated volume growth, where known (includes acquisitions) |
| `source` | An `http` URL or a `file:<pdf>#p<page>` reference. Required on every row. |

Nine quarters are present: 2022Q1, 2022Q4 and 2023Q1 parsed from the company's results statement, and 2025Q1 to 2026Q2 from a table that aggregates NSE and BSE filings. The definitions above are fixed so every quarter is comparable. The validator checks the accounting identity, margin plausibility bands (gross margin 48% to 60%, EBITDA margin 10% to 35%, as warnings) and that every row has a source. A test pins all nine quarters to independently computed margins, so a bad re-extraction cannot slip in.

`data/quarantine/` holds an earlier 30-quarter extraction that read the wrong line items. Do not use it. See the README in that folder for what was wrong.

## Adding quarters

```bash
pip install -e ".[pdf]"
pwl extract /path/to/vbl/results/pdfs --dry-run   # parse and report, write nothing
pwl extract /path/to/vbl/results/pdfs             # merge new quarters into the table
```

The parser reads only the consolidated statement, takes the three-month columns, converts the stated unit (million, crore or lakh) to crore, and keeps existing quarters unless `--overwrite` is passed. A merged table that fails validation is not written. Check any newly added quarter against the filing before trusting it, especially restated comparatives and statements with unusual layouts.

## Quickstart

```bash
pip install -e ".[dev]"
make test         # 67 tests, coverage gate 75% (set in pyproject.toml)
make lint         # ruff check and format check
make typecheck    # mypy
make validate     # validate the VBL table
make leverage     # fit the operating leverage model, write models/ and reports/
make refit        # validate, leverage, features, train, backtest (never forces training)
```

Exit codes: 0 success, 1 error or failed validation, 2 missing input file, 3 not enough data for the requested model.

## Making it live (hybrid design)

Automated monthly (`.github/workflows/refresh.yml`, also runnable by hand):

1. Add a repository secret `FRED_API_KEY` (free from fred.stlouisfed.org).
2. The workflow pulls Brent and USD/INR from FRED and population-weighted weather from Open-Meteo, writes dated snapshots to `data/external/` with SHA-256 hashes in `manifest.json`, rebuilds the quarterly feature table, and commits to `main`.

Manual, about four times a year: run `pwl extract` on the new results PDF (or append the row by hand with its source), then `make refit`.

Model binaries (`models/*.joblib`) are not committed. Publish them as release assets with the feature snapshot hash from `models/*.json`. Metadata and reports are committed.

## Layout

```
data/raw/            reconciled VBL table and Q2 history, each row sourced
data/external/       API snapshots, quarterly features, manifest.json
data/quarantine/     earlier extraction that read the wrong line items (do not use)
claims/claims.csv    original claims vs evidence and verdicts
src/pwl/extract.py   statement parser with fixed line-item definitions
src/pwl/ingest/      FRED and Open-Meteo clients
src/pwl/features.py  quarterly weather, oil in INR, lags (anomalies use earlier years only)
src/pwl/models/      operating leverage (OLS) and distributed-lag ridge
src/pwl/backtest.py  rolling-origin evaluation, baselines matched by calendar quarter
tests/               unit, integration (known planted signal), CLI and infrastructure tests
```

## Known limits

- VBL is a soft-drink-led bottler: packaged water was 16% to 22% of its volume in the quarters checked. Results describe VBL, not the water category.
- Reported volume growth includes acquisitions (BevCo in 2024, Twizza in 2026). Use India organic growth where available.
- Excise duty is counted as a cost of goods. It affects 2022 and 2023 quarters (about 1.5 points of gross margin) and is absent from later statements, which is a small comparability break.
- Oil lags are highly correlated in real data, so individual lag coefficients are unstable. Quote the cumulative effect (`cumulative_brent_inr_effect` in the model metadata).
- The population weights for the weather index in `settings.py` are rough assumptions.
- The 2025 to 2026 rows come from an aggregator that mirrors the exchange filings. Reconcile them against the filings when you extend the table.
