# packaged-water-lab

Testing the "economic engine of Indian packaged water" narrative against Varun Beverages (VBL) filings, oil, FX and weather data.

The project started from three industry reports with strong claims (a 42% Q2 volume surge, r = 0.89 between heat and volume, a 38% gross margin floor when Brent is above 90 dollars). Those claims are treated as hypotheses. `claims/claims.csv` records each one, the evidence found, and a verdict.

## Status

| Piece | State |
|---|---|
| Data: 38 quarters of VBL consolidated results (2016Q4 to 2026Q3) | Parsed from the company's results PDFs. Every accepted column ties out to the filing's own totals, and the table reproduces the margins the company itself reported. |
| Operating leverage model | Fitted on 38 quarters. |
| Gross margin model (Brent lags + seasonality) | Trained on real data (not forced) and backtested over 26 quarters. |
| Weather | Tested and left out of the gross margin model. It belongs in a volume model, which needs an organic volume series that the results PDFs do not contain. |
| Ingestion (FRED oil and FX, Open-Meteo weather) | Working against the live APIs. Snapshots are in `data/external/`. |

## What the data says

**Oil does lower gross margin, but about 15 times less than the original report claimed.** In the model with Brent lags and quarter-of-year terms, a 10 dollar per barrel rise in Brent held for a year lowers gross margin by about 0.43 points (90% block-bootstrap interval 0.25 to 0.66). The report implied at least 7 points per 10 dollars. Brent was above 90 dollars in five quarters (2022Q1 to Q3, 2026Q2 and Q3) and gross margin stayed between 50.5% and 55.0% in all of them, nowhere near 38%.

**It is a real effect, not noise.** In a one-quarter-ahead backtest over 26 quarters (`reports/backtest.json`):

| Features | MAE (gross margin points) | vs seasonal-naive |
|---|---|---|
| Seasonal-naive baseline (same quarter last year) | 1.90 | n/a |
| Quarter-of-year terms only | 1.76 | 7% better |
| Quarter-of-year terms + oil lags | **1.54** | **19% better** |
| Quarter-of-year terms + weather | 1.81 | 5% better |
| Everything | 1.65 | 13% better |

Oil cuts error by 12% beyond seasonality. Weather makes gross margin forecasts worse, which fits the revised theory: weather should act on volume, not on gross margin. The feature set was chosen by looking at this same backtest, so treat the 19% as a slightly optimistic figure.

**The seasonal margin swing is mostly operating leverage.** Across the 38 quarters the opex ratio swings 35 points and gross margin 16, and median EBITDA margin runs 27.7% in Q2, 21.4% in Q3, 18.8% in Q1 and 12.5% in Q4 (the report had 21.8%, 13.5%, 14.2% and 16.0%). The fit puts fixed cost near 205 crore a quarter (90% interval 130 to 281) and variable cost near 26% of net revenue (24% to 28%). Leave-one-out error is about 6.5 points because revenue grew about 15-fold over the period, so read it as a description of the pattern, not a forecasting tool.

## Data

`data/raw/vbl_quarterly.csv` (all amounts in crore, consolidated):

| Column | Meaning |
|---|---|
| `quarter` | Calendar quarter, e.g. `2025Q2` (quarter ended 30 June 2025) |
| `revenue_cr` | Revenue from operations |
| `excise_duty_cr` | Excise duty, shown inside revenue and cost of goods in the filings |
| `gross_profit_cr` | Revenue minus materials, excise duty, purchases of stock-in-trade and change in inventories |
| `opex_cr` | Employee benefits expense plus other expenses |
| `ebitda_cr` | `gross_profit_cr - opex_cr` |
| `volume_growth_yoy_pct` | Reported consolidated volume growth, where known (includes acquisitions) |
| `source` | `file:<pdf>#p<page>` (or an `http` URL). Required on every row. |

**Margins are computed on net revenue** (`revenue_cr - excise_duty_cr`), because that is the base VBL uses for its reported margins. On that basis the table reproduces the six Q2 gross margins the company reported (2021 to 2026) to within 0.04 points, five Q2 EBITDA margins to within 0.04 points, and the 2025Q3 gross margin (56.7%). `tests/test_validate.py` enforces this, so a bad re-extraction or a change to the definitions fails the build.

### How the data was built

The results PDFs are filings with an OCR-quality text layer: decimal points come out as commas or periods, headers have typos, and amounts are sometimes garbled. The parser handles this by normalising numbers by shape (every amount has two decimals) and by refusing any column that fails two tie-outs against the filing's own totals:

- revenue + other income = total income
- the eight expense lines sum to total expenses

A misread digit therefore gets rejected instead of silently merged. 36 quarters came from 29 text-layer PDFs. The five scanned PDFs have no usable text, and OCR cannot read their table rows reliably enough to pass the tie-outs, so two quarters were read from the page images by hand and entered in `data/raw/manual_statements.csv`. Those rows go through the same tie-outs. The two missing quarters, 2017Q1 and 2017Q2, are not in any supplied PDF.

`data/quarantine/` holds an earlier extraction that read the wrong line items. Do not use it. See the README in that folder.

## Adding quarters

```bash
pip install -e ".[pdf]"
pwl extract /path/to/vbl/results/pdfs --dry-run   # parse and report, write nothing
pwl extract /path/to/vbl/results/pdfs --overwrite # rebuild the table (keeps volume fields)
```

The parser reads only the consolidated statement, takes the three-month columns, converts the stated unit to crore, and merges `data/raw/manual_statements.csv` as well. Without `--overwrite`, existing quarters are kept. A merged table that fails validation is not written. Prefer each filing's current-quarter column: later filings restate comparatives (the 2026Q3 filing restates 2026Q2 revenue up by about 1.1%), and the extractor keeps the original column.

For a scanned PDF, add one row per quarter to `data/raw/manual_statements.csv` from the page image. `--ocr` exists but rarely passes the tie-outs on scans.

## Quickstart

```bash
pip install -e ".[dev]"
make test         # 77 tests, about 3 minutes (runs the full pipeline several times)
make lint         # ruff check and format check
make typecheck    # mypy
make validate     # validate the VBL table
make leverage     # fit the operating leverage model, write models/ and reports/
make refit        # validate, leverage, features, train, backtest (never forces training)
```

Exit codes: 0 success, 1 error or failed validation, 2 missing input file, 3 not enough data for the requested model.

The lag model refuses to train below 20 usable quarters (`--force` allows a smoke run, recorded as `forced: true` in the model metadata).

## Making it live (hybrid design)

Automated monthly (`.github/workflows/refresh.yml`, also runnable by hand):

1. Add a repository secret `FRED_API_KEY` (free from fred.stlouisfed.org).
2. The workflow pulls Brent and USD/INR from FRED and population-weighted weather from Open-Meteo, writes dated snapshots to `data/external/` with SHA-256 hashes in `manifest.json`, rebuilds the quarterly feature table, and commits to `main`.

Manual, about four times a year: when VBL files a quarter, run `pwl extract` (or add a manual row), then `make refit`.

Model binaries (`models/*.joblib`) are not committed. Publish them as release assets with the feature snapshot hash from `models/*.json`. Metadata and reports are committed.

## Layout

```
data/raw/            reconciled VBL table, manual statements, Q2 history (company-reported margins)
data/external/       API snapshots, quarterly features, manifest.json
data/quarantine/     earlier extraction that read the wrong line items (do not use)
claims/claims.csv    original claims vs evidence and verdicts
src/pwl/extract.py   statement parser, tie-outs, manual statements, optional OCR
src/pwl/accounting.py net revenue basis shared by every module
src/pwl/ingest/      FRED and Open-Meteo clients
src/pwl/features.py  quarterly weather, oil in INR, lags, quarter dummies (anomalies use earlier years only)
src/pwl/models/      operating leverage (OLS) and distributed-lag ridge with bootstrap interval
src/pwl/backtest.py  rolling-origin evaluation, baselines matched by calendar quarter
tests/               unit, integration (known planted signal), CLI and infrastructure tests
```

## Known limits

- VBL is a soft-drink-led bottler: packaged water was 16% to 22% of its volume in the quarters checked. Results describe VBL, not the water category.
- Q4 gross margin is inflated by inventory build-up (60% to 66% in 2017, 2018 and 2020 against about 55% recently). The validator reports these as warnings, and the model controls for quarter of year.
- The weather to volume hypothesis is not tested. It needs organic volume by quarter, which is in VBL's investor presentations, not in the results PDFs.
- Oil lags are highly correlated, so individual lag coefficients are unstable. Quote the cumulative effect (`gm_points_per_10usd_brent` in the model metadata).
- Reported volume growth includes acquisitions (BevCo in 2024, Twizza in 2026). The 2026Q3 filing also restates earlier quarters (its note 10), so 2026Q3 is on a slightly different basis from the original columns used for earlier quarters.
- The population weights for the weather index in `settings.py` are rough assumptions.
- 38 quarters is a small sample. The intervals are wide and a single quarter proves little.
