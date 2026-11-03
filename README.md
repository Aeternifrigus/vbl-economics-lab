# vbl-economics-lab

**Live site:** [aeternifrigus.github.io/vbl-economics-lab](https://aeternifrigus.github.io/vbl-economics-lab/) (the story, the hypotheses, and the charts). By [Aeternifrigus](https://aeternifrigus.netlify.app).

Testing the "economic engine of Indian packaged water" narrative against Varun Beverages (VBL) filings, oil, FX and weather data, and operating the result as a small, reproducible ML system.

The project started from three industry reports with strong claims (a 42% Q2 volume surge, r = 0.89 between heat and volume, a 38% gross margin floor when Brent is above 90 dollars). Those claims are treated as hypotheses. `claims/claims.csv` records each one, the evidence found, and a verdict. VBL is a soft-drink-led bottler, so the findings describe VBL, not the packaged water category.

## What the data says

**Oil does lower gross margin, but about 15 times less than the original report claimed.** In a model with Brent lags and quarter-of-year terms, a 10 dollar per barrel rise in Brent held for a year lowers gross margin by about 0.43 points (90% block-bootstrap interval 0.25 to 0.66). The report implied at least 7 points per 10 dollars. Brent was above 90 dollars in five quarters (2022Q1 to Q3, 2026Q2 and Q3) and gross margin stayed between 50.5% and 55.0% in all of them, nowhere near 38%.

**It is a real effect, not noise.** One-quarter-ahead backtest over 26 quarters (`reports/backtest.json`):

| Features | MAE (gross margin points) | vs seasonal-naive |
|---|---|---|
| Seasonal-naive baseline (same quarter last year) | 1.90 | n/a |
| Quarter-of-year terms only | 1.76 | 7% better |
| Quarter-of-year terms + oil lags | **1.54** | **19% better** |
| Quarter-of-year terms + weather | 1.81 | 5% better |
| Everything | 1.65 | 13% better |

Oil cuts error by 12% beyond seasonality. Weather makes gross margin forecasts worse, which fits the theory that weather should act on volume, not gross margin. The feature set was chosen by looking at this same backtest, so treat the 19% as slightly optimistic.

**The seasonal margin swing is mostly operating leverage.** Median EBITDA margin is 27.7% in Q2, 21.4% in Q3, 18.8% in Q1 and 12.5% in Q4 (the report had 21.8%, 13.5%, 14.2% and 16.0%). The cleanest evidence is within each year: from Q2 to Q4, net sales fall by a median 51% but operating costs by only 15%, so costs go from about 26% of sales to about 43%. (A single pooled cost line gives about 205 crore fixed plus 26% of sales, but it mixes ten years of growth with the seasons and understates how sticky costs are within a year.)

## How the system works

```
 VBL results PDFs ──> pwl extract ──> data/raw/vbl_quarterly.csv ──┐   (quarterly, manual)
 (tie-out checked)                    (+ manual_statements.csv)     │
                                                                    ├──> pwl features ──> pwl train ──> model + metadata
 FRED (Brent, USD/INR) ──┐                                          │         │                │
 Open-Meteo (weather) ───┴──> pwl ingest ──> data/external/ ────────┘         │                v
                                (monthly, automated, SHA-256 manifest)        │       pwl backtest (ablations)
                                                                              v                │
                                              pwl nowcast <── oil to date     pwl monitor <────┘
                                              (current quarter, interval,     (drift and error checks)
                                               extrapolation flag)                    │
                                                                                      v
                                                                          pwl release  ──>  verified bundle
```

| Command | What it does |
|---|---|
| `pwl extract PDFS` | Parse consolidated statements into reconciled line items, merge manual rows, refuse columns that fail tie-outs |
| `pwl validate` | Check the table: accounting identity, margin plausibility, sources, gaps |
| `pwl ingest` | Pull Brent, USD/INR and weather, write hashed snapshots |
| `pwl features` | Build the quarterly feature table |
| `pwl train` | Train the gross margin model. Refuses below 20 quarters unless `--force` |
| `pwl backtest` | Rolling-origin backtest with feature-group ablations |
| `pwl leverage` | Fit the operating leverage model and report |
| `pwl nowcast` | Running estimate of the current quarter's gross margin, with a 90% interval and an extrapolation flag |
| `pwl monitor` | Input drift (Brent regime) and performance drift checks, writes `reports/monitor.json` |
| `pwl site` | Write `docs/data.js`, the data behind the project website, from the repository's own outputs |
| `pwl release --tag vX` | Bundle model, data, evaluation and model card with a verified hash manifest |
| `pwl refit` | validate, leverage, features, train, backtest in one go. Never forces training |

Exit codes: 0 success, 1 error or failed validation, 2 missing input, 3 not enough data, 4 drift (`monitor --strict`).

## Production practices

- **Reproducible installs.** `requirements.lock` and `requirements-dev.lock` are hash-pinned. CI, the Docker image and both scheduled workflows install with `--require-hashes`. The results in this README reproduce to the digit across two library versions I tested.
- **CI on every push.** ruff, ruff format, mypy and the full test suite (100 tests, gate 75%). Tests include an end-to-end pipeline on synthetic data with a planted signal, so a misaligned join or leaky feature fails the build.
- **Data you can trust.** Every parsed column must tie out to the filing's own totals, and a test pins the table to the margins the company itself reported. The earlier extraction that read the wrong line items is quarantined in `data/quarantine/`.
- **Automated refresh.** A monthly workflow pulls oil, FX and weather, validates, retrains, nowcasts, runs the drift checks, prints them in the job summary and commits the snapshots and reports.
- **Honest outputs.** Nowcasts carry an empirical interval and flag when oil is outside the training range. The monitor warns when Brent is in a regime the model has not seen (it is, as of 2026-11-03).
- **Verified releases.** `pwl release` refuses a model trained on different data than the current snapshot and verifies its own bundle. A tag like `v2026Q3` publishes it through GitHub Actions.
- **Operations.** [docs/RUNBOOK.md](docs/RUNBOOK.md) covers the quarterly routine, reading the monitor, rollback and troubleshooting. [SECURITY.md](SECURITY.md) covers secrets.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install                  # hash-verified, exactly what CI runs (Python 3.12)
make test                     # 90 tests, about 5 minutes (runs the full pipeline several times)
make lint typecheck
make refit                    # validate, leverage, features, train, backtest
make nowcast monitor
make release TAG=v2026Q3      # writes dist/vbl-economics-lab-v2026Q3.tar.gz and .sha256
```

To run the live pieces you need a free FRED API key: copy `.env.example` to `.env` and set `FRED_API_KEY`. Weather needs no key. In GitHub, add the same key as the repository secret `FRED_API_KEY`.

Docker: `docker compose run --rm pwl validate` (the image installs from the hash-pinned lock; I could not build it in the environment this was written in, so run `docker compose build` once yourself).

## Data

`data/raw/vbl_quarterly.csv` holds 38 quarters (2016Q4 to 2026Q3) of VBL consolidated results in crore. Missing: 2017Q1 and 2017Q2, which are not in the supplied PDFs.

| Column | Meaning |
|---|---|
| `quarter` | Calendar quarter, e.g. `2025Q2` (quarter ended 30 June 2025) |
| `revenue_cr` | Revenue from operations |
| `excise_duty_cr` | Excise duty, included in revenue and in cost of goods in the filings |
| `gross_profit_cr` | Revenue minus materials, excise duty, purchases of stock-in-trade and change in inventories |
| `opex_cr` | Employee benefits expense plus other expenses |
| `ebitda_cr` | `gross_profit_cr - opex_cr` |
| `volume_growth_yoy_pct` | Reported consolidated volume growth, where known (includes acquisitions) |
| `source` | `file:<pdf>#p<page>` or an `http` URL. Required on every row |

**Margins are computed on net revenue** (`revenue_cr - excise_duty_cr`), which is the base VBL uses for its reported margins. On that basis the table reproduces the six Q2 gross margins the company reported for 2021 to 2026 to within 0.04 points, five Q2 EBITDA margins to within 0.04 points, and the 2025Q3 gross margin.

The PDFs have an OCR-quality text layer (decimal points as commas or periods, header typos). The parser normalises amounts by shape and rejects any column where revenue plus other income differs from total income, or the expense lines differ from total expenses. 36 quarters came from 29 text-layer PDFs. Five PDFs are scans OCR cannot read reliably, so two quarters (2018Q2 and 2026Q3) were read from the page images and entered in `data/raw/manual_statements.csv`, where they go through the same tie-outs.

To add quarters: `pwl extract /path/to/pdfs --dry-run`, then `--overwrite` to write (volume fields are kept), then `make validate`. The 2026Q3 filing restates earlier quarters (its note 10), and the extractor keeps the original current-quarter column for each.

### Data sources and licensing

- **VBL results**: public filings by Varun Beverages Limited on NSE and BSE. The repository stores extracted figures and one statement page of text as a test fixture. The PDFs are not redistributed.
- **Brent crude (`DCOILBRENTEU`) and USD/INR (`DEXINUS`)**: from FRED, Federal Reserve Bank of St. Louis, via the FRED API with your own key. Check [FRED's terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html) before redistributing snapshots.
- **Weather**: [Open-Meteo.com](https://open-meteo.com/), historical archive API, licensed CC BY 4.0. The free API is for non-commercial use, see [its license page](https://open-meteo.com/en/license).

`data/external/` contains snapshots of the FRED and Open-Meteo data. Check each provider's current terms if you fork this for anything commercial.

## Known limits

- VBL is a soft-drink-led bottler (packaged water was 16% to 22% of volume in the quarters checked), and 38 quarters is a small sample. Intervals are wide and a single quarter proves little.
- The oil effect is estimated from quarterly data with highly correlated lags, so individual lag coefficients are unstable. Quote the cumulative effect (`gm_points_per_10usd_brent` in the model metadata).
- Q4 gross margin is inflated by inventory build-up (60% to 66% in 2017, 2018 and 2020 against about 55% recently). The validator reports these as warnings and the model controls for quarter of year.
- Weather to volume is not tested. It needs organic volume by quarter, which is in VBL's investor presentations, not the results PDFs.
- Brent is currently outside the range the model was trained on. Nowcasts say so (`extrapolating`) and should not be trusted as point estimates.
- Reported volume growth includes acquisitions (BevCo in 2024, Twizza in 2026), and the 2026Q3 filing restates prior quarters, so 2026Q3 is on a slightly different basis from earlier columns.
- Population weights for the weather index in `settings.py` are rough assumptions.

## License

MIT, see [LICENSE](LICENSE).
