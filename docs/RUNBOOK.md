# Runbook

How to operate vbl-economics-lab: what runs when, what the alarms mean, and what to do about them.

## What runs automatically

| What | When | Where |
|---|---|---|
| Lint, type-check, tests | every push and pull request | `.github/workflows/ci.yml` |
| Refresh: ingest oil, FX and weather, validate, rebuild features, retrain, nowcast, monitor, commit | 05:00 UTC on the 1st of each month, or by hand | `.github/workflows/refresh.yml` |
| Release: rebuild the model from committed data, bundle it with hashes, publish | when you push a tag like `v2026Q3` | `.github/workflows/release.yml` |
| Dependency update pull requests | weekly | `.github/dependabot.yml` |

The only secret is `FRED_API_KEY` (repository secret). Weather needs no key.

## Every quarter (about 20 minutes)

VBL reports about a month after quarter end.

1. Download the consolidated results PDF into your results folder.
2. `pwl extract /path/to/pdfs --dry-run` and read the output. Every accepted column has passed the
   tie-outs. If the PDF is a scan, add a row to `data/raw/manual_statements.csv` from the page image.
3. `pwl extract /path/to/pdfs --overwrite` to write the table, then `make validate`.
4. Check the new row against the company's reported margins in its earnings release. The table uses
   net revenue (revenue less excise), which is the company's base.
5. `make refit`, read `reports/backtest.json` and `reports/monitor.json`, then commit.
6. To publish a model: `git tag v2026Q4 && git push --tags`. The release workflow does the rest.

## Reading the monitor

`pwl monitor` (also run monthly) writes `reports/monitor.json`. A warning is a prompt to look, not a failure.

| Check | Warns when | What it usually means | What to do |
|---|---|---|---|
| `brent_psi` | PSI of the last 180 days of Brent against earlier history exceeds 0.25 | Oil is in a different regime from most of the training data (it did in 2026) | Treat nowcasts as extrapolation. Check whether the oil effect is stable by retraining after the next quarter lands |
| `brent_out_of_range_share` | over 20% of recent days are outside the historical Brent range | Oil is somewhere the model has never seen | Do not trust the interval. Widen it by hand or pause nowcasts |
| `recent_error_ratio` | the last 4 quarters' error is over 1.5 times the backtest average | The relationship may have shifted (cost structure, hedging, mix) | Inspect the latest quarters, consider retraining on a recent window |
| `latest_error_zscore` | the latest error is more than 2 standard deviations from the earlier errors | One quarter was unusual | Look for a one-off (restatement, acquisition, accounting change) before reacting |

`pwl monitor --strict` exits 4 when any check warns, if you want it to fail a pipeline.

## Nowcasts

`pwl nowcast` estimates the current quarter's gross margin from the average Brent in INR so far and the
three previous quarters. The interval comes from the model's own backtest errors (5th to 95th
percentile), so it is an empirical interval over about 26 quarters, not a guarantee. Early in a
quarter there is little oil data, and in the first days there is none, in which case the command exits 3.

Read `extrapolating` before the number. When it is `true`, `outside_training_range` lists the oil inputs
that are above or below anything the model was trained on. The point estimate is then an extrapolation of
a linear model and the interval understates the uncertainty. This is the case at the time of writing:
Brent in the first days of 2026Q4 is far above the training range. Nowcasting a quarter that is already
in the training data (for example 2026Q3) is in-sample and flatters the model.

## Website

`docs/index.html` is a self-contained page (no libraries, no CDN) that tells the story and draws every chart
from `docs/data.js`. `pwl site` regenerates that file from the repository's own outputs, and the monthly
workflow does it automatically, so the page cannot drift from the results.

- Publish: Settings, Pages, "Deploy from a branch", branch `main`, folder `/docs`. The site appears at
  `https://aeternifrigus.github.io/vbl-economics-lab/`.
- Preview locally: open `docs/index.html` in a browser. It works straight from disk.
- Change the story or wording by editing `docs/index.html`. Change a number by fixing its source, never the page.

## Releases and rollback

A release is `vbl-economics-lab-<tag>.tar.gz` plus a `.sha256` file. Inside, `RELEASE_MANIFEST.json`
lists the SHA-256 of the model, metadata, training data, evaluation and model card.

- Verify a download: compare the archive's hash with the `.sha256` file, then unpack it and check
  each file against the manifest.
- `pwl release` refuses a model that was not trained on the current feature snapshot, so a stale model
  cannot be published.
- Roll back by pointing consumers at the previous tag. Releases are immutable, so do not re-tag. Cut a
  new tag instead.

## Dependencies

Installs are hash-pinned (`requirements.lock`, `requirements-dev.lock`) and tested on Python 3.12.
After changing `pyproject.toml`, run `make lock`, then the full test suite, and commit the lock files
together with the change. Merge Dependabot pull requests only when CI is green.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pwl ingest` exits 1 with "FRED_API_KEY not set" | key missing | export it or add it to `.env` |
| `pwl ingest` fails with an ingestion error | API outage or rate limit | rerun later. The error names the host and never prints the key |
| `pwl extract` rejects a column ("tie-out failed") | a misread amount, almost always a scan or OCR damage | enter the quarter by hand in `data/raw/manual_statements.csv` |
| `pwl train` exits 3 | fewer than 20 usable quarters | extend the table. Do not use `--force` for anything you publish |
| `pwl release` says the model is stale | features changed after training | run `pwl features` then `pwl train` |
| exit codes | 1 error, 2 missing input, 3 not enough data, 4 drift (strict monitor) | see the message printed before exit |

## If a secret leaks

Revoke the FRED key at fred.stlouisfed.org, create a new one, and update the repository secret.
Logs and snapshots never contain it, but the git history of a committed `.env` would.
