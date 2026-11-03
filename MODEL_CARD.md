# Model cards (v2026Q3)

Project: vbl-economics-lab. Both models describe VBL, a soft-drink-led bottler, not the packaged water category.

## Gross margin model: Brent lags and seasonality

- Purpose: estimate how much of VBL's quarterly gross margin moves with the price of Brent, and test the original report's claim that it moves a lot.
- Model: ridge regression of gross margin (percent of net revenue) on Brent in INR at lags 0 to 3 quarters plus quarter-of-year dummies. The ridge penalty is chosen by time-series cross-validation. Weather features were tested and left out because they worsened the backtest.
- Data: 38 consolidated quarters, 2016Q4 to 2026Q3, from `data/raw/vbl_quarterly.csv`, with FRED Brent and USD/INR snapshots. Hashes are recorded in `models/lagged_gm_2026Q3.json`.
- Result: about -0.43 gross margin points per 10 USD per barrel of Brent held for a year (90% moving-block bootstrap interval -0.66 to -0.25).
- Evaluation: one-quarter-ahead rolling backtest over 26 quarters. MAE 1.54 points against 1.90 for seasonal-naive and 1.76 for seasonality alone. In the latest quarter (2026Q3, gross margin 53.1%) it predicted 54.4 against 56.7 for seasonal-naive.
- Intended use: sizing the oil sensitivity of VBL's gross margin and checking claims about it.
- Not for: investment decisions, or conclusions about packaged water specifically (it is a soft-drink-led bottler).
- Nowcasts: `pwl nowcast` applies the model to the current quarter using Brent to date. The 90% interval is the 5th to 95th percentile of backtest errors (about 26 quarters), so it is empirical. The output flags `extrapolating` when oil is outside the training range, and the metadata stores that range. Nowcasting a quarter that is already in the training data is in-sample.
- Monitoring: `pwl monitor` checks the PSI of recent Brent against earlier history, the share of recent days outside the historical range, and whether the latest backtest errors are unusually large. At the time of writing Brent is far outside the earlier range, so the input-drift check warns, and nowcasts for the current quarter are extrapolations.
- Release: `pwl release` bundles the model, metadata, training data, backtest and this card with a SHA-256 manifest, and refuses a model trained on different data than the current snapshot.
- Main risks: 38 observations, collinear oil lags, a feature set chosen on the same backtest, a linear model that extrapolates poorly when oil is far outside its range, and a Q4 inventory effect that the quarter dummies only partly absorb.

## Operating leverage model

- Purpose: describe how much of the quarterly EBITDA margin swing comes from fixed cost spread over volume.
- Model: opex = fixed + variable x net revenue, ordinary least squares with t-based 90% intervals and leave-one-out error.
- Data: the same 38 quarters.
- Result: R squared 0.93, fixed cost about 205 crore a quarter (130 to 281), variable cost about 26% of net revenue (24% to 28%), leave-one-out error about 6.5 points of EBITDA margin. Caveat: this pooled line mixes ten years of growth with the seasons, so it understates how sticky costs are within a year. Compared within each year, Q2 to Q4 net sales fall by a median 51% and operating costs by only 15%.
- Main risks: revenue grew about 15-fold over the period, so one linear cost line is a rough description. Consolidated multi-product numbers. A gap at 2017Q1 and 2017Q2.
