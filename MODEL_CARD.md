# Model cards (v2026Q3)

## Gross margin model: Brent lags and seasonality

- Purpose: estimate how much of VBL's quarterly gross margin moves with the price of Brent, and test the original report's claim that it moves a lot.
- Model: ridge regression of gross margin (percent of net revenue) on Brent in INR at lags 0 to 3 quarters plus quarter-of-year dummies. The ridge penalty is chosen by time-series cross-validation. Weather features were tested and left out because they worsened the backtest.
- Data: 38 consolidated quarters, 2016Q4 to 2026Q3, from `data/raw/vbl_quarterly.csv`, with FRED Brent and USD/INR snapshots. Hashes are recorded in `models/lagged_gm_2026Q3.json`.
- Result: about -0.43 gross margin points per 10 USD per barrel of Brent held for a year (90% moving-block bootstrap interval -0.66 to -0.25).
- Evaluation: one-quarter-ahead rolling backtest over 26 quarters. MAE 1.54 points against 1.90 for seasonal-naive and 1.76 for seasonality alone. In the latest quarter (2026Q3, gross margin 53.1%) it predicted 54.4 against 56.7 for seasonal-naive.
- Intended use: sizing the oil sensitivity of VBL's gross margin and checking claims about it.
- Not for: investment decisions, or conclusions about packaged water specifically (it is a soft-drink-led bottler).
- Main risks: 38 observations, collinear oil lags, a feature set chosen on the same backtest, and a Q4 inventory effect that the quarter dummies only partly absorb.

## Operating leverage model

- Purpose: describe how much of the quarterly EBITDA margin swing comes from fixed cost spread over volume.
- Model: opex = fixed + variable x net revenue, ordinary least squares with t-based 90% intervals and leave-one-out error.
- Data: the same 38 quarters.
- Result: R squared 0.93, fixed cost about 205 crore a quarter (130 to 281), variable cost about 26% of net revenue (24% to 28%), leave-one-out error about 6.5 points of EBITDA margin.
- Main risks: revenue grew about 15-fold over the period, so one linear cost line is a rough description. Consolidated multi-product numbers. A gap at 2017Q1 and 2017Q2.
