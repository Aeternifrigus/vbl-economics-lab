# Model card: operating leverage (v2026Q2)

- Purpose: describe how much of VBL's quarterly EBITDA margin swing comes from fixed cost spread over volume.
- Model: opex = fixed + variable x revenue, fit by ordinary least squares, with t-based 90% intervals and leave-one-out error.
- Data: six quarters, 2025Q1 to 2026Q2, from the filings table in `data/raw/vbl_quarterly.csv` (hash recorded in `models/leverage_2026Q2.json`).
- Results: R squared 0.79, fixed cost about 1,030 crore a quarter, variable cost about 16% of revenue, leave-one-out error 2.8 points of EBITDA margin.
- Intended use: explaining margin seasonality and sanity-checking the narrative in the original reports.
- Not for: forecasting, investment decisions, or conclusions about packaged water specifically.
- Main risks: six observations, one outlier-like quarter (2025Q4), consolidated multi-product numbers, and a margin definition that differs slightly from the company's.

The distributed-lag gross margin model has no release yet. See the status table in the README.
