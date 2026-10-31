# Model card: operating leverage (v2026Q2)

- Purpose: describe how much of VBL's quarterly EBITDA margin swing comes from fixed cost spread over volume.
- Model: opex = fixed + variable x revenue, fit by ordinary least squares, with t-based 90% intervals and leave-one-out error.
- Data: nine consolidated quarters (2022Q1, 2022Q4, 2023Q1, 2025Q1 to 2026Q2) from `data/raw/vbl_quarterly.csv`. The hash is recorded in `models/leverage_2026Q2.json`.
- Results: R squared 0.89, fixed cost about 400 crore a quarter (90% interval 51 to 748), variable cost about 25% of revenue (90% interval 19% to 31%), leave-one-out error about 5 points of EBITDA margin.
- Margin attribution: the opex ratio spans 14.4 points and gross margin 7.2 points across these quarters.
- Intended use: explaining margin seasonality and sanity-checking the narrative in the original reports.
- Not for: forecasting, investment decisions, or conclusions about packaged water specifically.
- Main risks: nine observations with a gap from 2023Q2 to 2024Q4, pooled across years with different cost scales, consolidated multi-product numbers, and a mix of two source types (a filings statement and an aggregator).

The distributed-lag gross margin model has no release. It refuses to train below 20 usable quarters. See the README status table.
