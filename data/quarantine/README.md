# Quarantined data (do not use for modelling)

These files came from an early PDF extraction that read the wrong line items.

- `vbl_quarterly_unverified.csv`: 30 quarters, 2019Q1 to 2026Q2. Revenue spot-checks correctly
  (for example 2023Q2 and 2022Q2 match the reported growth), but the other columns do not.
- `vbl_extracted_raw_v4.csv`: the raw extractor output behind it, including standalone rows.

What was wrong, checked against the six quarters 2025Q1 to 2026Q2 in the filings table and against
the full statement for 2023Q1:

- "Gross profit" left out the change in inventories and mixed in excise duty differently, so gross
  margin swung between about 42% and 63% when the reported figure stays near 50% to 58%.
- "EBITDA" was the wrong line (about 5% of revenue against 22% to 28% reported), and was later
  overwritten as gross profit minus opex, which made the validation identity pass by construction.

The reconciled table is `data/raw/vbl_quarterly.csv`, rebuilt from the original results PDFs. The
parser that produces correct line items is `src/pwl/extract.py` (run it with
`python -m pwl extract PDF_DIR`). Every accepted column ties out to the filing's own totals.
