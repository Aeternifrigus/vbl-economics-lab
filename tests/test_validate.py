import pandas as pd
import pytest

from pwl.config import DATA_RAW
from pwl.exceptions import DataValidationError
from pwl.validate import check_vbl, load_vbl, validate_or_raise, vbl_warnings

# Margins from the filings (six quarters) and from the full 2023Q1 statement (three quarters),
# written down independently of the CSV so a bad edit or re-extraction cannot slip through.
GOLDEN = {
    # quarter: (gross margin %, EBITDA margin %)
    "2022Q1": (50.82, 18.52),
    "2022Q4": (55.22, 13.62),
    "2023Q1": (51.64, 20.19),
    "2025Q1": (55.47, 22.25),
    "2025Q2": (55.45, 27.90),
    "2025Q3": (58.02, 22.73),
    "2025Q4": (56.72, 14.75),
    "2026Q1": (56.18, 22.75),
    "2026Q2": (56.00, 27.09),
}


def _shipped():
    return load_vbl(DATA_RAW / "vbl_quarterly.csv")


def test_shipped_data_is_clean():
    assert check_vbl(_shipped()) == []


def test_shipped_data_matches_independent_reconciliation():
    df = _shipped().set_index("quarter")
    for quarter, (gm, ebitda) in GOLDEN.items():
        row = df.loc[quarter]
        assert row["gross_profit_cr"] / row["revenue_cr"] * 100 == pytest.approx(gm, abs=0.1)
        assert row["ebitda_cr"] / row["revenue_cr"] * 100 == pytest.approx(ebitda, abs=0.1)


def test_no_shipped_quarter_trips_the_plausibility_bands():
    assert not [w for w in vbl_warnings(_shipped()) if "margin" in w]


def test_detects_broken_identity():
    df = _shipped()
    df.loc[0, "ebitda_cr"] += 50
    assert any("differs from EBITDA" in p for p in check_vbl(df))


def test_detects_duplicates():
    df = _shipped()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "duplicate quarters" in check_vbl(dup)


def test_gaps_are_warnings_not_errors():
    df = _shipped()
    assert check_vbl(df) == []
    assert any("quarters missing" in w for w in vbl_warnings(df))


def test_implausible_margins_warn():
    df = _shipped()
    df.loc[0, "gross_profit_cr"] = df.loc[0, "revenue_cr"] * 0.42
    df.loc[0, "ebitda_cr"] = df.loc[0, "gross_profit_cr"] - df.loc[0, "opex_cr"]
    warnings = vbl_warnings(df)
    assert any("gross margin" in w for w in warnings)


def test_every_row_has_a_source():
    df = _shipped()
    assert df["source"].str.startswith(("http", "file:")).all()
    df.loc[0, "source"] = "somewhere"
    assert any("source must be" in p for p in check_vbl(df))


def test_missing_columns_reported():
    df = _shipped().drop(columns=["opex_cr"])
    assert check_vbl(df) == ["missing columns: ['opex_cr']"]


def test_validate_or_raise():
    validate_or_raise(_shipped())
    df = _shipped()
    df.loc[0, "revenue_cr"] = -1
    with pytest.raises(DataValidationError):
        validate_or_raise(df)


def test_legacy_source_url_column_is_accepted(tmp_path):
    path = tmp_path / "legacy.csv"
    _shipped().drop(columns=["period"]).rename(columns={"source": "source_url"}).to_csv(
        path, index=False
    )
    assert check_vbl(load_vbl(path)) == []
