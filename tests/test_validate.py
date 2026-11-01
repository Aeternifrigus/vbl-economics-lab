import pandas as pd
import pytest

from pwl.config import DATA_RAW
from pwl.exceptions import DataValidationError
from pwl.validate import check_vbl, load_vbl, validate_or_raise, vbl_warnings

# Gross margin the company itself reported for these quarters, quoted in earnings commentary and
# collected in data/raw/vbl_q2_history.csv. They are independent of the parser, so a bad
# re-extraction or a change to the line-item definitions cannot pass this test by accident.
REPORTED_Q3_2025_GM = 56.7


def _shipped():
    return load_vbl(DATA_RAW / "vbl_quarterly.csv")


def test_shipped_data_is_clean():
    assert check_vbl(_shipped()) == []


def test_table_reproduces_company_reported_q2_margins():
    table = _shipped().set_index("quarter")
    reported = pd.read_csv(DATA_RAW / "vbl_q2_history.csv").set_index("quarter")
    checked = 0
    for quarter, row in reported.iterrows():
        net = table.loc[quarter, "revenue_cr"] - table.loc[quarter, "excise_duty_cr"]
        gm = table.loc[quarter, "gross_profit_cr"] / net * 100
        assert gm == pytest.approx(row["gross_margin_pct"], abs=0.1), quarter
        if pd.notna(row["ebitda_margin_company_pct"]):
            ebitda = table.loc[quarter, "ebitda_cr"] / net * 100
            assert ebitda == pytest.approx(row["ebitda_margin_company_pct"], abs=0.15), quarter
        checked += 1
    assert checked >= 6


def test_table_reproduces_company_reported_q3_2025_gross_margin():
    row = _shipped().set_index("quarter").loc["2025Q3"]
    gm = row["gross_profit_cr"] / (row["revenue_cr"] - row["excise_duty_cr"]) * 100
    assert gm == pytest.approx(REPORTED_Q3_2025_GM, abs=0.1)


def test_margins_use_net_revenue_not_gross():
    row = _shipped().set_index("quarter").loc["2026Q2"]
    on_gross = row["ebitda_cr"] / row["revenue_cr"] * 100
    on_net = row["ebitda_cr"] / (row["revenue_cr"] - row["excise_duty_cr"]) * 100
    assert on_net - on_gross > 0.4  # the gap that made 27.1% versus the reported 27.7%


def test_only_q4_seasonal_quarters_trip_the_plausibility_bands():
    """Q4 has an inventory build and a low season, so its margins legitimately sit outside."""
    odd = [w for w in vbl_warnings(_shipped()) if "margin" in w]
    assert odd, "expected the Q4 warnings to be reported, not hidden"
    assert all(w.split(":")[0].endswith("Q4") for w in odd)


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
