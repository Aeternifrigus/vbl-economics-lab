import builtins
from pathlib import Path

import pandas as pd
import pytest

from pwl.exceptions import ConfigurationError
from pwl.extract import (
    extract_from_pages,
    merge_into_table,
    merge_rows,
    parse_numbers,
    quarter_label,
    read_pdf_pages,
    split_pages,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vbl_results_2023Q1_statement.txt"


def _rows():
    return extract_from_pages(split_pages(FIXTURE.read_text()), "1-2.-Results.pdf")


def test_parse_numbers_handles_brackets_dashes_and_commas():
    assert parse_numbers("(a) Revenue 39,525.91 22,572.01 - (2,343.25)") == [
        39525.91,
        22572.01,
        0.0,
        -2343.25,
    ]


def test_quarter_label():
    assert quarter_label("31", "March", "2023") == "2023Q1"
    assert quarter_label("31", "December", "2022") == "2022Q4"
    assert quarter_label("30", "April", "2023") is None


def test_only_the_consolidated_statement_is_parsed():
    rows = _rows()
    assert [r.quarter for r in rows] == ["2023Q1", "2022Q4", "2022Q1"]  # not the standalone page
    assert all("#p1" in r.source for r in rows)


def test_line_items_are_mapped_to_the_documented_definitions():
    # Hand calculation from the statement for 2023Q1, in million then divided by 10 for crore:
    # cogs = 19,643.11 + 596.16 + 1,217.30 - 2,343.25 = 19,113.32
    # gross = 39,525.91 - 19,113.32 = 20,412.59 ; opex = 3,391.22 + 9,040.99 = 12,432.21
    q1 = {r.quarter: r for r in _rows()}["2023Q1"]
    assert q1.revenue_cr == pytest.approx(3952.6, abs=0.05)
    assert q1.excise_duty_cr == pytest.approx(59.62, abs=0.05)
    assert q1.gross_profit_cr == pytest.approx(2041.26, abs=0.05)
    assert q1.opex_cr == pytest.approx(1243.22, abs=0.05)
    assert q1.ebitda_cr == pytest.approx(798.04, abs=0.1)


def test_inventory_build_lowers_cost_and_raises_gross_profit():
    # The earlier extraction ignored this line and understated gross profit by about 23 crore here.
    q1 = {r.quarter: r for r in _rows()}["2023Q1"]
    without_inventory = 3952.591 - (1964.311 + 59.616 + 121.73)
    assert q1.gross_profit_cr > without_inventory + 20


def test_margins_land_in_vbl_reported_range():
    for r in _rows():
        assert 48 < r.gross_profit_cr / r.revenue_cr * 100 < 60
        assert 10 < r.ebitda_cr / r.revenue_cr * 100 < 35


def test_merge_prefers_current_quarter_column_and_sorts():
    rows = _rows()
    df = merge_rows(rows + rows)
    assert df["quarter"].tolist() == ["2022Q1", "2022Q4", "2023Q1"]


def test_merge_into_table_keeps_existing_quarters_unless_overwrite():
    new = merge_rows(_rows())
    existing = new.iloc[[2]].copy()
    existing["revenue_cr"] = 1.0
    kept = merge_into_table(existing, new)
    assert kept.loc[kept["quarter"] == "2023Q1", "revenue_cr"].iloc[0] == 1.0
    assert len(kept) == 3
    replaced = merge_into_table(existing, new, overwrite=True)
    assert replaced.loc[replaced["quarter"] == "2023Q1", "revenue_cr"].iloc[0] > 3000


def test_statement_without_units_or_items_is_skipped():
    header = "Statement of consolidated financial results for the quarter ended on 31 March 2023"
    pages = [header + "\nParticulars"]
    assert extract_from_pages(pages, "x.pdf") == []
    assert merge_rows([]).empty
    assert isinstance(merge_into_table(pd.DataFrame(), pd.DataFrame()), pd.DataFrame)


def test_missing_pdf_dependency_message(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ImportError("no pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigurationError):
        read_pdf_pages(Path("x.pdf"))


def test_ocr_damaged_numbers_are_normalised_by_shape():
    # Real examples from the filings' text layer: periods or commas where thousands separators go.
    assert parse_numbers("Total income 24.773.68 25.074.52") == [24773.68, 25074.52]
    assert parse_numbers("Revenue 17,768,57 2.296.48 9.967.60") == [17768.57, 2296.48, 9967.60]
    assert parse_numbers("Changes (331.67) 1.128.75") == [-331.67, 1128.75]


def test_unicode_dashes_count_as_nil_not_as_missing_columns():
    assert parse_numbers("Exceptional items \u2013 \u2014 -") == [0.0, 0.0, 0.0]
    assert parse_numbers("Exceptional items - 665.29") == [0.0, 665.29]


def test_header_typos_do_not_hide_the_statement():
    text = FIXTURE.read_text().replace(
        "consolidated financial results", "consoiidated financial results", 1
    )
    rows = extract_from_pages(split_pages(text), "typo.pdf")
    assert [r.quarter for r in rows] == ["2023Q1", "2022Q4", "2022Q1"]


def test_a_misread_amount_is_rejected_by_the_tie_out_not_merged():
    # Corrupt one expense amount in the first column: that column must be dropped, the others kept.
    text = FIXTURE.read_text().replace("19,643.11", "19,043.11", 1)
    rows = extract_from_pages(split_pages(text), "corrupt.pdf")
    assert "2023Q1" not in [r.quarter for r in rows]
    assert {"2022Q4", "2022Q1"} <= {r.quarter for r in rows}


def test_tie_out_also_protects_the_revenue_line():
    text = FIXTURE.read_text().replace("39,525.91", "39,825.91", 1)
    rows = extract_from_pages(split_pages(text), "corrupt.pdf")
    assert "2023Q1" not in [r.quarter for r in rows]


def test_overwrite_carries_hand_entered_volume_fields():
    new = merge_rows(_rows())
    existing = new.iloc[[2]].copy()
    existing["volume_growth_yoy_pct"] = 16.3
    existing["volume_note"] = "consolidated"
    out = merge_into_table(existing, new, overwrite=True)
    row = out[out["quarter"] == "2023Q1"].iloc[0]
    assert row["volume_growth_yoy_pct"] == 16.3 and row["volume_note"] == "consolidated"
