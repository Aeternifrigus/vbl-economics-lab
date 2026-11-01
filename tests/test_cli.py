from pathlib import Path

import pandas as pd

from pwl import cli
from pwl.exceptions import IngestionError
from pwl.extract import extract_from_pages, merge_rows, split_pages
from pwl.validate import load_vbl

FIXTURE = Path(__file__).parent / "fixtures" / "vbl_results_2023Q1_statement.txt"


def _invoke(runner, *args, expect=0):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == expect, result.output
    return result


def test_validate_passes_and_strict_fails_on_coverage_gap(runner, sandbox, real_vbl):
    ok = _invoke(runner, "validate")
    assert "Validation passed" in ok.output
    assert "quarters missing" in ok.output
    _invoke(runner, "validate", "--strict", expect=1)


def test_validate_fails_on_broken_table(runner, sandbox, real_vbl):
    df = pd.read_csv(real_vbl)
    df.loc[0, "ebitda_cr"] += 100
    df.to_csv(real_vbl, index=False)
    result = _invoke(runner, "validate", expect=1)
    assert "differs from EBITDA" in result.output


def test_exit_code_is_not_swallowed_into_a_second_error(runner, sandbox, real_vbl):
    df = pd.read_csv(real_vbl)
    df.loc[0, "ebitda_cr"] += 100
    df.to_csv(real_vbl, index=False)
    result = _invoke(runner, "validate", expect=1)
    assert "Error: 1" not in result.output


def test_leverage_writes_model_and_report(runner, sandbox, real_vbl):
    n = len(pd.read_csv(real_vbl))
    result = _invoke(runner, "leverage")
    assert f"n={n}" in result.output
    report = (sandbox["REPORTS"] / "operating_leverage.md").read_text()
    assert f"n = {n} quarters" in report
    assert "Coverage has gaps" in report
    assert "Six quarters" not in report


def test_leverage_refuses_fewer_than_four_quarters(runner, sandbox, real_vbl):
    pd.read_csv(real_vbl).head(3).to_csv(real_vbl, index=False)
    _invoke(runner, "leverage", expect=3)


def test_missing_inputs_exit_2(runner, sandbox, real_vbl):
    _invoke(runner, "features", expect=2)
    _invoke(runner, "train", expect=2)
    _invoke(runner, "backtest", expect=2)
    _invoke(runner, "extract", str(sandbox["MODELS"] / "nope"), expect=2)


def test_ingest_requires_a_key(runner, sandbox, monkeypatch):
    monkeypatch.setattr(cli.settings, "fred_api_key", None)
    result = _invoke(runner, "ingest", expect=1)
    assert "FRED_API_KEY" in result.output


def test_ingest_failure_is_reported(runner, sandbox, monkeypatch):
    def boom(*args, **kwargs):
        raise IngestionError("GET failed after 3 tries", source="api.stlouisfed.org")

    monkeypatch.setattr(cli.settings, "fred_api_key", "k")
    monkeypatch.setattr(cli.fred, "fetch_series", boom)
    result = _invoke(runner, "ingest", expect=1)
    assert "Ingestion failed" in result.output


def _fake_extraction(monkeypatch):
    rows = merge_rows(extract_from_pages(split_pages(FIXTURE.read_text()), "1-2.-Results.pdf"))
    monkeypatch.setattr(cli, "extract_directory", lambda path, manual=None, ocr=False: rows)


def test_extract_dry_run_then_merge(runner, sandbox, real_vbl, monkeypatch, tmp_path):
    _fake_extraction(monkeypatch)
    before = real_vbl.read_text()
    dry = _invoke(runner, "extract", str(tmp_path), "--into", str(real_vbl), "--dry-run")
    assert "0 new" in dry.output  # the three fixture quarters are already in the table
    assert real_vbl.read_text() == before

    trimmed = pd.read_csv(real_vbl)
    trimmed = trimmed[~trimmed["quarter"].isin(["2022Q1", "2022Q4", "2023Q1"])]
    trimmed.to_csv(real_vbl, index=False)
    merged = _invoke(runner, "extract", str(tmp_path), "--into", str(real_vbl))
    assert "3 new" in merged.output
    assert {"2022Q1", "2022Q4", "2023Q1"} <= set(load_vbl(real_vbl)["quarter"])


def test_extract_with_no_statements_fails(runner, sandbox, real_vbl, monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "extract_directory", lambda path, manual=None, ocr=False: pd.DataFrame()
    )
    _invoke(runner, "extract", str(tmp_path), "--into", str(real_vbl), expect=1)


def test_claims_table_is_well_formed():
    claims = pd.read_csv(Path(__file__).parents[1] / "claims" / "claims.csv", keep_default_na=False)
    assert list(claims.columns) == [
        "claim_id",
        "claim",
        "report_value",
        "evidence",
        "verdict",
        "source",
    ]
    assert claims["claim_id"].is_unique and len(claims) == 15
    assert (claims["verdict"] != "").all()
