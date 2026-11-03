"""Nowcast, monitoring and release: the pieces that turn a trained model into an operated system."""

import json

import numpy as np
import pandas as pd
import pytest

from pwl import __version__, cli
from pwl.monitor import input_drift_checks, performance_checks, psi
from pwl.release import MANIFEST_NAME, file_sha256, verify_release


def _run(runner, *args, expect=0):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == expect, result.output
    return result


@pytest.fixture
def trained(runner, sandbox, synthetic_vbl, fake_apis):
    for command in ("ingest", "features", "train", "backtest", "leverage"):
        _run(runner, command)
    return sandbox


def _actual_gm(table, quarter):
    row = table.set_index("quarter").loc[quarter]
    return row["gross_profit_cr"] / (row["revenue_cr"] - row["excise_duty_cr"]) * 100


def test_version_flag(runner):
    result = _run(runner, "--version")
    assert __version__ in result.output


def test_nowcast_matches_the_quarter_and_brackets_it(runner, trained, synthetic_vbl):
    _run(runner, "nowcast", "--quarter", "2026Q2")
    out = json.loads((trained["REPORTS"] / "nowcast.json").read_text())
    actual = _actual_gm(synthetic_vbl, "2026Q2")
    assert out["quarter"] == "2026Q2" and out["quarter_complete"] is True
    assert abs(out["predicted_gm_pct"] - actual) < 1.0
    assert out["extrapolating"] is False and out["outside_training_range"] == []
    low, high = out["interval90"]
    assert low < actual < high
    assert out["seasonal_naive_gm_pct"] == pytest.approx(
        _actual_gm(synthetic_vbl, "2025Q2"), abs=0.01
    )


def test_nowcast_uses_oil_to_date_for_a_partial_quarter(runner, trained):
    for name in ("brent_daily", "usdinr_daily"):
        path = trained["DATA_EXT"] / f"{name}.csv"
        df = pd.read_csv(path)
        df[pd.to_datetime(df["date"]) <= "2026-05-15"].to_csv(path, index=False)
    _run(runner, "nowcast", "--quarter", "2026Q2")
    out = json.loads((trained["REPORTS"] / "nowcast.json").read_text())
    assert out["quarter_complete"] is False
    assert 40 < out["days_of_oil_data"] < 91


def test_nowcast_flags_extrapolation_when_oil_leaves_the_training_range(runner, trained):
    # Push the final quarter's Brent far above anything the model was trained on.
    path = trained["DATA_EXT"] / "brent_daily.csv"
    df = pd.read_csv(path)
    in_q = pd.to_datetime(df["date"]) >= "2026-04-01"
    df.loc[in_q, "value"] += 120
    df.to_csv(path, index=False)
    _run(runner, "nowcast", "--quarter", "2026Q2")
    out = json.loads((trained["REPORTS"] / "nowcast.json").read_text())
    assert out["extrapolating"] is True
    assert "brent_inr_lag0" in out["outside_training_range"]


def test_model_metadata_records_training_ranges_and_error_quantiles(trained):
    meta = json.loads(next(trained["MODELS"].glob("lagged_gm_*.json")).read_text())
    low, high = meta["training_feature_ranges"]["brent_inr_lag0"]
    assert low < high
    assert meta["backtest_error_quantiles"]["q05"] < 0 < meta["backtest_error_quantiles"]["q95"]


def test_nowcast_error_paths(runner, sandbox, synthetic_vbl, fake_apis):
    _run(runner, "ingest")
    _run(runner, "features")
    _run(runner, "nowcast", "--quarter", "2026Q2", expect=2)  # no trained model yet
    _run(runner, "train")
    _run(runner, "nowcast", "--quarter", "2027Q1", expect=3)  # no oil data for that quarter


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(70, 10, 2000)
    assert psi(ref, rng.normal(70, 10, 500)) < 0.05
    assert psi(ref, rng.normal(110, 10, 500)) > 0.25


def _brent(n=1500, recent_shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    values = rng.normal(70, 8, n)
    values[-180:] += recent_shift
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "value": values})


def test_input_drift_flags_a_new_oil_regime_only():
    calm = {c.name: c.status for c in input_drift_checks(_brent())}
    assert set(calm.values()) == {"ok"}
    shifted = {c.name: c.status for c in input_drift_checks(_brent(recent_shift=45))}
    assert shifted["brent_psi"] == "warn" and shifted["brent_out_of_range_share"] == "warn"


def test_performance_drift_flags_a_run_of_large_errors():
    rng = np.random.default_rng(2)
    actual = np.full(20, 55.0)
    steady = pd.DataFrame({"actual": actual, "model": actual + rng.normal(0, 1, 20)})
    assert {c.status for c in performance_checks(steady)} == {"ok"}
    broken = steady.copy()
    broken.loc[broken.index[-4:], "model"] = 55.0 + np.array([6.0, -7.0, 8.0, 9.0])
    statuses = {c.name: c.status for c in performance_checks(broken)}
    assert statuses["recent_error_ratio"] == "warn"


def test_monitor_cli_reports_and_strict_mode_exits_4_on_drift(runner, trained):
    result = _run(runner, "monitor")
    report = json.loads((trained["REPORTS"] / "monitor.json").read_text())
    assert report["status"] in {"ok", "warn"} and report["checks"]
    assert "status:" in result.output

    path = trained["DATA_EXT"] / "brent_daily.csv"
    df = pd.read_csv(path)
    df.loc[df.index[-200:], "value"] += 80
    df.to_csv(path, index=False)
    _run(runner, "monitor", "--strict", expect=4)
    assert json.loads((trained["REPORTS"] / "monitor.json").read_text())["status"] == "warn"


def test_release_bundle_is_verifiable_and_tamper_evident(runner, trained, tmp_path):
    _run(runner, "nowcast", "--quarter", "2026Q2")
    out = tmp_path / "dist"
    _run(runner, "release", "--tag", "vtest", "--out", str(out))
    archive = out / "vbl-economics-lab-vtest.tar.gz"
    assert archive.exists()
    digest = (out / f"{archive.name}.sha256").read_text().split()[0]
    assert digest == file_sha256(archive)

    bundle = out / "vbl-economics-lab-vtest"
    manifest = json.loads((bundle / MANIFEST_NAME).read_text())
    assert manifest["version"] == "vtest" and manifest["model_latest_quarter"] == "2026Q2"
    assert any(name.endswith(".joblib") for name in manifest["files"])
    assert "data/features_quarterly.csv" in manifest["files"]
    assert verify_release(bundle) == []

    (bundle / "data" / "features_quarterly.csv").write_text("tampered\n")
    assert verify_release(bundle) == ["data/features_quarterly.csv: hash mismatch"]
    (bundle / MANIFEST_NAME).unlink()
    assert verify_release(bundle) == [f"{MANIFEST_NAME} is missing"]


def test_release_refuses_a_model_trained_on_different_data(
    runner, trained, synthetic_vbl, tmp_path
):
    synthetic_vbl.iloc[1:].to_csv(trained["VBL_CSV"], index=False)  # drop the first quarter
    _run(runner, "features")  # features change, the model does not
    result = _run(runner, "release", "--tag", "vstale", "--out", str(tmp_path), expect=1)
    assert "stale" in result.output


def test_release_without_a_model_fails_cleanly(runner, sandbox, synthetic_vbl, fake_apis, tmp_path):
    _run(runner, "ingest")
    _run(runner, "features")
    _run(runner, "release", "--tag", "vnone", "--out", str(tmp_path), expect=1)
