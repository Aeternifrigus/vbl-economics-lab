"""End-to-end tests: ingest, features, train and backtest wired together on a known signal."""

import hashlib
import json

import pandas as pd
import pytest
from synthetic import TRUE_LAG1_COEF

from pwl import cli
from pwl.features import quarterly_weather
from pwl.snapshot import verify_snapshot


def _run(runner, *args, expect=0):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == expect, result.output
    return result


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_pipeline_recovers_the_known_signal(runner, sandbox, synthetic_vbl, fake_apis):
    for command in ("ingest", "features", "train", "backtest"):
        _run(runner, command)

    meta_path = next(sandbox["MODELS"].glob("lagged_gm_*.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["forced"] is False
    coefs = {c["feature"]: c["coef_raw"] for c in meta["coefficients"]}
    # The effect was planted on the one-quarter lag; a misaligned join would put it elsewhere.
    assert coefs["brent_inr_lag1"] == pytest.approx(TRUE_LAG1_COEF, rel=0.5)
    assert abs(coefs["brent_inr_lag1"]) > 3 * abs(coefs["brent_inr_lag3"])
    assert meta["cumulative_brent_inr_effect"] == pytest.approx(TRUE_LAG1_COEF, rel=0.5)

    summary = json.loads((sandbox["REPORTS"] / "backtest.json").read_text())
    assert summary["skill_vs_seasonal_naive"] > 0.3
    assert (sandbox["MODELS"] / f"lagged_gm_{meta['latest_quarter']}.joblib").exists()


def test_refit_runs_every_stage_without_forcing(runner, sandbox, synthetic_vbl, fake_apis):
    _run(runner, "ingest")
    result = _run(runner, "refit")
    assert "[4/4]" in result.output
    assert list(sandbox["REPORTS"].glob("operating_leverage.md"))
    meta = json.loads(next(sandbox["MODELS"].glob("lagged_gm_*.json")).read_text())
    assert meta["forced"] is False


def test_guard_blocks_the_lag_model_on_the_real_table(runner, sandbox, real_vbl, fake_apis):
    _run(runner, "ingest")
    _run(runner, "features")
    train = _run(runner, "train", expect=3)
    assert "Refusing to train" in train.output
    _run(runner, "backtest", expect=3)
    assert not list(sandbox["MODELS"].glob("lagged_gm_*"))


def test_refit_on_real_table_still_refreshes_leverage_then_exits_3(
    runner, sandbox, real_vbl, fake_apis
):
    _run(runner, "ingest")
    result = _run(runner, "refit", expect=3)
    assert "Lag model skipped" in result.output
    assert (sandbox["REPORTS"] / "operating_leverage.md").exists()
    assert list(sandbox["MODELS"].glob("leverage_*.json"))
    assert not list(sandbox["MODELS"].glob("lagged_gm_*"))


def test_snapshots_are_hashed_verifiable_and_deterministic(
    runner, sandbox, synthetic_vbl, fake_apis
):
    _run(runner, "ingest")
    _run(runner, "features")
    manifest = json.loads((sandbox["DATA_EXT"] / "manifest.json").read_text())
    assert set(manifest) == {"brent_daily", "usdinr_daily", "weather_daily", "features_quarterly"}
    for name, entry in manifest.items():
        assert verify_snapshot(name)
        assert entry["sha256"] == _sha(sandbox["DATA_EXT"] / f"{name}.csv")

    first = _sha(sandbox["FEATURES_CSV"])
    _run(runner, "features")
    assert _sha(sandbox["FEATURES_CSV"]) == first

    (sandbox["DATA_EXT"] / "brent_daily.csv").write_text("date,value\n2020-01-01,1\n")
    assert not verify_snapshot("brent_daily")


def test_changing_the_latest_weather_cannot_change_earlier_features(world):
    _, _, weather = world
    base = quarterly_weather(weather)
    altered = weather.copy()
    last_quarter = altered["date"].dt.to_period("Q") == altered["date"].dt.to_period("Q").max()
    altered.loc[last_quarter, ["tmax_c", "precip_mm"]] += 25
    after = quarterly_weather(altered)
    earlier = base.index[:-1]
    pd.testing.assert_frame_equal(base.loc[earlier], after.loc[earlier])
    assert not base.iloc[-1].equals(after.iloc[-1])


def test_features_align_company_quarters_with_oil_and_weather(
    runner, sandbox, synthetic_vbl, fake_apis
):
    _run(runner, "ingest")
    _run(runner, "features")
    table = pd.read_csv(sandbox["FEATURES_CSV"]).set_index("quarter")
    assert list(table.index) == list(synthetic_vbl["quarter"])
    row = table.loc["2020Q3"]
    prev = table.loc["2020Q2"]
    assert row["brent_inr_lag1"] == pytest.approx(prev["brent_inr_lag0"])
    assert table[cli.LAG_FEATURES].dropna().shape[0] >= 30
