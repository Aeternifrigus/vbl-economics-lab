import numpy as np
import pandas as pd
import pytest

from pwl.backtest import rolling_backtest
from pwl.models.lagged import InsufficientDataError, coef_table, fit_lagged_ridge


def _synthetic(n=80, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005Q1", periods=n, freq="Q")
    oil = rng.normal(0, 1, n + 3)
    df = pd.DataFrame(index=idx)
    for k in range(4):
        df[f"oil_lag{k}"] = oil[3 - k : 3 - k + n]
    df["heat"] = rng.normal(0, 1, n)
    df["gm"] = 55 - 0.2 * df["oil_lag0"] - 1.0 * df["oil_lag1"] - 0.3 * df["oil_lag2"] + 0.5 * df["heat"] + rng.normal(0, 0.3, n)
    return df


FEATS = ["oil_lag0", "oil_lag1", "oil_lag2", "oil_lag3", "heat"]


def test_recovers_lag_structure_from_synthetic_data():
    df = _synthetic()
    model = fit_lagged_ridge(df, "gm", FEATS, alphas=(0.001, 0.01, 0.1))
    coefs = coef_table(model, FEATS).set_index("feature")["coef_raw"]
    assert abs(coefs["oil_lag1"] - (-1.0)) < 0.2
    assert abs(coefs["oil_lag3"]) < 0.2
    assert abs(coefs["heat"] - 0.5) < 0.2
    assert coefs["oil_lag1"] < coefs["oil_lag0"] < 0.0 or coefs["oil_lag1"] < -0.5


def test_refuses_small_samples_unless_forced():
    df = _synthetic(n=10)
    with pytest.raises(InsufficientDataError):
        fit_lagged_ridge(df, "gm", FEATS)
    assert fit_lagged_ridge(df, "gm", FEATS, force=True) is not None


def test_backtest_beats_seasonal_naive_on_signal_data():
    df = _synthetic()
    _, summary = rolling_backtest(df, "gm", FEATS, min_train=30)
    assert summary["mae_model"] < summary["mae_seasonal_naive"]
    assert summary["skill_vs_seasonal_naive"] > 0.3


def test_backtest_refuses_tiny_input():
    with pytest.raises(InsufficientDataError):
        rolling_backtest(_synthetic(n=8), "gm", FEATS, min_train=12)
