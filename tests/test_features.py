import numpy as np
import pandas as pd

from pwl.features import add_lags, quarterly_oil_fx, quarterly_weather


def _daily(years=5):
    dates = pd.date_range("2019-01-01", periods=365 * years + 2, freq="D")
    rng = np.random.default_rng(1)
    tmax = (
        30 + 8 * np.sin((dates.day_of_year - 80) / 365 * 2 * np.pi) + rng.normal(0, 1, len(dates))
    )
    return pd.DataFrame(
        {
            "date": dates,
            "tmax_c": tmax,
            "precip_mm": rng.gamma(1.0, 2.0, len(dates)),
            "heat_share": (tmax >= 38).astype(float),
        }
    )


def test_incomplete_quarters_are_dropped():
    d = _daily()
    q = quarterly_weather(d)
    assert q.index.max() < pd.Period(d["date"].max(), "Q") or d["date"].max().is_quarter_end


def test_anomaly_uses_only_earlier_years():
    q = quarterly_weather(_daily())
    first_q = q[q.index.quarter == 2].iloc[0]
    assert np.isnan(first_q["tmax_anom"])
    later = q[q.index.quarter == 2].iloc[1]
    prev = q[q.index.quarter == 2].iloc[0]["tmax_mean"]
    assert abs(later["tmax_anom"] - (later["tmax_mean"] - prev)) < 1e-9


def test_oil_in_inr_and_forward_filled_fx():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    brent = pd.DataFrame({"date": dates, "value": 80.0})
    fx = pd.DataFrame({"date": dates[::7], "value": 83.0})
    q = quarterly_oil_fx(brent, fx)
    assert abs(q["brent_inr"].iloc[0] - 80 * 83) < 1e-9


def test_lags_shift_correctly():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = add_lags(df, ["x"], 2)
    assert out["x_lag1"].tolist()[1:] == [1.0, 2.0, 3.0]
    assert np.isnan(out["x_lag2"].iloc[1])
