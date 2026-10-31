"""Feature engineering for quarterly data."""

import numpy as np
import pandas as pd

from .config import LAG_FEATURES
from .logging_config import get_logger

logger = get_logger(__name__)


def _expected_days(period_index: pd.PeriodIndex) -> np.ndarray:
    """Calculate expected number of days in each quarter."""
    start = period_index.to_timestamp(how="start")
    end = period_index.to_timestamp(how="end").normalize()
    return ((end - start).days + 1).to_numpy()


def quarterly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather to quarters. Anomalies use only earlier years (no leakage)."""
    logger.debug("Aggregating daily weather to quarters: %d rows", len(daily))
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["period"] = d["date"].dt.to_period("Q")

    g = d.groupby("period").agg(
        tmax_mean=("tmax_c", "mean"),
        rain_mm=("precip_mm", "sum"),
        heat_days=("heat_share", "sum"),
        n_days=("date", "count"),
    )

    expected = _expected_days(g.index)
    complete = g["n_days"].to_numpy() >= expected
    g = g[complete].drop(columns="n_days")

    qnum = g.index.quarter
    for col, name in [("tmax_mean", "tmax_anom"), ("rain_mm", "rain_anom")]:
        clim = g.groupby(qnum)[col].transform(lambda s: s.shift().expanding().mean())
        g[name] = g[col] - clim

    logger.info("Quarterly weather: %d complete quarters", len(g))
    return g


def quarterly_oil_fx(brent_daily: pd.DataFrame, fx_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily Brent (USD) and FX to quarterly Brent in INR."""
    logger.debug(
        "Computing quarterly oil/FX: brent=%d fx=%d rows",
        len(brent_daily),
        len(fx_daily),
    )

    b = brent_daily.rename(columns={"value": "brent_usd"}).set_index("date")
    f = fx_daily.rename(columns={"value": "usdinr"}).set_index("date")
    m = b.join(f, how="left")
    m["usdinr"] = m["usdinr"].ffill()
    m = m.dropna()
    m["brent_inr"] = m["brent_usd"] * m["usdinr"]
    result = m.groupby(m.index.to_period("Q")).mean()

    logger.info("Quarterly oil/FX: %d quarters", len(result))
    return result


def add_lags(df: pd.DataFrame, cols: list[str], n_lags: int) -> pd.DataFrame:
    """Add lagged columns to a DataFrame."""
    out = df.copy()
    for c in cols:
        for k in range(n_lags + 1):
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def build_table(
    vbl: pd.DataFrame, oil_q: pd.DataFrame, weather_q: pd.DataFrame, n_lags: int = 3
) -> pd.DataFrame:
    """Build the quarterly feature table from VBL, oil, and weather data."""
    logger.info(
        "Building feature table from VBL (%d), oil (%d), weather (%d)",
        len(vbl),
        len(oil_q),
        len(weather_q),
    )

    t = vbl.set_index("period")[
        ["revenue_cr", "gross_profit_cr", "opex_cr", "ebitda_cr", "volume_growth_yoy_pct"]
    ].copy()
    t["gm_pct"] = t["gross_profit_cr"] / t["revenue_cr"] * 100
    t["opex_ratio_pct"] = t["opex_cr"] / t["revenue_cr"] * 100
    t["opm_pct"] = (t["gross_profit_cr"] - t["opex_cr"]) / t["revenue_cr"] * 100

    # Oil is lagged on the full quarterly index so lags reach before the first company quarter.
    oil_l = add_lags(oil_q, ["brent_inr"], n_lags)
    t = t.join(oil_l, how="left").join(weather_q, how="left")

    logger.info(
        "Feature table built: %d quarters, %d with all features",
        len(t),
        t[LAG_FEATURES].dropna().shape[0],
    )
    return t
