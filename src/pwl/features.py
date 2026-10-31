import numpy as np
import pandas as pd


def _expected_days(period_index):
    start = period_index.to_timestamp(how="start")
    end = period_index.to_timestamp(how="end").normalize()
    return ((end - start).days + 1).to_numpy()


def quarterly_weather(daily):
    """Aggregate daily weather to quarters. Anomalies use only earlier years (no leakage)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["period"] = d["date"].dt.to_period("Q")
    g = d.groupby("period").agg(
        tmax_mean=("tmax_c", "mean"),
        rain_mm=("precip_mm", "sum"),
        heat_days=("heat_share", "sum"),
        n_days=("date", "count"),
    )
    g = g[g["n_days"].to_numpy() >= _expected_days(g.index)].drop(columns="n_days")
    qnum = g.index.quarter
    for col, name in [("tmax_mean", "tmax_anom"), ("rain_mm", "rain_anom")]:
        clim = g.groupby(qnum)[col].transform(lambda s: s.shift().expanding().mean())
        g[name] = g[col] - clim
    return g


def quarterly_oil_fx(brent_daily, fx_daily):
    b = brent_daily.rename(columns={"value": "brent_usd"}).set_index("date")
    f = fx_daily.rename(columns={"value": "usdinr"}).set_index("date")
    m = b.join(f, how="left")
    m["usdinr"] = m["usdinr"].ffill()
    m = m.dropna()
    m["brent_inr"] = m["brent_usd"] * m["usdinr"]
    return m.groupby(m.index.to_period("Q")).mean()


def add_lags(df, cols, n_lags):
    out = df.copy()
    for c in cols:
        for k in range(n_lags + 1):
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def build_table(vbl, oil_q, weather_q, n_lags=3):
    t = vbl.set_index("period")[
        ["revenue_cr", "gross_profit_cr", "opex_cr", "ebitda_cr", "volume_growth_yoy_pct"]
    ].copy()
    t["gm_pct"] = t["gross_profit_cr"] / t["revenue_cr"] * 100
    t["opex_ratio_pct"] = t["opex_cr"] / t["revenue_cr"] * 100
    t["opm_pct"] = (t["gross_profit_cr"] - t["opex_cr"]) / t["revenue_cr"] * 100
    # Oil is lagged on the full quarterly index so lags reach before the first company quarter.
    oil_l = add_lags(oil_q, ["brent_inr"], n_lags)
    t = t.join(oil_l, how="left").join(weather_q, how="left")
    return t
