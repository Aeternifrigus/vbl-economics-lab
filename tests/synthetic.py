"""Synthetic world with a known signal, used to test that the pipeline preserves it end to end.

Gross margin depends on Brent (in INR) lagged one quarter, with a known coefficient. If the join
between company quarters, oil quarters and weather quarters were misaligned, the recovered
coefficient would be wrong, not just the code failing to run.
"""

import numpy as np
import pandas as pd

from pwl.features import quarterly_oil_fx

TRUE_LAG1_COEF = -0.002  # gross margin points per INR of lagged Brent
FIRST_QUARTER = "2017Q1"
LAST_QUARTER = "2026Q2"


def daily_inputs(seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2026-06-30", freq="D")
    n = len(dates)
    # Quarter-level shocks keep the lags from being collinear, as a smooth series would make them.
    quarters = dates.to_period("Q")
    shocks = pd.Series(rng.normal(0, 12, quarters.nunique()), index=quarters.unique())
    level = shocks.reindex(quarters).to_numpy()
    brent = pd.DataFrame({"date": dates, "value": 75 + level + rng.normal(0, 2, n)})
    fx = pd.DataFrame({"date": dates, "value": np.full(n, 80.0)})
    tmax = 31 + 8 * np.sin((dates.day_of_year - 80) / 365 * 2 * np.pi) + rng.normal(0, 1.5, n)
    weather = pd.DataFrame(
        {
            "date": dates,
            "tmax_c": tmax,
            "precip_mm": rng.gamma(1.0, 3.0, n),
            "heat_share": (tmax >= 38).astype(float),
            "n_cities": 10,
        }
    )
    return brent, fx, weather


def vbl_table(brent, fx, n_quarters=None, seed=11):
    rng = np.random.default_rng(seed)
    periods = pd.period_range(FIRST_QUARTER, LAST_QUARTER, freq="Q")
    if n_quarters:
        periods = periods[-n_quarters:]
    oil = quarterly_oil_fx(brent, fx)["brent_inr"]
    lag1 = oil.shift(1).reindex(periods)
    gm = 56 + TRUE_LAG1_COEF * (lag1 - lag1.mean()) + rng.normal(0, 0.25, len(periods))
    season = np.array([0.85, 1.25, 0.95, 0.75])[periods.quarter - 1]
    revenue = (4000 + 60 * np.arange(len(periods))) * season
    excise = 0.02 * revenue
    net = revenue - excise
    opex = 700 + 0.12 * net
    gross = gm.to_numpy() / 100 * net
    return pd.DataFrame(
        {
            "quarter": [str(p) for p in periods],
            "revenue_cr": revenue.round(1),
            "excise_duty_cr": excise.round(1),
            "gross_profit_cr": gross.round(1),
            "opex_cr": opex.round(1),
            "ebitda_cr": (gross - opex).round(1),
            "volume_growth_yoy_pct": np.nan,
            "volume_note": "",
            "source": "file:synthetic",
        }
    )
