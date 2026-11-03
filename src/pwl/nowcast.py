"""In-quarter nowcast of gross margin from oil to date and the trained model.

VBL reports a quarter about a month after it ends, but oil and FX are known daily. Once a quarter
has started, the model's inputs are already available: lag 0 is the average Brent in INR so far,
and lags 1 to 3 are the three completed quarters before it. The result is a running estimate that
sharpens as the quarter fills in, with an interval taken from the model's own backtest errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import LAG_FEATURES
from .exceptions import InsufficientDataError
from .features import quarterly_oil_fx
from .logging_config import get_logger

logger = get_logger(__name__)

N_LAGS = 3
COMPLETE_WITHIN_DAYS = 4  # daily oil data lags the calendar by a few days


@dataclass
class Nowcast:
    quarter: str
    predicted_gm_pct: float
    interval90: list[float] | None
    extrapolating: bool
    outside_training_range: list[str]
    brent_usd_to_date: float
    days_of_oil_data: int
    quarter_complete: bool
    seasonal_naive_gm_pct: float | None
    model_latest_quarter: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nowcast_row(
    brent_daily: pd.DataFrame, fx_daily: pd.DataFrame, quarter: pd.Period
) -> tuple[pd.DataFrame, int, bool, float]:
    """Build the one-row feature frame for `quarter` from daily oil and FX.

    Returns (features, days of oil data in the quarter, quarter complete, Brent USD to date).
    """
    oil_q = quarterly_oil_fx(brent_daily, fx_daily)
    brent_inr = dict(zip(oil_q.index, oil_q["brent_inr"], strict=True))
    brent_usd = dict(zip(oil_q.index, oil_q["brent_usd"], strict=True))
    inputs = {}
    for lag in range(N_LAGS + 1):
        period = quarter - lag
        if period not in brent_inr:
            raise InsufficientDataError(
                f"no oil data for {period}, needed for the {quarter} nowcast",
                available=0,
                required=1,
            )
        inputs[f"brent_inr_lag{lag}"] = float(brent_inr[period])
    for q in (2, 3, 4):
        inputs[f"q{q}"] = float(quarter.quarter == q)

    dates = pd.to_datetime(brent_daily["date"])
    in_quarter = dates[(dates >= quarter.start_time) & (dates <= quarter.end_time)]
    last = in_quarter.max()
    complete = bool(last >= quarter.end_time.normalize() - pd.Timedelta(days=COMPLETE_WITHIN_DAYS))
    frame = pd.DataFrame([inputs])[[c for c in LAG_FEATURES if c in inputs]]
    return frame, int(len(in_quarter)), complete, float(brent_usd[quarter])


def run_nowcast(
    model: Any,
    meta: dict[str, Any],
    brent_daily: pd.DataFrame,
    fx_daily: pd.DataFrame,
    quarter: pd.Period,
    actual_gm: pd.Series | None = None,
) -> Nowcast:
    """Predict the quarter's gross margin and attach a backtest-based 90% interval."""
    frame, days, complete, brent_usd = nowcast_row(brent_daily, fx_daily, quarter)
    predicted = float(model.predict(frame)[0])
    ranges = meta.get("training_feature_ranges", {})
    outside = [
        name
        for name, (low, high) in ranges.items()
        if name in frame.columns and not low <= float(frame[name].iloc[0]) <= high
    ]
    if outside:
        logger.warning("Nowcast %s extrapolates beyond the training range: %s", quarter, outside)
    q = meta.get("backtest_error_quantiles")
    interval = None if not q else [round(predicted + q["q05"], 2), round(predicted + q["q95"], 2)]
    naive = None
    if actual_gm is not None:
        last_year = dict(zip(actual_gm.index, actual_gm, strict=True)).get(quarter - 4)
        naive = None if last_year is None else round(float(last_year), 2)
    logger.info("Nowcast %s: %.2f (days of data %d)", quarter, predicted, days)
    return Nowcast(
        quarter=str(quarter),
        predicted_gm_pct=round(predicted, 2),
        interval90=interval,
        extrapolating=bool(outside),
        outside_training_range=outside,
        brent_usd_to_date=round(brent_usd, 2),
        days_of_oil_data=days,
        quarter_complete=complete,
        seasonal_naive_gm_pct=naive,
        model_latest_quarter=str(meta.get("latest_quarter", "")),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
