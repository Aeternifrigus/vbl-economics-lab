"""Rolling-origin backtesting against baselines.

Baselines are looked up by calendar quarter, not by row position, so gaps in company data cannot
turn "same quarter last year" into some other quarter.
"""

from typing import Any

import pandas as pd

from .exceptions import InsufficientDataError
from .logging_config import get_logger
from .models.lagged import fit_lagged_ridge

logger = get_logger(__name__)


def rolling_backtest(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    min_train: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One-quarter-ahead rolling-origin backtest against seasonal-naive and last-value baselines."""
    features = list(features)
    d = df[[target, *features]].dropna()
    start = max(min_train, 4)
    if len(d) <= start:
        raise InsufficientDataError(
            f"{len(d)} usable quarters, need more than {start} to backtest",
            available=len(d),
            required=start + 1,
        )

    actual_by_quarter = df[target].dropna()
    logger.info("Running rolling backtest: %d candidate test quarters", len(d) - start)

    rows = []
    for t in range(start, len(d)):
        period = d.index[t]
        last_year = actual_by_quarter.get(period - 4)
        last_quarter = actual_by_quarter.get(period - 1)
        if last_year is None or last_quarter is None:
            continue  # a baseline is unavailable, so the comparison would not be like for like
        model = fit_lagged_ridge(d.iloc[:t], target, features, force=True)
        rows.append(
            {
                "period": period,
                "actual": float(d[target].iloc[t]),
                "model": float(model.predict(d.iloc[[t]][features])[0]),
                "seasonal_naive": float(last_year),
                "last_value": float(last_quarter),
            }
        )

    if not rows:
        raise InsufficientDataError(
            "no test quarter has both baselines available",
            available=len(d),
            required=start + 1,
        )

    res = pd.DataFrame(rows).set_index("period")
    mae = {
        c: float((res[c] - res["actual"]).abs().mean())
        for c in ["model", "seasonal_naive", "last_value"]
    }
    skill = 1 - mae["model"] / mae["seasonal_naive"] if mae["seasonal_naive"] else float("nan")
    summary = {
        "n_test_quarters": int(len(res)),
        "mae_model": mae["model"],
        "mae_seasonal_naive": mae["seasonal_naive"],
        "mae_last_value": mae["last_value"],
        "skill_vs_seasonal_naive": skill,
    }
    logger.info(
        "Backtest: MAE model=%.4f seasonal=%.4f last=%.4f skill=%.1f%%",
        mae["model"],
        mae["seasonal_naive"],
        mae["last_value"],
        0.0 if pd.isna(skill) else skill * 100,
    )
    return res, summary
