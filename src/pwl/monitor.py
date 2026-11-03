"""Drift and performance monitoring for the gross margin model.

Two kinds of checks, both cheap and explainable:

- Input drift: has Brent moved into a regime the model was not trained on? Measured with the
  population stability index (PSI) of the last 180 days against all earlier history, and the share
  of recent days outside the historical range.
- Performance drift: are the latest one-quarter-ahead errors larger than usual? Measured against
  the model's own rolling backtest.

A warning is information, not an error: the model may still be fine, but a person should look.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

RECENT_DAYS = 180
PSI_WARN = 0.25  # conventional threshold for a significant shift
OUT_OF_RANGE_WARN = 0.20
ERROR_RATIO_WARN = 1.5
ERROR_Z_WARN = 2.0


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    status: str  # "ok" or "warn"
    detail: str


def psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    """Population stability index of `recent` against `reference` using reference deciles."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    new_share = np.histogram(recent, bins=edges)[0] / len(recent)
    eps = 1e-4
    ref_share, new_share = np.clip(ref_share, eps, None), np.clip(new_share, eps, None)
    return float(np.sum((new_share - ref_share) * np.log(new_share / ref_share)))


def input_drift_checks(brent_daily: pd.DataFrame, recent_days: int = RECENT_DAYS) -> list[Check]:
    """Compare the most recent days of Brent (USD) with all earlier history."""
    series = brent_daily.sort_values("date")["value"].to_numpy(dtype=float)
    if len(series) < recent_days * 3:
        return [Check("brent_input_drift", float("nan"), PSI_WARN, "ok", "not enough history")]
    reference, recent = series[:-recent_days], series[-recent_days:]
    shift = psi(reference, recent)
    outside = float(np.mean((recent < reference.min()) | (recent > reference.max())))
    return [
        Check(
            "brent_psi",
            round(shift, 3),
            PSI_WARN,
            "warn" if shift > PSI_WARN else "ok",
            f"PSI of the last {recent_days} days of Brent against earlier history",
        ),
        Check(
            "brent_out_of_range_share",
            round(outside, 3),
            OUT_OF_RANGE_WARN,
            "warn" if outside > OUT_OF_RANGE_WARN else "ok",
            "share of recent days outside the historical Brent range",
        ),
    ]


def performance_checks(backtest: pd.DataFrame) -> list[Check]:
    """Compare the latest backtest errors with the model's usual error."""
    errors = (backtest["model"] - backtest["actual"]).to_numpy(dtype=float)
    if len(errors) < 8:
        return [Check("recent_error_ratio", float("nan"), ERROR_RATIO_WARN, "ok", "too few errors")]
    overall = float(np.mean(np.abs(errors)))
    recent = float(np.mean(np.abs(errors[-4:])))
    ratio = recent / overall if overall else float("nan")
    history = errors[:-1]
    z = float((errors[-1] - history.mean()) / history.std(ddof=1)) if history.std(ddof=1) else 0.0
    return [
        Check(
            "recent_error_ratio",
            round(ratio, 3),
            ERROR_RATIO_WARN,
            "warn" if ratio > ERROR_RATIO_WARN else "ok",
            "mean absolute error of the last 4 quarters over the backtest average",
        ),
        Check(
            "latest_error_zscore",
            round(z, 3),
            ERROR_Z_WARN,
            "warn" if abs(z) > ERROR_Z_WARN else "ok",
            "latest one-quarter-ahead error against the earlier error distribution",
        ),
    ]


def run_monitor(brent_daily: pd.DataFrame, backtest: pd.DataFrame) -> dict[str, Any]:
    checks = [*input_drift_checks(brent_daily), *performance_checks(backtest)]
    status = "warn" if any(c.status == "warn" for c in checks) else "ok"
    for c in checks:
        if c.status == "warn":
            logger.warning("monitor: %s = %s exceeds %s", c.name, c.value, c.threshold)
    return {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": [asdict(c) for c in checks],
    }
