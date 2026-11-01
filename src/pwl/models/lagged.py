"""Distributed-lag ridge model for gross margin on lagged oil cost and weather."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MIN_QUARTERS_FOR_RELEASE
from ..exceptions import InsufficientDataError, ModelTrainingError
from ..logging_config import get_logger

logger = get_logger(__name__)


def fit_lagged_ridge(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    force: bool = False,
) -> Pipeline:
    """Fit a distributed-lag ridge regression with time-series cross-validation."""
    features = list(features)
    d = df[[target] + features].dropna()

    logger.info(
        "Training lagged ridge: %d usable quarters, target=%s, features=%d",
        len(d),
        target,
        len(features),
    )

    if len(d) < MIN_QUARTERS_FOR_RELEASE and not force:
        raise InsufficientDataError(
            f"{len(d)} usable quarters, need at least {MIN_QUARTERS_FOR_RELEASE}. "
            "Extend data/raw/vbl_quarterly.csv or pass --force for a smoke run.",
            available=len(d),
            required=MIN_QUARTERS_FOR_RELEASE,
        )

    n_splits = max(2, min(5, len(d) // 5))
    if len(d) < n_splits + 2:
        raise InsufficientDataError(
            f"{len(d)} usable quarters is too few even for a forced fit",
            available=len(d),
            required=n_splits + 2,
        )

    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipe,
        {"ridge__alpha": list(alphas)},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
    )

    try:
        search.fit(d[features], d[target])
    except Exception as exc:
        raise ModelTrainingError(
            f"Lagged ridge training failed: {exc}",
            model_type="distributed_lag_ridge",
            details={"n_samples": len(d), "n_features": len(features)},
        ) from exc

    logger.info(
        "Best alpha: %.4f, CV MAE: %.4f", search.best_params_["ridge__alpha"], -search.best_score_
    )
    return search.best_estimator_


def coef_table(model: Pipeline, features: list[str]) -> pd.DataFrame:
    """Extract coefficients in standardized and raw space."""
    scale = model.named_steps["scale"].scale_
    coef = model.named_steps["ridge"].coef_
    return pd.DataFrame({"feature": list(features), "coef_std": coef, "coef_raw": coef / scale})


def seasonal_naive(series: pd.Series, period: int = 4) -> pd.Series:
    """Seasonal naive forecast (same quarter previous year)."""
    return series.shift(period)


def cumulative_effect_interval(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    effect_features: list[str],
    alpha: float,
    n_boot: int = 300,
    block: int = 4,
    seed: int = 0,
    level: float = 0.9,
) -> tuple[float, float]:
    """Moving-block bootstrap interval for the summed raw coefficient of `effect_features`.

    Blocks of consecutive quarters keep short-run dependence intact. The ridge penalty is held at
    the value chosen for the main fit so the interval reflects sampling noise, not tuning.
    """
    d = df[[target, *features]].dropna().reset_index(drop=True)
    n = len(d)
    if n < block * 2:
        raise InsufficientDataError(
            f"{n} usable quarters is too few for a block bootstrap", available=n, required=block * 2
        )
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(n_boot):
        rows: list[int] = []
        while len(rows) < n:
            start = int(rng.integers(0, n - block + 1))
            rows.extend(range(start, start + block))
        sample = d.iloc[rows[:n]].reset_index(drop=True)
        model = fit_lagged_ridge(sample, target, features, alphas=(alpha,), force=True)
        table = coef_table(model, features).set_index("feature")["coef_raw"]
        effects.append(float(table.loc[effect_features].sum()))
    lo, hi = np.percentile(effects, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
