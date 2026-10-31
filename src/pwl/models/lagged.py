"""Distributed-lag ridge model for gross margin on lagged oil cost and weather."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MIN_QUARTERS_FOR_RELEASE


class InsufficientDataError(ValueError):
    pass


def fit_lagged_ridge(df, target, features, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), force=False):
    features = list(features)
    d = df[[target] + features].dropna()
    if len(d) < MIN_QUARTERS_FOR_RELEASE and not force:
        raise InsufficientDataError(
            f"{len(d)} usable quarters, need at least {MIN_QUARTERS_FOR_RELEASE}. "
            "Extend data/raw/vbl_quarterly.csv or pass --force for a smoke run."
        )
    n_splits = max(2, min(5, len(d) // 5))
    if len(d) < n_splits + 2:
        raise InsufficientDataError(f"{len(d)} usable quarters is too few even for a forced fit")
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipe,
        {"ridge__alpha": list(alphas)},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
    )
    search.fit(d[features], d[target])
    return search.best_estimator_


def coef_table(model, features):
    scale = model.named_steps["scale"].scale_
    coef = model.named_steps["ridge"].coef_
    return pd.DataFrame(
        {"feature": list(features), "coef_std": coef, "coef_raw": coef / scale}
    )


def seasonal_naive(series, period=4):
    return series.shift(period)
