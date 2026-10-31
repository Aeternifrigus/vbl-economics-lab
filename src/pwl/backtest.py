import pandas as pd

from .models.lagged import InsufficientDataError, fit_lagged_ridge


def rolling_backtest(df, target, features, min_train=12):
    """One-quarter-ahead rolling-origin backtest against seasonal-naive and last-value baselines."""
    features = list(features)
    d = df[[target] + features].dropna()
    start = max(min_train, 4)
    if len(d) <= start:
        raise InsufficientDataError(f"{len(d)} usable quarters, need more than {start} to backtest")
    rows = []
    for t in range(start, len(d)):
        model = fit_lagged_ridge(d.iloc[:t], target, features, force=True)
        pred = float(model.predict(d.iloc[[t]][features])[0])
        rows.append(
            {
                "period": d.index[t],
                "actual": float(d[target].iloc[t]),
                "model": pred,
                "seasonal_naive": float(d[target].iloc[t - 4]),
                "last_value": float(d[target].iloc[t - 1]),
            }
        )
    res = pd.DataFrame(rows).set_index("period")
    mae = {c: float((res[c] - res["actual"]).abs().mean()) for c in ["model", "seasonal_naive", "last_value"]}
    summary = {
        "n_test_quarters": int(len(res)),
        "mae_model": mae["model"],
        "mae_seasonal_naive": mae["seasonal_naive"],
        "mae_last_value": mae["last_value"],
        "skill_vs_seasonal_naive": 1 - mae["model"] / mae["seasonal_naive"] if mae["seasonal_naive"] else float("nan"),
    }
    return res, summary
