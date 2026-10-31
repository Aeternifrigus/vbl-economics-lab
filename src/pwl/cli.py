import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from .backtest import rolling_backtest
from .config import DATA_EXT, DATA_RAW, MODELS, REPORTS
from .features import build_table, quarterly_oil_fx, quarterly_weather
from .models.lagged import InsufficientDataError, coef_table, fit_lagged_ridge
from .models.leverage import fit_leverage, render_report, variance_decomposition
from .snapshot import save_snapshot
from .validate import check_vbl, load_vbl

VBL_CSV = DATA_RAW / "vbl_quarterly.csv"
FEATURES_CSV = DATA_EXT / "features_quarterly.csv"
LAG_FEATURES = [
    "brent_inr_lag0",
    "brent_inr_lag1",
    "brent_inr_lag2",
    "brent_inr_lag3",
    "tmax_anom",
    "heat_days",
    "rain_anom",
]


def cmd_validate(args):
    problems = check_vbl(load_vbl(VBL_CSV))
    if problems:
        print("Validation failed:")
        for p in problems:
            print(" -", p)
        return 1
    print("Validation passed.")
    return 0


def cmd_leverage(args):
    df = load_vbl(VBL_CSV)
    problems = check_vbl(df)
    if problems:
        print("Fix data first:", problems)
        return 1
    fit = fit_leverage(df)
    decomp = variance_decomposition(df)
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    latest = str(df["quarter"].iloc[-1])
    payload = {
        "model": "operating_leverage_ols",
        "latest_quarter": latest,
        "data_sha256": hashlib.sha256(VBL_CSV.read_bytes()).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit": fit.to_dict(),
        "variance_decomposition": decomp,
    }
    (MODELS / f"leverage_{latest}.json").write_text(json.dumps(payload, indent=2))
    (REPORTS / "operating_leverage.md").write_text(render_report(df, fit, decomp))
    print(f"n={fit.n} R2={fit.r2:.3f} fixed={fit.fixed_cr:.0f}cr variable={fit.var_cost_ratio * 100:.1f}%")
    print(f"opex ratio share of margin variance: {decomp['share_opex_ratio'] * 100:.1f}%")
    print(f"wrote models/leverage_{latest}.json and reports/operating_leverage.md")
    return 0


def cmd_ingest(args):
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("Set FRED_API_KEY (free key from fred.stlouisfed.org).")
        return 2
    from .ingest import fred, openmeteo

    start = args.start
    end = datetime.now(timezone.utc).date().isoformat()
    brent = fred.fetch_series(fred.BRENT, key, start)
    fx = fred.fetch_series(fred.USDINR, key, start)
    weather = openmeteo.fetch_weighted(start, end)
    save_snapshot(brent, "brent_daily", "FRED DCOILBRENTEU")
    save_snapshot(fx, "usdinr_daily", "FRED DEXINUS")
    save_snapshot(weather, "weather_daily", "Open-Meteo archive, population-weighted")
    print(f"snapshots written: brent={len(brent)} fx={len(fx)} weather={len(weather)} rows")
    return 0


def _read(name):
    path = DATA_EXT / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python -m pwl ingest` first.")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def cmd_features(args):
    vbl = load_vbl(VBL_CSV)
    oil = quarterly_oil_fx(_read("brent_daily"), _read("usdinr_daily"))
    weather = quarterly_weather(_read("weather_daily"))
    table = build_table(vbl, oil, weather)
    out = table.copy()
    out.insert(0, "quarter", out.index.astype(str))
    save_snapshot(out.reset_index(drop=True), "features_quarterly", "built from snapshots and vbl_quarterly.csv")
    print(f"feature table: {len(table)} quarters, {table[LAG_FEATURES].dropna().shape[0]} with all features")
    return 0


def _load_features():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError("features_quarterly.csv missing. Run ingest then features.")
    df = pd.read_csv(FEATURES_CSV)
    df.index = pd.PeriodIndex(df["quarter"], freq="Q")
    return df


def cmd_train(args):
    try:
        df = _load_features()
    except FileNotFoundError as exc:
        print(exc)
        return 2
    try:
        model = fit_lagged_ridge(df, "gm_pct", LAG_FEATURES, force=args.force)
    except InsufficientDataError as exc:
        print("Refusing to train:", exc)
        return 3
    MODELS.mkdir(exist_ok=True)
    latest = str(df.index[-1])
    joblib.dump(model, MODELS / f"lagged_gm_{latest}.joblib")
    table = coef_table(model, LAG_FEATURES)
    meta = {
        "model": "distributed_lag_ridge",
        "target": "gm_pct",
        "latest_quarter": latest,
        "alpha": model.named_steps["ridge"].alpha,
        "data_sha256": hashlib.sha256(FEATURES_CSV.read_bytes()).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coefficients": table.to_dict(orient="records"),
        "forced": bool(args.force),
    }
    (MODELS / f"lagged_gm_{latest}.json").write_text(json.dumps(meta, indent=2))
    print(table.to_string(index=False))
    return 0


def cmd_backtest(args):
    try:
        df = _load_features()
    except FileNotFoundError as exc:
        print(exc)
        return 2
    try:
        _, summary = rolling_backtest(df, "gm_pct", LAG_FEATURES, min_train=args.min_train)
    except InsufficientDataError as exc:
        print("Cannot backtest:", exc)
        return 3
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="pwl")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    sub.add_parser("leverage").set_defaults(fn=cmd_leverage)
    ing = sub.add_parser("ingest")
    ing.add_argument("--start", default="2015-01-01")
    ing.set_defaults(fn=cmd_ingest)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    tr = sub.add_parser("train")
    tr.add_argument("--force", action="store_true")
    tr.set_defaults(fn=cmd_train)
    bt = sub.add_parser("backtest")
    bt.add_argument("--min-train", type=int, default=12)
    bt.set_defaults(fn=cmd_backtest)
    args = p.parse_args(argv)
    return args.fn(args)
