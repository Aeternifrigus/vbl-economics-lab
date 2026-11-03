"""Command-line interface for vbl-economics-lab.

Exit codes: 0 success, 1 error or validation failure, 2 missing input file,
3 not enough data for the requested model.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import joblib
import pandas as pd
import typer

from . import __version__
from .backtest import rolling_backtest
from .config import ABLATIONS, DATA_EXT, DATA_RAW, LAG_FEATURES, MODELS, REPORTS
from .exceptions import (
    ConfigurationError,
    IngestionError,
    InsufficientDataError,
    ModelTrainingError,
    PWLError,
    ReleaseError,
)
from .extract import extract_directory, merge_into_table
from .features import build_table, quarterly_oil_fx, quarterly_weather
from .ingest import fred, openmeteo
from .logging_config import setup_logging
from .models.lagged import coef_table, cumulative_effect_interval, fit_lagged_ridge
from .models.leverage import fit_leverage, render_report, variance_decomposition
from .monitor import run_monitor
from .nowcast import run_nowcast
from .release import build_release, latest_model_files, verify_release
from .settings import settings
from .snapshot import save_snapshot
from .validate import check_vbl, load_vbl, vbl_warnings

app = typer.Typer(
    name="pwl",
    help="VBL Economics Lab: packaged-water economics narrative vs VBL filings, oil, FX, weather",
    add_completion=False,
)

REPO_ROOT = DATA_RAW.parents[1]
VBL_CSV = DATA_RAW / "vbl_quarterly.csv"
MANUAL_CSV = DATA_RAW / "manual_statements.csv"
FEATURES_CSV = DATA_EXT / "features_quarterly.csv"
TARGET = "gm_pct"

EXIT_ERROR = 1
EXIT_MISSING = 2
EXIT_INSUFFICIENT = 3
EXIT_DRIFT = 4


def _fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_snapshot(name: str) -> pd.DataFrame:
    path = DATA_EXT / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `pwl ingest` first.")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            "features_quarterly.csv missing. Run `pwl ingest` then `pwl features`."
        )
    df = pd.read_csv(FEATURES_CSV)
    df.index = pd.PeriodIndex(df["quarter"], freq="Q")
    return df


def _validated_vbl() -> pd.DataFrame:
    """Load the VBL table, print warnings, and exit with code 1 if it has errors."""
    df = load_vbl(VBL_CSV)
    problems = check_vbl(df)
    if problems:
        typer.echo("Validation failed:", err=True)
        for p in problems:
            typer.echo(f" - {p}", err=True)
        raise typer.Exit(EXIT_ERROR)
    for w in vbl_warnings(df):
        typer.echo(f"warning: {w}", err=True)
    return df


def _fit_leverage_and_report(df: pd.DataFrame) -> None:
    fit = fit_leverage(df)
    decomp = variance_decomposition(df)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    latest = str(df["quarter"].iloc[-1])
    payload = {
        "model": "operating_leverage_ols",
        "latest_quarter": latest,
        "n_quarters": fit.n,
        "data_sha256": hashlib.sha256(VBL_CSV.read_bytes()).hexdigest(),
        "trained_at": _now(),
        "fit": fit.to_dict(),
        "variance_decomposition": decomp,
    }
    model_path = MODELS / f"leverage_{latest}.json"
    model_path.write_text(json.dumps(payload, indent=2))
    report_path = REPORTS / "operating_leverage.md"
    report_path.write_text(render_report(df, fit, decomp))
    typer.echo(
        f"n={fit.n} R2={fit.r2:.3f} fixed={fit.fixed_cr:.0f}cr "
        f"variable={fit.var_cost_ratio * 100:.1f}%"
    )
    typer.echo(f"opex ratio share of margin variance: {decomp['share_opex_ratio'] * 100:.1f}%")
    typer.echo(f"wrote {model_path} and {report_path}")


def _build_features() -> pd.DataFrame:
    vbl = load_vbl(VBL_CSV)
    oil = quarterly_oil_fx(_read_snapshot("brent_daily"), _read_snapshot("usdinr_daily"))
    weather = quarterly_weather(_read_snapshot("weather_daily"))
    table = build_table(vbl, oil, weather)
    out = table.copy()
    out.insert(0, "quarter", out.index.astype(str))
    save_snapshot(
        out.reset_index(drop=True),
        "features_quarterly",
        "built from snapshots and vbl_quarterly.csv",
    )
    complete = table[LAG_FEATURES].dropna().shape[0]
    typer.echo(f"Feature table: {len(table)} quarters, {complete} with all features")
    return table


def _backtest_error_quantiles(df: pd.DataFrame) -> dict[str, float] | None:
    """5th, 50th and 95th percentile of (actual - predicted) over the rolling backtest.

    Used to put an empirical 90% interval around nowcasts. None when there are too few quarters.
    """
    try:
        result, _ = rolling_backtest(df, TARGET, LAG_FEATURES, min_train=12)
    except InsufficientDataError:
        return None
    errors = (result["actual"] - result["model"]).to_numpy(dtype=float)
    q05, q50, q95 = (float(v) for v in pd.Series(errors).quantile([0.05, 0.5, 0.95]))
    return {"q05": q05, "q50": q50, "q95": q95, "n": float(len(errors))}


def _train(df: pd.DataFrame, force: bool) -> None:
    model = fit_lagged_ridge(df, TARGET, LAG_FEATURES, force=force)
    MODELS.mkdir(parents=True, exist_ok=True)
    latest = str(df.index[-1])
    joblib.dump(model, MODELS / f"lagged_gm_{latest}.joblib")
    table = coef_table(model, LAG_FEATURES)
    # Oil lags are highly correlated in real data, so individual lag coefficients are unstable.
    # The cumulative effect across lags is the number to quote.
    oil_features = [f for f in LAG_FEATURES if f.startswith("brent_inr_lag")]
    oil_rows = table[table["feature"].isin(oil_features)]
    alpha = float(model.named_steps["ridge"].alpha)
    effect = float(oil_rows["coef_raw"].sum())
    error_quantiles = _backtest_error_quantiles(df)
    continuous = [f for f in LAG_FEATURES if f.startswith("brent_inr_lag")]
    used = df[continuous].dropna()
    ranges = {f: [float(used[f].min()), float(used[f].max())] for f in continuous}
    low, high = cumulative_effect_interval(
        df, TARGET, LAG_FEATURES, oil_features, alpha, n_boot=settings.bootstrap_samples
    )
    # Translate to something readable: gross margin points per 10 USD/bbl, at the average USD/INR.
    rate = float(df["usdinr"].mean()) if "usdinr" in df.columns else None
    per_10usd = None if rate is None else [round(v * 10 * rate, 3) for v in (effect, low, high)]
    meta = {
        "model": "distributed_lag_ridge",
        "target": TARGET,
        "latest_quarter": latest,
        "alpha": model.named_steps["ridge"].alpha,
        "data_sha256": hashlib.sha256(FEATURES_CSV.read_bytes()).hexdigest(),
        "trained_at": _now(),
        "coefficients": table.to_dict(orient="records"),
        "backtest_error_quantiles": error_quantiles,
        "training_feature_ranges": ranges,
        "cumulative_brent_inr_effect": effect,
        "cumulative_brent_inr_effect_ci90": [low, high],
        "gm_points_per_10usd_brent": None
        if per_10usd is None
        else {"estimate": per_10usd[0], "ci90": [per_10usd[1], per_10usd[2]]},
        "forced": force,
    }
    (MODELS / f"lagged_gm_{latest}.json").write_text(json.dumps(meta, indent=2))
    typer.echo(table.to_string(index=False))


def _backtest(df: pd.DataFrame, min_train: int) -> None:
    _, summary = rolling_backtest(df, TARGET, LAG_FEATURES, min_train=min_train)
    ablation = {}
    for name, feature_set in ABLATIONS.items():
        _, result = rolling_backtest(df, TARGET, feature_set, min_train=min_train)
        ablation[name] = {
            "mae": result["mae_model"],
            "skill_vs_seasonal_naive": result["skill_vs_seasonal_naive"],
        }
    summary["ablation"] = ablation
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vbl-economics-lab {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version"
    ),
) -> None:
    """VBL Economics Lab."""


@app.command()
def validate(
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as failures"),
) -> None:
    """Validate the VBL quarterly table."""
    setup_logging()
    df = _validated_vbl()
    if strict and vbl_warnings(df):
        _fail("Validation failed in strict mode (warnings above).")
    typer.echo("Validation passed.")


@app.command()
def extract(
    pdf_dir: Path = typer.Argument(..., help="Folder of VBL results PDFs"),
    into: Path = typer.Option(VBL_CSV, "--into", help="Table to merge new quarters into"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace quarters already present"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change, write nothing"
    ),
    manual: Path = typer.Option(MANUAL_CSV, "--manual", help="Hand-entered statements for scans"),
    ocr: bool = typer.Option(
        False, "--ocr", help="OCR scanned pages (slow, rarely passes tie-out)"
    ),
) -> None:
    """Parse consolidated statements from PDFs into reconciled line items and merge them."""
    setup_logging()
    if not pdf_dir.is_dir():
        _fail(f"{pdf_dir} is not a directory.", EXIT_MISSING)
    try:
        new = extract_directory(pdf_dir, manual=manual, ocr=ocr)
    except ConfigurationError as exc:
        _fail(exc.message)
    if new.empty:
        _fail("No consolidated statements found in the PDFs.")
    existing = load_vbl(into).drop(columns="period") if into.exists() else new.iloc[0:0]
    combined = merge_into_table(existing, new, overwrite=overwrite)
    added = sorted(set(combined["quarter"]) - set(existing["quarter"]))
    typer.echo(f"Parsed {len(new)} quarters; {len(added)} new: {', '.join(added) or 'none'}")
    problems = check_vbl(combined)
    if problems:
        for p in problems:
            typer.echo(f" - {p}", err=True)
        _fail("Merged table failed validation, nothing written.")
    for w in vbl_warnings(combined):
        typer.echo(f"warning: {w}", err=True)
    if dry_run:
        typer.echo("Dry run: nothing written.")
        return
    combined.to_csv(into, index=False)
    typer.echo(f"Wrote {into}")


@app.command()
def leverage() -> None:
    """Fit the operating leverage model and write the report."""
    setup_logging()
    df = _validated_vbl()
    try:
        _fit_leverage_and_report(df)
    except InsufficientDataError as exc:
        _fail(f"Cannot fit: {exc.message}", EXIT_INSUFFICIENT)


@app.command()
def ingest(
    start: str = typer.Option(settings.ingest_start_date, help="Start date for data ingestion"),
) -> None:
    """Ingest oil, FX and weather data from external APIs."""
    setup_logging()
    key = settings.fred_api_key
    if not key:
        _fail("FRED_API_KEY not set. Get a free key from fred.stlouisfed.org and export it.")
    end = datetime.now(UTC).date().isoformat()
    typer.echo(f"Ingesting data from {start} to {end}...")
    try:
        brent = fred.fetch_series(fred.BRENT, key, start)
        fx = fred.fetch_series(fred.USDINR, key, start)
        weather = openmeteo.fetch_weighted(start, end)
    except IngestionError as exc:
        _fail(f"Ingestion failed ({exc.source}): {exc.message}")
    save_snapshot(brent, "brent_daily", "FRED DCOILBRENTEU")
    save_snapshot(fx, "usdinr_daily", "FRED DEXINUS")
    save_snapshot(weather, "weather_daily", "Open-Meteo archive, population-weighted")
    typer.echo(f"Snapshots written: brent={len(brent)} fx={len(fx)} weather={len(weather)} rows")


@app.command()
def features() -> None:
    """Build the quarterly feature table from snapshots and the VBL table."""
    setup_logging()
    _validated_vbl()
    try:
        _build_features()
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)


@app.command()
def train(
    force: bool = typer.Option(False, "--force", help="Smoke run below the minimum quarter count"),
) -> None:
    """Train the distributed-lag ridge model on gross margin."""
    setup_logging()
    try:
        _train(_load_features(), force)
    except InsufficientDataError as exc:
        _fail(f"Refusing to train: {exc.message}", EXIT_INSUFFICIENT)
    except ModelTrainingError as exc:
        _fail(f"Training failed: {exc.message}")
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)


@app.command()
def backtest(
    min_train: int = typer.Option(12, "--min-train", help="Minimum training quarters"),
) -> None:
    """Run the rolling-origin backtest against baselines."""
    setup_logging()
    try:
        _backtest(_load_features(), min_train)
    except InsufficientDataError as exc:
        _fail(f"Cannot backtest: {exc.message}", EXIT_INSUFFICIENT)
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)


@app.command()
def nowcast(
    quarter: str = typer.Option(
        "", "--quarter", help="Quarter to nowcast, e.g. 2026Q4 (default: now)"
    ),
) -> None:
    """Estimate a quarter's gross margin from oil to date and the trained model."""
    setup_logging()
    try:
        model_path, meta_path = latest_model_files(MODELS)
        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text())
        target = pd.Period(quarter or datetime.now(UTC).date(), freq="Q")
        features_df = _load_features()
        result = run_nowcast(
            model,
            meta,
            _read_snapshot("brent_daily"),
            _read_snapshot("usdinr_daily"),
            target,
            actual_gm=features_df[TARGET],
        )
    except ReleaseError as exc:
        _fail(exc.message, EXIT_MISSING)
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)
    except InsufficientDataError as exc:
        _fail(f"Cannot nowcast: {exc.message}", EXIT_INSUFFICIENT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "nowcast.json").write_text(json.dumps(result.to_dict(), indent=2))
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def monitor(
    strict: bool = typer.Option(False, "--strict", help="Exit 4 when any check warns"),
) -> None:
    """Check for input drift (Brent regime) and performance drift (recent model errors)."""
    setup_logging()
    try:
        features_df = _load_features()
        brent = _read_snapshot("brent_daily")
        backtest_df, _ = rolling_backtest(features_df, TARGET, LAG_FEATURES, min_train=12)
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)
    except InsufficientDataError as exc:
        _fail(f"Cannot monitor: {exc.message}", EXIT_INSUFFICIENT)
    report = run_monitor(brent, backtest_df)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "monitor.json").write_text(json.dumps(report, indent=2))
    typer.echo(f"status: {report['status']}")
    for check in report["checks"]:
        typer.echo(
            f"  [{check['status']:>4}] {check['name']} = {check['value']} "
            f"(limit {check['threshold']})"
        )
    if strict and report["status"] != "ok":
        raise typer.Exit(EXIT_DRIFT)


@app.command()
def release(
    tag: str = typer.Option(..., "--tag", help="Release label, e.g. v2026Q3"),
    out: Path = typer.Option(Path("dist"), "--out", help="Where to write the release"),
) -> None:
    """Bundle the trained model, its data, evaluation and model card with verified hashes."""
    setup_logging()
    try:
        archive = build_release(
            tag,
            MODELS,
            FEATURES_CSV,
            VBL_CSV,
            REPORTS,
            [REPO_ROOT / "MODEL_CARD.md", REPO_ROOT / "LICENSE"],
            out,
            REPO_ROOT,
        )
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)
    except ReleaseError as exc:
        _fail(f"Cannot release: {exc.message}")
    problems = verify_release(archive.with_name(archive.name.removesuffix(".tar.gz")))
    if problems:
        _fail("Release failed its own verification: " + "; ".join(problems))
    typer.echo(f"Wrote {archive} and {archive}.sha256")


@app.command()
def refit() -> None:
    """Full pipeline: validate, leverage, features, train, backtest. Never forces training."""
    setup_logging()
    df = _validated_vbl()
    typer.echo("[1/4] validation passed")
    try:
        _fit_leverage_and_report(df)
    except InsufficientDataError as exc:
        _fail(f"Cannot fit leverage: {exc.message}", EXIT_INSUFFICIENT)
    typer.echo("[2/4] leverage model refreshed")
    try:
        _build_features()
    except FileNotFoundError as exc:
        _fail(str(exc), EXIT_MISSING)
    typer.echo("[3/4] features built")
    try:
        features_df = _load_features()
        _train(features_df, force=False)
        _backtest(features_df, 12)
    except InsufficientDataError as exc:
        _fail(
            f"Lag model skipped: {exc.message}\n"
            "The leverage model and report were still refreshed.",
            EXIT_INSUFFICIENT,
        )
    except PWLError as exc:
        _fail(f"Error: {exc.message}")
    typer.echo("[4/4] lag model trained and backtested")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
