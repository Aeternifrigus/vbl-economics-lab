"""Build the data file behind the project website (docs/index.html).

The page draws every chart from `docs/data.js`, which this module writes from the repository's own
outputs (feature table, claims, backtest, model metadata, nowcast and monitor reports). Nothing on
the page is typed in by hand, so it cannot drift from the results: rerun `pwl site` and commit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .backtest import rolling_backtest
from .config import LAG_FEATURES
from .exceptions import InsufficientDataError

REPO_URL = "https://github.com/Aeternifrigus/vbl-economics-lab"
TARGET = "gm_pct"

# What the original industry reports claimed, in the reports' own words and numbers.
REPORT_CLAIMS = {
    "gm_low_oil_pct": 52.0,  # "Brent below 70 pushes gross margin past 52%"
    "gm_high_oil_pct": 38.0,  # "Brent above 90 pushes gross margin toward 38%"
    "oil_low_usd": 70.0,
    "oil_high_usd": 90.0,
    "ebitda_by_quarter": {"Q1": 14.2, "Q2": 21.8, "Q3": 13.5, "Q4": 16.0},
}


def implied_points_per_10usd(claims: dict[str, Any] = REPORT_CLAIMS) -> float:
    """Gross margin points per 10 USD of Brent implied by the reports' two anchor claims.

    A fall from 52% to 38% over a rise from 70 to 90 USD is 14 points over 20 USD, so 7 points per
    10 USD. That is a minimum, since "below 70" and "above 90" leave the gap at least this wide.
    """
    drop = claims["gm_low_oil_pct"] - claims["gm_high_oil_pct"]
    rise = claims["oil_high_usd"] - claims["oil_low_usd"]
    return round(drop / rise * 10, 2)


def _quarterly_rows(features: pd.DataFrame) -> list[dict[str, Any]]:
    cols = ["quarter", "gm_pct", "opm_pct", "net_revenue_cr", "opex_cr", "brent_usd"]
    return [
        {
            "quarter": str(r["quarter"]),
            "gm": round(float(r["gm_pct"]), 2),
            "ebitda": round(float(r["opm_pct"]), 2),
            "net_revenue": round(float(r["net_revenue_cr"]), 1),
            "opex": round(float(r["opex_cr"]), 1),
            "brent": round(float(r["brent_usd"]), 2),
        }
        for _, r in features[cols].iterrows()
    ]


def _ebitda_by_quarter(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for label, group in features.assign(q=features["quarter"].str[-2:]).groupby("q"):
        values = group["opm_pct"].astype(float)
        out[str(label)] = {
            "min": round(float(values.min()), 1),
            "median": round(float(values.median()), 1),
            "max": round(float(values.max()), 1),
            "n": int(len(values)),
        }
    return out


def _within_year_costs(features: pd.DataFrame) -> dict[str, Any]:
    """How much sales and operating costs each fall from Q2 to Q4 within the same year.

    This isolates the seasons from the ten years of growth that a pooled cost line mixes in, and is
    the direct evidence for operating leverage: sales drop by half while costs barely move.
    """
    frame = features.assign(
        year=features["quarter"].str[:4].astype(int), q=features["quarter"].str[-2:]
    )
    sales = frame.pivot(index="year", columns="q", values="net_revenue_cr")
    opex = frame.pivot(index="year", columns="q", values="opex_cr")
    years = []
    for year in sales.index:
        if (
            {"Q2", "Q4"} <= set(sales.columns)
            and pd.notna(sales.loc[year, "Q2"])
            and pd.notna(sales.loc[year, "Q4"])
        ):
            years.append(
                {
                    "year": int(year),
                    "sales_change_pct": round(
                        float(sales.loc[year, "Q4"] / sales.loc[year, "Q2"] - 1) * 100, 1
                    ),
                    "opex_change_pct": round(
                        float(opex.loc[year, "Q4"] / opex.loc[year, "Q2"] - 1) * 100, 1
                    ),
                    "opex_ratio_q2": round(
                        float(opex.loc[year, "Q2"] / sales.loc[year, "Q2"]) * 100, 1
                    ),
                    "opex_ratio_q4": round(
                        float(opex.loc[year, "Q4"] / sales.loc[year, "Q4"]) * 100, 1
                    ),
                }
            )
    if not years:
        return {"years": []}
    frame_y = pd.DataFrame(years)
    return {
        "years": years,
        "median_sales_change_pct": round(float(frame_y["sales_change_pct"].median()), 1),
        "median_opex_change_pct": round(float(frame_y["opex_change_pct"].median()), 1),
        "median_opex_ratio_q2": round(float(frame_y["opex_ratio_q2"].median()), 1),
        "median_opex_ratio_q4": round(float(frame_y["opex_ratio_q4"].median()), 1),
    }


def _backtest_series(features: pd.DataFrame, min_train: int = 12) -> list[dict[str, Any]]:
    frame = features.copy()
    frame.index = pd.PeriodIndex(frame["quarter"], freq="Q")
    try:
        result, _ = rolling_backtest(frame, TARGET, LAG_FEATURES, min_train=min_train)
    except InsufficientDataError:
        return []
    return [
        {
            "quarter": str(period),
            "actual": round(float(row["actual"]), 2),
            "model": round(float(row["model"]), 2),
            "seasonal_naive": round(float(row["seasonal_naive"]), 2),
        }
        for period, row in result.iterrows()
    ]


def build_site_data(
    features: pd.DataFrame,
    claims: pd.DataFrame,
    backtest_summary: dict[str, Any],
    model_meta: dict[str, Any],
    leverage_meta: dict[str, Any],
    nowcast: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble everything the page draws, from the repository's own outputs."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "repo_url": REPO_URL,
        "report_claims": {**REPORT_CLAIMS, "implied_points_per_10usd": implied_points_per_10usd()},
        "quarters": _quarterly_rows(features),
        "ebitda_by_quarter": _ebitda_by_quarter(features),
        "seasonal_costs": _within_year_costs(features),
        "claims": claims.fillna("").to_dict(orient="records"),
        "backtest": {
            "summary": {k: v for k, v in backtest_summary.items() if k != "ablation"},
            "ablation": backtest_summary.get("ablation", {}),
            "series": _backtest_series(features),
        },
        "model": {
            "latest_quarter": model_meta.get("latest_quarter"),
            "alpha": model_meta.get("alpha"),
            "effect_per_10usd": model_meta.get("gm_points_per_10usd_brent"),
            "coefficients": model_meta.get("coefficients", []),
            "error_quantiles": model_meta.get("backtest_error_quantiles"),
            "forced": model_meta.get("forced"),
        },
        "leverage": leverage_meta.get("fit", {}),
        "nowcast": nowcast,
        "monitor": monitor,
    }


def write_data_js(data: dict[str, Any], path: Path) -> None:
    """Write the data as a script the page loads, so it also works opened straight from disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "// Generated by `pwl site`. Do not edit by hand.\n"
        f"window.SITE_DATA = {json.dumps(data, indent=1, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
