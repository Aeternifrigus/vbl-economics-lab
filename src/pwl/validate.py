"""Data validation for the VBL quarterly table.

Errors (`check_vbl`) mean the table is internally broken. Warnings (`vbl_warnings`) mean a value is
unusual and worth a look but may be real, such as a gap in coverage or a COVID-era margin.
"""

from pathlib import Path

import pandas as pd

from .accounting import net_revenue
from .config import DATA_RAW
from .exceptions import DataValidationError
from .logging_config import get_logger

logger = get_logger(__name__)

REQUIRED = [
    "quarter",
    "revenue_cr",
    "excise_duty_cr",
    "gross_profit_cr",
    "opex_cr",
    "ebitda_cr",
    "source",
]
GM_MIN = 40.0
GM_MAX = 70.0
# Plausibility bands from VBL's reported history. Outside them is a warning, not an error.
GM_WARN = (48.0, 60.0)
EBITDA_MARGIN_WARN = (10.0, 35.0)


def load_vbl(path: Path | None = None) -> pd.DataFrame:
    """Load and parse the VBL quarterly CSV."""
    csv_path = path or DATA_RAW / "vbl_quarterly.csv"
    logger.info("Loading VBL data from %s", csv_path)
    df = pd.read_csv(csv_path)
    if "source" not in df.columns and "source_url" in df.columns:
        df = df.rename(columns={"source_url": "source"})
    df["period"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df.sort_values("period").reset_index(drop=True)


def check_vbl(df: pd.DataFrame, tol_cr: float = 3.0) -> list[str]:
    """Return problems that make the table unusable (an empty list means clean)."""
    problems: list[str] = []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if df["quarter"].duplicated().any():
        problems.append("duplicate quarters")

    for col in ["revenue_cr", "gross_profit_cr", "opex_cr", "ebitda_cr"]:
        if (df[col] <= 0).any():
            problems.append(f"non-positive values in {col}")
    if (df["excise_duty_cr"] < 0).any() or (df["excise_duty_cr"] >= df["revenue_cr"]).any():
        problems.append("excise_duty_cr must be between 0 and revenue_cr")

    gap = (df["gross_profit_cr"] - df["opex_cr"] - df["ebitda_cr"]).abs()
    for q in df.loc[gap > tol_cr, "quarter"]:
        problems.append(
            f"{q}: gross profit minus opex differs from EBITDA by more than {tol_cr} crore"
        )

    gm = df["gross_profit_cr"] / net_revenue(df) * 100
    for q in df.loc[(gm < GM_MIN) | (gm > GM_MAX), "quarter"]:
        problems.append(f"{q}: gross margin outside {GM_MIN:.0f} to {GM_MAX:.0f} percent")

    source = df["source"].astype(str)
    bad_source = df["source"].isna() | ~(
        source.str.startswith("http") | source.str.startswith("file:")
    )
    for q in df.loc[bad_source, "quarter"]:
        problems.append(f"{q}: source must be an http URL or a 'file:' reference")

    if problems:
        logger.warning("Validation found %d problems", len(problems))
        for p in problems:
            logger.warning("  - %s", p)
    else:
        logger.info("Validation passed: %d rows clean", len(df))
    return problems


def vbl_warnings(df: pd.DataFrame) -> list[str]:
    """Return unusual-but-possible findings: coverage gaps and out-of-band margins."""
    warnings: list[str] = []

    periods = pd.PeriodIndex(df["quarter"], freq="Q").sort_values()
    if len(periods) > 1:
        full = pd.period_range(periods[0], periods[-1], freq="Q")
        missing = [str(p) for p in full if p not in set(periods)]
        if missing:
            warnings.append(
                f"{len(missing)} quarters missing between {periods[0]} and {periods[-1]} "
                f"({missing[0]} to {missing[-1]}); lag features use calendar quarters so this is "
                "safe, but fewer rows are usable"
            )

    net = net_revenue(df)
    gm = df["gross_profit_cr"] / net * 100
    margin = df["ebitda_cr"] / net * 100
    for q, value in zip(df["quarter"], gm, strict=True):
        if not GM_WARN[0] <= value <= GM_WARN[1]:
            warnings.append(
                f"{q}: gross margin {value:.1f}% outside {GM_WARN[0]:.0f} to {GM_WARN[1]:.0f}%"
            )
    for q, value in zip(df["quarter"], margin, strict=True):
        if not EBITDA_MARGIN_WARN[0] <= value <= EBITDA_MARGIN_WARN[1]:
            warnings.append(
                f"{q}: EBITDA margin {value:.1f}% outside "
                f"{EBITDA_MARGIN_WARN[0]:.0f} to {EBITDA_MARGIN_WARN[1]:.0f}%"
            )
    return warnings


def validate_or_raise(df: pd.DataFrame, tol_cr: float = 3.0) -> None:
    """Validate and raise DataValidationError if problems are found."""
    problems = check_vbl(df, tol_cr)
    if problems:
        raise DataValidationError("VBL data validation failed", problems)
