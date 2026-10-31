"""Parse VBL consolidated results statements into reconciled quarterly line items.

Definitions (fixed, so every quarter is comparable):

    cost of goods = materials consumed + excise duty + purchases of stock-in-trade
                    + changes in inventories (a build-up is negative, so it lowers cost)
    gross profit  = revenue from operations - cost of goods
    opex          = employee benefits expense + other expenses
    EBITDA        = gross profit - opex

The earlier extraction left out the change in inventories and read the wrong EBITDA line, which is
what this module replaces. All amounts are converted to crore using the unit stated in the header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)

UNIT_TO_CRORE = {"million": 0.1, "crore": 1.0, "lakh": 0.01}
MONTH_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
QUARTER_END_MONTHS = {3, 6, 9, 12}

_NUMBER = re.compile(r"\(\s*[\d,]+\.\d+\s*\)|-?[\d,]+\.\d+|(?<=\s)-(?=\s|$)")
_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_STATEMENT = re.compile(r"Statement of (consolidated|standalone) financial results", re.I)
_UNIT = re.compile(r"in\s+(million|crore|lakh)", re.I)
_PAGE_MARK = re.compile(r"^--- PAGE (\d+) ---$", re.M)

# item name -> regex searched in a statement line (first match wins)
_ITEMS = {
    "revenue": re.compile(r"revenue from operations", re.I),
    "materials": re.compile(r"cost of materials consumed", re.I),
    "excise": re.compile(r"excise duty", re.I),
    "purchases": re.compile(r"purchases of stock-in-trade", re.I),
    "changes": re.compile(r"changes in inventories", re.I),
    "employee": re.compile(r"employee benefits? expense", re.I),
    "other": re.compile(r"\bother expenses\b", re.I),
}
_REQUIRED = ("revenue", "materials", "purchases", "changes", "employee", "other")


@dataclass(frozen=True)
class StatementRow:
    """One quarter parsed from one statement column."""

    quarter: str
    revenue_cr: float
    gross_profit_cr: float
    opex_cr: float
    ebitda_cr: float
    col_index: int
    source: str


def parse_numbers(line: str) -> list[float]:
    """Pull amounts from a statement line. Parentheses mean negative, a lone dash means nil."""
    values: list[float] = []
    for raw in _NUMBER.findall(line):
        token = raw.strip()
        if token == "-":
            values.append(0.0)
        elif token.startswith("("):
            values.append(-float(token.strip("()").replace(",", "").strip()))
        else:
            values.append(float(token.replace(",", "")))
    return values


def quarter_label(day: str, month_name: str, year: str) -> str | None:
    """Turn '31 March 2023' into '2023Q1'. Returns None when it is not a quarter end."""
    month = MONTH_NUMBER.get(month_name.lower())
    if month is None or month not in QUARTER_END_MONTHS:
        return None
    return f"{int(year)}Q{(month - 1) // 3 + 1}"


def split_pages(text: str) -> list[str]:
    """Split a saved text dump on '--- PAGE n ---' markers (used by the test fixture)."""
    parts = _PAGE_MARK.split(text)
    # parts = [preamble, "1", text1, "2", text2, ...]
    return [parts[i + 1] for i in range(1, len(parts) - 1, 2)]


def _header_dates(lines: list[str], start: int) -> tuple[list[str | None], int]:
    """Collect column dates and the count of 'Three months' columns after the Particulars line."""
    dates: list[str | None] = []
    n_quarterly = 0
    for line in lines[start : start + 8]:
        if re.match(r"\s*1\.\s*Income", line):
            break
        n_quarterly += len(re.findall(r"Three months", line, re.I))
        for day, month, year in _DATE.findall(line):
            dates.append(quarter_label(day, month, year))
    return dates, n_quarterly


def parse_statement_lines(lines: list[str], source: str) -> list[StatementRow]:
    """Parse one consolidated statement block into per-quarter rows."""
    text = "\n".join(lines)
    unit_match = _UNIT.search(text)
    if unit_match is None:
        logger.warning("%s: no unit found in header, skipping statement", source)
        return []
    scale = UNIT_TO_CRORE[unit_match.group(1).lower()]

    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("Particulars")), None)
    if start is None:
        logger.warning("%s: no Particulars header, skipping statement", source)
        return []
    dates, n_quarterly = _header_dates(lines, start)
    n_cols = n_quarterly or 3
    quarters = dates[:n_cols]

    found: dict[str, list[float]] = {}
    for line in lines[start:]:
        for name, pattern in _ITEMS.items():
            if name not in found and pattern.search(line):
                nums = parse_numbers(line)
                if len(nums) >= n_cols:
                    found[name] = nums[:n_cols]
    missing = [k for k in _REQUIRED if k not in found]
    if missing:
        logger.warning("%s: missing line items %s, skipping statement", source, missing)
        return []

    rows = []
    for col, quarter in enumerate(quarters):
        if quarter is None:
            continue
        revenue = found["revenue"][col]
        cogs = (
            found["materials"][col]
            + (found["excise"][col] if "excise" in found else 0.0)
            + found["purchases"][col]
            + found["changes"][col]
        )
        opex = found["employee"][col] + found["other"][col]
        gross = revenue - cogs
        rows.append(
            StatementRow(
                quarter=quarter,
                revenue_cr=round(revenue * scale, 1),
                gross_profit_cr=round(gross * scale, 1),
                opex_cr=round(opex * scale, 1),
                ebitda_cr=round((gross - opex) * scale, 1),
                col_index=col,
                source=source,
            )
        )
    return rows


def extract_from_pages(pages: list[str], file_name: str) -> list[StatementRow]:
    """Find consolidated statements across pages and parse each one."""
    rows: list[StatementRow] = []
    block: list[str] = []
    block_page = 0
    kind: str | None = None

    def flush() -> None:
        nonlocal block
        if kind == "consolidated" and block:
            rows.extend(parse_statement_lines(block, f"file:{file_name}#p{block_page}"))
        block = []

    for number, page in enumerate(pages, start=1):
        lines = page.splitlines()
        header_idx = next((i for i, ln in enumerate(lines) if _STATEMENT.search(ln)), None)
        if header_idx is not None:
            flush()
            kind = _STATEMENT.search(lines[header_idx]).group(1).lower()  # type: ignore[union-attr]
            block_page = number
            block = lines[header_idx:]
        elif kind is not None:
            block.extend(lines)
    flush()
    return rows


def merge_rows(rows: list[StatementRow]) -> pd.DataFrame:
    """One row per quarter. The current-quarter column of a filing beats a later comparative."""
    best: dict[str, StatementRow] = {}
    for row in sorted(rows, key=lambda r: r.col_index):
        previous = best.get(row.quarter)
        if previous is None:
            best[row.quarter] = row
            continue
        drift = abs(row.revenue_cr - previous.revenue_cr) / previous.revenue_cr
        gm_prev = previous.gross_profit_cr / previous.revenue_cr
        gm_new = row.gross_profit_cr / row.revenue_cr
        if drift > 0.005 or abs(gm_new - gm_prev) > 0.005:
            logger.warning(
                "%s differs between %s and %s (restatement or parse error), keeping %s",
                row.quarter,
                previous.source,
                row.source,
                previous.source,
            )
    frame = pd.DataFrame(
        [
            {
                "quarter": r.quarter,
                "revenue_cr": r.revenue_cr,
                "gross_profit_cr": r.gross_profit_cr,
                "opex_cr": r.opex_cr,
                "ebitda_cr": r.ebitda_cr,
                "volume_growth_yoy_pct": float("nan"),
                "volume_note": "",
                "source": r.source,
            }
            for r in best.values()
        ]
    )
    if frame.empty:
        return frame
    return frame.sort_values("quarter").reset_index(drop=True)


def read_pdf_pages(path: Path) -> list[str]:
    """Read page text from a PDF. Needs the optional pdfplumber dependency."""
    try:
        import pdfplumber  # noqa: PLC0415  (optional dependency, imported on use)
    except ImportError as exc:
        raise ConfigurationError(
            "pdfplumber is not installed. Run: pip install 'packaged-water-lab[pdf]'",
            missing=["pdfplumber"],
        ) from exc
    with pdfplumber.open(path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def extract_directory(pdf_dir: Path) -> pd.DataFrame:
    """Parse every PDF in a directory and return one reconciled row per quarter."""
    rows: list[StatementRow] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        parsed = extract_from_pages(read_pdf_pages(path), path.name)
        logger.info("%s: %d quarter columns parsed", path.name, len(parsed))
        rows.extend(parsed)
    return merge_rows(rows)


def merge_into_table(
    existing: pd.DataFrame, new: pd.DataFrame, overwrite: bool = False
) -> pd.DataFrame:
    """Add extracted quarters to the table. Existing quarters are kept unless overwrite is set."""
    if new.empty:
        return existing
    if overwrite:
        keep = existing[~existing["quarter"].isin(new["quarter"])]
        combined = pd.concat([keep, new], ignore_index=True)
    else:
        fresh = new[~new["quarter"].isin(existing["quarter"])]
        combined = pd.concat([existing, fresh], ignore_index=True)
    return combined.sort_values("quarter").reset_index(drop=True)
