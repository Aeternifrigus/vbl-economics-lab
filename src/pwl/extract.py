"""Parse VBL consolidated results statements into reconciled quarterly line items.

Definitions (fixed, so every quarter is comparable):

    cost of goods = materials consumed + excise duty + purchases of stock-in-trade
                    + changes in inventories (a build-up is negative, so it lowers cost)
    gross profit  = revenue from operations - cost of goods
    opex          = employee benefits expense + other expenses
    EBITDA        = gross profit - opex
    net revenue   = revenue from operations - excise duty (the base for reported margins)

Excise duty is stored separately. Gross profit above equals net revenue minus cost of goods
excluding excise, so gross profit divided by net revenue reproduces the company's reported gross
margin. Margins in this project are computed on net revenue.

The text layer of these PDFs is OCR quality: decimal points come out as commas or periods
("9.967.60"), minus signs and dashes vary, and headers have typos. Numbers are therefore normalised
by their shape (every amount has exactly two decimals), and each column must pass two tie-outs
against the filing's own totals before it is accepted:

    revenue + other income = total income
    sum of the eight expense lines = total expenses

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

_DASHES = "-\u2013\u2014\u2212"
_AMOUNT = re.compile(rf"\(?[{_DASHES}]?\d+(?:[.,]\d{{3}})*[.,]\d{{2}}\)?|(?<!\S)[{_DASHES}](?!\S)")
_AMOUNT_SHAPE = re.compile(r"(\d+(?:[.,]\d{3})*)[.,](\d{2})")
TIE_TOLERANCE = 0.2  # in statement units; rounding of nine two-decimal lines stays well below this
_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_STATEMENT = re.compile(r"Statement of\s+(\w+)\s+(?:unaudited\s+)?financial\s+res", re.I)
_THREE_MONTHS = re.compile(r"T\s*h\s*r\s*e\s*e\s*m\s*o\s*n\s*t\s*h\s*s", re.I)
_UNIT = re.compile(r"in\s+(million|crore|lakh)", re.I)
_PAGE_MARK = re.compile(r"^--- PAGE (\d+) ---$", re.M)

# item name -> regex searched in a statement line (first match wins)
_ITEMS = {
    "revenue": re.compile(r"revenue from op", re.I),
    "other_income": re.compile(r"other income", re.I),
    "total_income": re.compile(r"total income", re.I),
    "materials": re.compile(r"cost of materials", re.I),
    "excise": re.compile(r"excise du", re.I),
    "purchases": re.compile(r"purchases? of stock", re.I),
    "changes": re.compile(r"changes in inventor", re.I),
    "employee": re.compile(r"employee benefit", re.I),
    "finance": re.compile(r"finance cost", re.I),
    "depreciation": re.compile(r"deprec", re.I),
    "other": re.compile(r"\bother expenses\b", re.I),
    "total_expenses": re.compile(r"total expenses", re.I),
}
_REQUIRED = (
    "revenue",
    "materials",
    "purchases",
    "changes",
    "employee",
    "finance",
    "depreciation",
    "other",
    "total_expenses",
)


@dataclass(frozen=True)
class StatementRow:
    """One quarter parsed from one statement column."""

    quarter: str
    revenue_cr: float
    excise_duty_cr: float
    gross_profit_cr: float
    opex_cr: float
    ebitda_cr: float
    col_index: int
    source: str


def _to_amount(token: str) -> float:
    """Convert one OCR-damaged token into a float using its shape, not its separators."""
    token = token.strip()
    if token and token in _DASHES:
        return 0.0
    negative = token.startswith("(") or token.endswith(")") or any(c in token for c in _DASHES)
    core = token.strip("()" + _DASHES + " ")
    match = _AMOUNT_SHAPE.fullmatch(core)
    if match is None:
        raise ValueError(f"unreadable amount {token!r}")
    value = float(re.sub(r"[.,]", "", match.group(1)) + "." + match.group(2))
    return -value if negative else value


def parse_numbers(line: str) -> list[float]:
    """Pull amounts from a statement line. Brackets mean negative, a lone dash means nil."""
    return [_to_amount(t) for t in _AMOUNT.findall(line)]


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
        n_quarterly += len(_THREE_MONTHS.findall(line))
        for day, month, year in _DATE.findall(line):
            dates.append(quarter_label(day, month, year))
    return dates, n_quarterly


def _tie_out(found: dict[str, list[float]], col: int) -> tuple[float, float | None]:
    """Return (expense tie-out difference, income tie-out difference or None)."""
    expenses = sum(
        found[k][col]
        for k in (
            "materials",
            "purchases",
            "changes",
            "employee",
            "finance",
            "depreciation",
            "other",
        )
    ) + (found["excise"][col] if "excise" in found else 0.0)
    expense_gap = expenses - found["total_expenses"][col]
    income_gap = None
    if "other_income" in found and "total_income" in found:
        income_gap = found["revenue"][col] + found["other_income"][col] - found["total_income"][col]
    return expense_gap, income_gap


def rows_from_items(
    found: dict[str, list[float]],
    quarters: list[str | None],
    scale: float,
    source: str,
) -> list[StatementRow]:
    """Turn line items into rows, keeping only columns that pass both tie-outs.

    Used for parsed PDFs and for hand-entered statements, so both get identical checks.
    """
    rows = []
    for col, quarter in enumerate(quarters):
        if quarter is None:
            continue
        expense_gap, income_gap = _tie_out(found, col)
        if abs(expense_gap) > TIE_TOLERANCE or (
            income_gap is not None and abs(income_gap) > TIE_TOLERANCE
        ):
            logger.warning(
                "%s %s: rejected, tie-out failed (expenses off by %.2f, income off by %s)",
                source,
                quarter,
                expense_gap,
                "n/a" if income_gap is None else f"{income_gap:.2f}",
            )
            continue
        revenue = found["revenue"][col]
        excise = found["excise"][col] if "excise" in found else 0.0
        cogs = found["materials"][col] + excise + found["purchases"][col] + found["changes"][col]
        opex = found["employee"][col] + found["other"][col]
        gross = revenue - cogs
        rows.append(
            StatementRow(
                quarter=quarter,
                revenue_cr=round(revenue * scale, 1),
                excise_duty_cr=round(excise * scale, 1),
                gross_profit_cr=round(gross * scale, 1),
                opex_cr=round(opex * scale, 1),
                ebitda_cr=round((gross - opex) * scale, 1),
                col_index=col,
                source=source,
            )
        )
    return rows


def parse_statement_lines(lines: list[str], source: str) -> list[StatementRow]:
    """Parse one consolidated statement block into per-quarter rows that pass both tie-outs."""
    text = "\n".join(lines)
    if not any(_ITEMS["revenue"].search(ln) for ln in lines):
        logger.debug("%s: block has no revenue line, not an income statement", source)
        return []
    unit_match = _UNIT.search(text)
    if unit_match is None:
        logger.warning("%s: no unit found in header, skipping statement", source)
        return []
    scale = UNIT_TO_CRORE[unit_match.group(1).lower()]

    # OCR sometimes drops the word "Particulars", so the column header is also found by its
    # "Three months" labels.
    start = next(
        (i for i, ln in enumerate(lines) if "articulars" in ln or _THREE_MONTHS.search(ln)), None
    )
    if start is None:
        logger.warning("%s: no column header found, skipping statement", source)
        return []
    dates, n_quarterly = _header_dates(lines, start)
    n_cols = min(n_quarterly, 3) if n_quarterly else 3
    quarters = dates[:n_cols]

    found: dict[str, list[float]] = {}
    for line in lines:
        for name, pattern in _ITEMS.items():
            if name in found or not pattern.search(line):
                continue
            try:
                nums = parse_numbers(line)
            except ValueError:
                continue
            if len(nums) >= n_cols:
                found[name] = nums[:n_cols]
    missing = [k for k in _REQUIRED if k not in found]
    if missing:
        logger.warning("%s: missing line items %s, skipping statement", source, missing)
        return []

    return rows_from_items(found, quarters, scale, source)


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
            word = _STATEMENT.search(lines[header_idx]).group(1).lower()  # type: ignore[union-attr]
            kind = "consolidated" if word.startswith("con") else word
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
                "excise_duty_cr": r.excise_duty_cr,
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


MIN_TEXT_CHARS = 50  # a page with less text than this is treated as a scan
MAX_OCR_PAGES = 4  # the income statement is always in the first pages, and OCR is slow


def ocr_page(page: object) -> str:
    """OCR one pdfplumber page with two segmentation modes. Needs pytesseract (optional).

    Mode 4 reads table columns well but drops some wrapped lines, and mode 6 keeps those lines.
    Both outputs are concatenated, and every accepted column must still pass the tie-outs.
    """
    try:
        import pytesseract  # noqa: PLC0415  (optional dependency, imported on use)
        from PIL import ImageOps  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigurationError(
            "OCR needs pytesseract and the tesseract binary. "
            "Run: pip install 'vbl-economics-lab[ocr]'",
            missing=["pytesseract"],
        ) from exc
    image = page.to_image(resolution=400).original.convert("L")  # type: ignore[attr-defined]
    image = ImageOps.autocontrast(image, cutoff=2)
    runs = [
        str(
            pytesseract.image_to_string(image, config=f"--psm {psm} -c preserve_interword_spaces=1")
        )
        for psm in (4, 6)
    ]
    return "\n".join(runs)


def read_pdf_pages(path: Path, ocr: bool = False, max_ocr_pages: int = MAX_OCR_PAGES) -> list[str]:
    """Read page text from a PDF, falling back to OCR for pages with no text layer.

    Needs the optional pdfplumber dependency (and pytesseract for scans).
    """
    try:
        import pdfplumber  # noqa: PLC0415  (optional dependency, imported on use)
    except ImportError as exc:
        raise ConfigurationError(
            "pdfplumber is not installed. Run: pip install 'vbl-economics-lab[pdf]'",
            missing=["pdfplumber"],
        ) from exc
    texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if ocr and number <= max_ocr_pages and len(text.strip()) < MIN_TEXT_CHARS:
                logger.info("%s p%d: no text layer, running OCR", path.name, number)
                try:
                    text = ocr_page(page)
                except ConfigurationError:
                    logger.warning("%s p%d: scan skipped, OCR is not available", path.name, number)
            texts.append(text)
    return texts


_MANUAL_ITEMS = (
    "revenue",
    "other_income",
    "total_income",
    "materials",
    "excise",
    "purchases",
    "changes",
    "employee",
    "finance",
    "depreciation",
    "other",
    "total_expenses",
)


def load_manual_statements(path: Path) -> list[StatementRow]:
    """Load hand-entered statement columns (for scanned PDFs OCR cannot read).

    One row per quarter with the statement's own line items, in the unit given in the `unit`
    column. Rows go through the same tie-outs as parsed PDFs, so a typo is rejected, not merged.
    """
    table = pd.read_csv(path)
    rows: list[StatementRow] = []
    for _, r in table.iterrows():
        found = {
            name: [float(r[name])] for name in _MANUAL_ITEMS if name in r and pd.notna(r[name])
        }
        missing = [k for k in _REQUIRED if k not in found]
        if missing:
            logger.warning(
                "%s %s: manual row missing %s, skipped", path.name, r["quarter"], missing
            )
            continue
        scale = UNIT_TO_CRORE[str(r["unit"]).lower()]
        rows.extend(rows_from_items(found, [str(r["quarter"])], scale, str(r["source"])))
    return rows


def extract_directory(pdf_dir: Path, manual: Path | None = None, ocr: bool = False) -> pd.DataFrame:
    """Parse every PDF in a directory (plus optional manual statements) into one row per quarter."""
    rows: list[StatementRow] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        parsed = extract_from_pages(read_pdf_pages(path, ocr=ocr), path.name)
        logger.info("%s: %d quarter columns parsed", path.name, len(parsed))
        rows.extend(parsed)
    if manual is not None and manual.exists():
        entered = load_manual_statements(manual)
        logger.info("%s: %d manual columns accepted", manual.name, len(entered))
        rows.extend(entered)
    return merge_rows(rows)


TABLE_COLUMNS = (
    "quarter",
    "revenue_cr",
    "excise_duty_cr",
    "gross_profit_cr",
    "opex_cr",
    "ebitda_cr",
    "volume_growth_yoy_pct",
    "volume_note",
    "source",
)
_CARRIED_COLUMNS = ("volume_growth_yoy_pct", "volume_note")


def merge_into_table(
    existing: pd.DataFrame, new: pd.DataFrame, overwrite: bool = False
) -> pd.DataFrame:
    """Add extracted quarters to the table. Existing quarters are kept unless overwrite is set.

    When quarters are overwritten, hand-entered volume fields on the old row are carried over,
    because the statements do not contain them.
    """
    if new.empty:
        return existing
    if overwrite:
        replaced = new.copy()
        old = existing.set_index("quarter") if "quarter" in existing.columns else None
        if old is not None:
            for col in _CARRIED_COLUMNS:
                if col not in old.columns or col not in replaced.columns:
                    continue
                carry = replaced["quarter"].map(old[col])
                blank = replaced[col].isna() | (replaced[col].astype(str) == "")
                replaced.loc[blank, col] = carry[blank]
        keep = existing[~existing["quarter"].isin(new["quarter"])]
        combined = pd.concat([keep, replaced], ignore_index=True)
    else:
        fresh = new[~new["quarter"].isin(existing["quarter"])]
        combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.sort_values("quarter").reset_index(drop=True)
    ordered = [c for c in TABLE_COLUMNS if c in combined.columns]
    return combined[ordered + [c for c in combined.columns if c not in TABLE_COLUMNS]]
