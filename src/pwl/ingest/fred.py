"""FRED client for Brent crude (USD/bbl) and USD/INR. Needs a free FRED_API_KEY."""

import pandas as pd

from ..exceptions import IngestionError
from ..http import get_json
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
BRENT = "DCOILBRENTEU"
USDINR = "DEXINUS"


def parse_observations(payload: dict) -> pd.DataFrame:
    """Parse FRED observations response into a DataFrame."""
    rows = []
    for obs in payload.get("observations", []):
        value = obs.get("value")
        if value in (None, "", "."):
            continue
        try:
            rows.append((obs["date"], float(value)))
        except (ValueError, TypeError):
            logger.warning("Skipping invalid observation: %s", obs)
            continue
    df = pd.DataFrame(rows, columns=["date", "value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_series(
    series_id: str, api_key: str | None = None, start: str | None = None
) -> pd.DataFrame:
    """Fetch a time series from FRED."""
    key = api_key or settings.fred_api_key
    if not key:
        raise IngestionError(
            "FRED_API_KEY not set. Get a free key from fred.stlouisfed.org",
            source="FRED",
        )

    start_date = start or settings.ingest_start_date
    logger.info("Fetching FRED series %s from %s", series_id, start_date)

    payload = get_json(
        FRED_URL,
        {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start_date,
        },
    )
    df = parse_observations(payload)
    logger.info("Fetched %d rows for %s", len(df), series_id)
    return df
