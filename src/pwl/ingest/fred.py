"""FRED client for Brent crude (USD/bbl) and USD/INR. Needs a free FRED_API_KEY."""
import pandas as pd

from ..http import get_json

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
BRENT = "DCOILBRENTEU"
USDINR = "DEXINUS"


def parse_observations(payload):
    rows = []
    for obs in payload.get("observations", []):
        value = obs.get("value")
        if value in (None, "", "."):
            continue
        rows.append((obs["date"], float(value)))
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_series(series_id, api_key, start="2015-01-01"):
    payload = get_json(
        FRED_URL,
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        },
    )
    return parse_observations(payload)
