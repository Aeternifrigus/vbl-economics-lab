"""Open-Meteo historical archive client (no key needed). Builds a population-weighted index."""

import numpy as np
import pandas as pd

from ..config import HEAT_THRESHOLD_C
from ..http import get_json
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_daily(payload: dict, city: str) -> pd.DataFrame:
    """Parse Open-Meteo daily response into a DataFrame."""
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    precip = daily.get("precipitation_sum", [])

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(times),
            "tmax_c": pd.to_numeric(pd.Series(tmax), errors="coerce"),
            "precip_mm": pd.to_numeric(pd.Series(precip), errors="coerce"),
        }
    )
    df["city"] = city
    return df


def fetch_city(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch weather data for a single city."""
    logger.debug(
        "Fetching weather for %s (%.2f, %.2f) from %s to %s",
        city,
        lat,
        lon,
        start,
        end,
    )
    payload = get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "Asia/Kolkata",
        },
    )
    return parse_daily(payload, city)


def weight_cities(
    frames: list[pd.DataFrame],
    weights: dict[str, float],
    threshold: float = HEAT_THRESHOLD_C,
) -> pd.DataFrame:
    """Combine per-city frames into one daily frame: weighted tmax, weighted rain, heat share."""
    if not frames:
        return pd.DataFrame(columns=["date", "tmax_c", "precip_mm", "heat_share", "n_cities"])

    long = pd.concat(frames, ignore_index=True)
    long["w"] = long["city"].map(weights)
    long["hot"] = (long["tmax_c"] >= threshold).astype(float)

    out = []
    for date, g in long.groupby("date"):
        ok_t = g["tmax_c"].notna()
        ok_p = g["precip_mm"].notna()
        tmax = np.average(g.loc[ok_t, "tmax_c"], weights=g.loc[ok_t, "w"]) if ok_t.any() else np.nan
        rain = (
            np.average(g.loc[ok_p, "precip_mm"], weights=g.loc[ok_p, "w"]) if ok_p.any() else np.nan
        )
        heat = np.average(g.loc[ok_t, "hot"], weights=g.loc[ok_t, "w"]) if ok_t.any() else np.nan
        out.append((date, tmax, rain, heat, int(ok_t.sum())))
    return pd.DataFrame(out, columns=["date", "tmax_c", "precip_mm", "heat_share", "n_cities"])


def fetch_weighted(start: str, end: str, cities: dict | None = None) -> pd.DataFrame:
    """Fetch and combine weather for all cities."""
    city_data = cities or settings.cities
    logger.info(
        "Fetching weather for %d cities from %s to %s",
        len(city_data),
        start,
        end,
    )

    frames = [fetch_city(name, lat, lon, start, end) for name, (lat, lon, _) in city_data.items()]
    weights = {name: w for name, (_, _, w) in city_data.items()}
    result = weight_cities(frames, weights)
    logger.info("Combined weather data: %d daily rows", len(result))
    return result
