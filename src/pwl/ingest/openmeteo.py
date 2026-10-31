"""Open-Meteo historical archive client (no key needed). Builds a population-weighted index."""
import numpy as np
import pandas as pd

from ..config import CITIES, HEAT_THRESHOLD_C
from ..http import get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_daily(payload, city):
    daily = payload.get("daily", {})
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", [])),
            "tmax_c": pd.to_numeric(pd.Series(daily.get("temperature_2m_max", [])), errors="coerce"),
            "precip_mm": pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce"),
        }
    )
    df["city"] = city
    return df


def fetch_city(city, lat, lon, start, end):
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


def weight_cities(frames, weights, threshold=HEAT_THRESHOLD_C):
    """Combine per-city frames into one daily frame: weighted tmax, weighted rain, heat share."""
    long = pd.concat(frames, ignore_index=True)
    long["w"] = long["city"].map(weights)
    long["hot"] = (long["tmax_c"] >= threshold).astype(float)
    out = []
    for date, g in long.groupby("date"):
        ok_t = g["tmax_c"].notna()
        ok_p = g["precip_mm"].notna()
        tmax = np.average(g.loc[ok_t, "tmax_c"], weights=g.loc[ok_t, "w"]) if ok_t.any() else np.nan
        rain = np.average(g.loc[ok_p, "precip_mm"], weights=g.loc[ok_p, "w"]) if ok_p.any() else np.nan
        heat = np.average(g.loc[ok_t, "hot"], weights=g.loc[ok_t, "w"]) if ok_t.any() else np.nan
        out.append((date, tmax, rain, heat, int(ok_t.sum())))
    return pd.DataFrame(out, columns=["date", "tmax_c", "precip_mm", "heat_share", "n_cities"])


def fetch_weighted(start, end, cities=CITIES):
    frames = [fetch_city(name, lat, lon, start, end) for name, (lat, lon, _) in cities.items()]
    weights = {name: w for name, (_, _, w) in cities.items()}
    return weight_cities(frames, weights)
