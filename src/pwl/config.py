"""Configuration constants - thin wrapper around settings for backward compatibility."""

from pathlib import Path

from .settings import settings

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = settings.data_raw_dir
DATA_EXT = settings.data_external_dir
MODELS = settings.models_dir
REPORTS = settings.reports_dir

# A lagged or weather model is refused below this many quarters unless forced.
MIN_QUARTERS_FOR_RELEASE = settings.min_quarters_for_release

# Taken from the original report, treated as a tunable rather than a fact.
HEAT_THRESHOLD_C = settings.heat_threshold_c

# ASSUMPTION: rough metro populations in millions, used only as weights for a
# population-weighted weather index. Replace with a better demand weighting.
CITIES = settings.cities


# Feature columns for the lagged model (defined here for backward compat)
LAG_FEATURES = [
    "brent_inr_lag0",
    "brent_inr_lag1",
    "brent_inr_lag2",
    "brent_inr_lag3",
    "tmax_anom",
    "heat_days",
    "rain_anom",
]
