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
# Calendar-quarter dummies (Q1 is the base). Gross margin has a strong seasonal pattern, partly
# from inventory build-up in Q4, so oil is only credited with what seasonality does not explain.
SEASONAL_FEATURES = ["q2", "q3", "q4"]
OIL_FEATURES = ["brent_inr_lag0", "brent_inr_lag1", "brent_inr_lag2", "brent_inr_lag3"]
WEATHER_FEATURES = ["tmax_anom", "heat_days", "rain_anom"]
# Production gross margin model: oil lags plus seasonality. Weather is left out because the
# backtest shows it makes gross margin forecasts worse (see ABLATIONS), so it belongs in a
# volume model instead.
LAG_FEATURES = [*OIL_FEATURES, *SEASONAL_FEATURES]
ALL_FEATURES = [*OIL_FEATURES, *WEATHER_FEATURES, *SEASONAL_FEATURES]

# Feature sets compared in the backtest, to attribute any gain to seasonality, oil or weather.
ABLATIONS = {
    "seasonal_only": SEASONAL_FEATURES,
    "seasonal_plus_oil": LAG_FEATURES,
    "seasonal_plus_weather": [*WEATHER_FEATURES, *SEASONAL_FEATURES],
    "all": ALL_FEATURES,
}
