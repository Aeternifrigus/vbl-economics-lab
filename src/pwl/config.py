from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# A lagged or weather model is refused below this many quarters unless forced.
MIN_QUARTERS_FOR_RELEASE = 20

# Taken from the original report, treated as a tunable rather than a fact.
HEAT_THRESHOLD_C = 38.0

# ASSUMPTION: rough metro populations in millions, used only as weights for a
# population-weighted weather index. Replace with a better demand weighting.
CITIES = {
    "Delhi": (28.61, 77.21, 32.0),
    "Mumbai": (19.08, 72.88, 21.0),
    "Kolkata": (22.57, 88.36, 15.0),
    "Chennai": (13.08, 80.27, 11.0),
    "Bengaluru": (12.97, 77.59, 13.0),
    "Hyderabad": (17.39, 78.49, 11.0),
    "Lucknow": (26.85, 80.95, 4.0),
    "Patna": (25.59, 85.14, 2.5),
    "Jaipur": (26.91, 75.79, 4.0),
    "Ahmedabad": (23.02, 72.57, 8.5),
}
