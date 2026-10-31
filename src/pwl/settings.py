"""Application settings using pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")

    # Paths
    data_raw_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw"
    )
    data_external_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "external"
    )
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models")
    reports_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "reports"
    )

    # Model parameters
    min_quarters_for_release: int = Field(default=20, alias="MIN_QUARTERS_FOR_RELEASE")
    heat_threshold_c: float = Field(default=38.0, alias="HEAT_THRESHOLD_C")

    # Ingestion
    ingest_start_date: str = Field(default="2015-01-01", alias="INGEST_START_DATE")

    # Logging
    log_level: str = Field(default="WARNING", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")  # json or console

    # Cities for weather weighting (name -> (lat, lon, weight))
    cities: dict[str, tuple[float, float, float]] = Field(
        default_factory=lambda: {
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
    )


# Global settings instance
settings = Settings()
