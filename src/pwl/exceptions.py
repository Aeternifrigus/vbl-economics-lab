"""Custom exceptions for the packaged-water-lab package."""


class PWLError(Exception):
    """Base exception for all PWL errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InsufficientDataError(PWLError):
    """Raised when there isn't enough data to perform an operation."""

    def __init__(self, message: str, available: int, required: int):
        super().__init__(message, {"available": available, "required": required})
        self.available = available
        self.required = required


class DataValidationError(PWLError):
    """Raised when data validation fails."""

    def __init__(self, message: str, problems: list[str]):
        super().__init__(message, {"problems": problems})
        self.problems = problems


class IngestionError(PWLError):
    """Raised when data ingestion fails."""

    def __init__(self, message: str, source: str, original_error: Exception | None = None):
        details = {"source": source}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)
        self.source = source
        self.original_error = original_error


class ModelTrainingError(PWLError):
    """Raised when model training fails."""

    def __init__(self, message: str, model_type: str, details: dict | None = None):
        super().__init__(message, {"model_type": model_type, **(details or {})})
        self.model_type = model_type


class ConfigurationError(PWLError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, missing: list[str] | None = None):
        super().__init__(message, {"missing": missing or []})
        self.missing = missing or []
