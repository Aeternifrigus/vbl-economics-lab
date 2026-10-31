import json
import logging
import urllib.error

import pandas as pd
import pytest

from pwl import http, snapshot
from pwl.exceptions import (
    ConfigurationError,
    DataValidationError,
    IngestionError,
    InsufficientDataError,
    ModelTrainingError,
)
from pwl.ingest import fred, openmeteo
from pwl.logging_config import ConsoleFormatter, JSONFormatter, get_logger, setup_logging
from pwl.settings import settings


def _record(level=logging.INFO, msg="hello"):
    return logging.LogRecord("pwl.test", level, __file__, 1, msg, None, None)


def test_json_formatter_emits_parseable_utc_timestamp():
    out = json.loads(JSONFormatter().format(_record()))
    assert out["message"] == "hello" and out["timestamp"].endswith("+00:00")


def test_console_formatter_and_setup_logging(monkeypatch):
    assert "hello" in ConsoleFormatter().format(_record(logging.WARNING))
    for fmt, cls in (("json", JSONFormatter), ("console", ConsoleFormatter)):
        monkeypatch.setattr(settings, "log_format", fmt)
        setup_logging()
        assert isinstance(logging.getLogger().handlers[0].formatter, cls)
    assert get_logger("x").name == "x"


def test_exceptions_carry_structured_details():
    assert InsufficientDataError("m", available=3, required=20).details == {
        "available": 3,
        "required": 20,
    }
    assert DataValidationError("m", ["a"]).problems == ["a"]
    assert IngestionError("m", "host", ValueError("x")).details["source"] == "host"
    assert ModelTrainingError("m", "ridge").details["model_type"] == "ridge"
    assert ConfigurationError("m", ["K"]).missing == ["K"]


class _Resp:
    status = 200

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_json_success_and_retry_then_failure(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    monkeypatch.setattr(http.urllib.request, "urlopen", lambda *a, **k: _Resp(b'{"ok": 1}'))
    assert http.get_json("https://example.com", {"a": 1}) == {"ok": 1}

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    monkeypatch.setattr(http.urllib.request, "urlopen", flaky)
    with pytest.raises(IngestionError) as err:
        http.get_json("https://example.com/path?api_key=SECRET", retries=2)
    assert calls["n"] == 2
    assert "SECRET" not in str(err.value) and "SECRET" not in repr(err.value.details)


def test_fred_requires_key(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", None)
    with pytest.raises(IngestionError):
        fred.fetch_series("X")


def test_fred_skips_bad_values():
    df = fred.parse_observations(
        {
            "observations": [
                {"date": "2024-01-01", "value": "abc"},
                {"date": "2024-01-02", "value": "5"},
            ]
        }
    )
    assert df["value"].tolist() == [5.0]
    assert fred.parse_observations({}).empty


def test_openmeteo_fetch_weighted_uses_every_city(monkeypatch):
    seen = []

    def fake(url, params=None, **kw):
        seen.append(params["latitude"])
        return {
            "daily": {
                "time": ["2024-05-01"],
                "temperature_2m_max": [40.0],
                "precipitation_sum": [0.0],
            }
        }

    monkeypatch.setattr(openmeteo, "get_json", fake)
    out = openmeteo.fetch_weighted(
        "2024-05-01", "2024-05-01", {"A": (1.0, 2.0, 3.0), "B": (4.0, 5.0, 1.0)}
    )
    assert seen == [1.0, 4.0] and out["heat_share"].iloc[0] == 1.0
    assert openmeteo.weight_cities([], {}).empty


def test_verify_snapshot_failure_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "DATA_EXT", tmp_path)
    assert not snapshot.verify_snapshot("a")  # no manifest
    snapshot.save_snapshot(pd.DataFrame({"x": [1]}), "a", "test")
    assert snapshot.verify_snapshot("a")
    assert not snapshot.verify_snapshot("missing")  # not in manifest
    (tmp_path / "a.csv").unlink()
    assert not snapshot.verify_snapshot("a")  # file missing
