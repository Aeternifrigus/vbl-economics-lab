import pandas as pd
import pytest
from synthetic import daily_inputs, vbl_table
from typer.testing import CliRunner

from pwl import cli, snapshot
from pwl.config import DATA_RAW
from pwl.models import lagged


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every module-level path at a temp dir so tests never touch the real repo."""
    ext = tmp_path / "external"
    paths = {
        "DATA_EXT": ext,
        "FEATURES_CSV": ext / "features_quarterly.csv",
        "MODELS": tmp_path / "models",
        "REPORTS": tmp_path / "reports",
        "VBL_CSV": tmp_path / "vbl_quarterly.csv",
    }
    for name, value in paths.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(snapshot, "DATA_EXT", ext)
    monkeypatch.setattr(cli.settings, "bootstrap_samples", 40)  # keep integration tests fast
    monkeypatch.setattr(lagged, "ALPHAS", (0.1, 10.0, 1000.0))
    ext.mkdir()
    return paths


@pytest.fixture
def world():
    return daily_inputs()


@pytest.fixture
def synthetic_vbl(world, sandbox):
    brent, fx, _ = world
    table = vbl_table(brent, fx)
    table.to_csv(sandbox["VBL_CSV"], index=False)
    return table


@pytest.fixture
def real_vbl(sandbox):
    sandbox["VBL_CSV"].write_text((DATA_RAW / "vbl_quarterly.csv").read_text())
    return sandbox["VBL_CSV"]


@pytest.fixture
def short_vbl(sandbox):
    """Real rows, but only nine of them, so the 20-quarter guard applies."""
    pd.read_csv(DATA_RAW / "vbl_quarterly.csv").tail(9).to_csv(sandbox["VBL_CSV"], index=False)
    return sandbox["VBL_CSV"]


@pytest.fixture
def fake_apis(monkeypatch, world):
    brent, fx, weather = world
    monkeypatch.setattr(cli.settings, "fred_api_key", "test-key")
    monkeypatch.setattr(
        cli.fred,
        "fetch_series",
        lambda series_id, key=None, start=None: (
            brent if series_id == cli.fred.BRENT else fx
        ).copy(),
    )
    monkeypatch.setattr(
        cli.openmeteo, "fetch_weighted", lambda start, end, cities=None: weather.copy()
    )
