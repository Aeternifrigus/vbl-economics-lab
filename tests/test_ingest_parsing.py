from pwl.ingest import fred, openmeteo


def test_fred_parser_skips_missing_values():
    payload = {
        "observations": [
            {"date": "2024-01-02", "value": "80.5"},
            {"date": "2024-01-03", "value": "."},
        ]
    }
    df = fred.parse_observations(payload)
    assert len(df) == 1 and df["value"].iloc[0] == 80.5


def test_openmeteo_parser_and_weighting():
    payload = {
        "daily": {
            "time": ["2024-05-01", "2024-05-02"],
            "temperature_2m_max": [41.0, None],
            "precipitation_sum": [0.0, 3.0],
        }
    }
    a = openmeteo.parse_daily(payload, "A")
    b = openmeteo.parse_daily(payload, "B")
    b["tmax_c"] = [36.0, 37.0]
    out = openmeteo.weight_cities([a, b], {"A": 3.0, "B": 1.0}, threshold=38.0)
    first = out.iloc[0]
    assert abs(first["tmax_c"] - (41 * 3 + 36) / 4) < 1e-9
    assert abs(first["heat_share"] - 0.75) < 1e-9
    assert out.iloc[1]["n_cities"] == 1


def test_fetch_uses_http_layer(monkeypatch):
    seen = {}

    def fake(url, params=None, **kw):
        seen["params"] = params
        return {"observations": [{"date": "2024-01-02", "value": "1.0"}]}

    monkeypatch.setattr(fred, "get_json", fake)
    df = fred.fetch_series("X", "KEY", "2020-01-01")
    assert seen["params"]["series_id"] == "X" and len(df) == 1
