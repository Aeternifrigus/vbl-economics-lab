import numpy as np
import pandas as pd
import pytest

from pwl.config import DATA_RAW
from pwl.exceptions import InsufficientDataError
from pwl.models.leverage import fit_leverage, predict_opm_pp, variance_decomposition
from pwl.validate import load_vbl


def test_decomposition_shares_sum_to_one():
    d = variance_decomposition(load_vbl(DATA_RAW / "vbl_quarterly.csv"))
    total = d["share_gross_margin"] + d["share_opex_ratio"] + d["share_covariance"]
    assert abs(total - 1.0) < 1e-9


def test_opex_ratio_drives_margin_swing_on_shipped_data():
    """On the reconciled quarters the opex ratio swings about twice as much as gross margin."""
    d = variance_decomposition(load_vbl(DATA_RAW / "vbl_quarterly.csv"))
    assert d["opex_ratio_range_pp"] > 1.5 * d["gm_range_pp"]
    assert d["share_opex_ratio"] > d["share_gross_margin"]


def test_shipped_fit_has_sensible_signs():
    fit = fit_leverage(load_vbl(DATA_RAW / "vbl_quarterly.csv"))
    assert fit.fixed_cr > 0
    assert 0.0 < fit.var_cost_ratio < 0.5
    assert fit.r2 > 0.5


def test_recovers_known_cost_structure():
    rng = np.random.default_rng(0)
    rev = rng.uniform(4000, 9000, 40)
    opex = 1200 + 0.12 * rev + rng.normal(0, 20, 40)
    df = pd.DataFrame({"revenue_cr": rev, "opex_cr": opex, "gross_profit_cr": 0.56 * rev})
    fit = fit_leverage(df)
    assert abs(fit.fixed_cr - 1200) < 150
    assert abs(fit.var_cost_ratio - 0.12) < 0.02
    assert fit.ci90_var[0] < 0.12 < fit.ci90_var[1]


def test_prediction_matches_identity_without_noise():
    rev = np.array([4000.0, 5000.0, 6000.0, 8000.0, 9000.0])
    df = pd.DataFrame(
        {"revenue_cr": rev, "opex_cr": 1500 + 0.1 * rev, "gross_profit_cr": 0.55 * rev}
    )
    fit = fit_leverage(df)
    pred = predict_opm_pp(fit, 7000.0, 55.0)
    assert abs(pred - (55.0 - (1500 + 700) / 7000 * 100)) < 1e-6


def test_needs_four_quarters():
    df = pd.DataFrame(
        {
            "revenue_cr": [1.0, 2.0, 3.0],
            "opex_cr": [1.0, 1.0, 1.0],
            "gross_profit_cr": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(InsufficientDataError):
        fit_leverage(df)
