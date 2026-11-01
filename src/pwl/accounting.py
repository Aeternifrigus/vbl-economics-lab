"""Shared accounting helpers so every module uses the same revenue basis."""

import pandas as pd


def net_revenue(df: pd.DataFrame) -> pd.Series:
    """Revenue from operations less excise duty, the base VBL uses for its reported margins.

    Tables without an `excise_duty_cr` column are treated as having none.
    """
    excise = df["excise_duty_cr"] if "excise_duty_cr" in df.columns else 0.0
    return df["revenue_cr"] - excise
