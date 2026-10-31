import pandas as pd

REQUIRED = ["quarter", "revenue_cr", "gross_profit_cr", "opex_cr", "ebitda_cr", "source_url"]


def load_vbl(path):
    df = pd.read_csv(path)
    df["period"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df.sort_values("period").reset_index(drop=True)


def check_vbl(df, tol_cr=3.0):
    """Return a list of problems found in the VBL quarterly table (empty list means clean)."""
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df["quarter"].duplicated().any():
        problems.append("duplicate quarters")
    for col in ["revenue_cr", "gross_profit_cr", "opex_cr", "ebitda_cr"]:
        if (df[col] <= 0).any():
            problems.append(f"non-positive values in {col}")
    gap = (df["gross_profit_cr"] - df["opex_cr"] - df["ebitda_cr"]).abs()
    for q in df.loc[gap > tol_cr, "quarter"]:
        problems.append(f"{q}: gross profit minus opex differs from EBITDA by more than {tol_cr} crore")
    gm = df["gross_profit_cr"] / df["revenue_cr"] * 100
    for q in df.loc[(gm < 40) | (gm > 70), "quarter"]:
        problems.append(f"{q}: gross margin outside 40 to 70 percent")
    if df["source_url"].isna().any() or (df["source_url"].astype(str).str.len() == 0).any():
        problems.append("rows without a source_url")
    periods = pd.PeriodIndex(df["quarter"], freq="Q").sort_values()
    if len(periods) > 1 and (periods[1:] - periods[:-1]).map(lambda x: x.n).max() > 1:
        problems.append("gaps between quarters")
    return problems
