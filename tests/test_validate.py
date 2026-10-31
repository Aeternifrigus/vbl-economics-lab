import pandas as pd

from pwl.config import DATA_RAW
from pwl.validate import check_vbl, load_vbl


def test_shipped_data_is_clean():
    assert check_vbl(load_vbl(DATA_RAW / "vbl_quarterly.csv")) == []


def test_detects_broken_identity():
    df = load_vbl(DATA_RAW / "vbl_quarterly.csv")
    df.loc[0, "ebitda_cr"] += 50
    assert any("differs from EBITDA" in p for p in check_vbl(df))


def test_detects_duplicates_and_gaps():
    df = load_vbl(DATA_RAW / "vbl_quarterly.csv")
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "duplicate quarters" in check_vbl(dup)
    gap = df.drop(index=2).reset_index(drop=True)
    assert "gaps between quarters" in check_vbl(gap)


def test_every_row_has_a_source():
    df = load_vbl(DATA_RAW / "vbl_quarterly.csv")
    assert df["source_url"].str.startswith("http").all()
