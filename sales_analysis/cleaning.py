"""Data loading and cleaning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"unit_price": "string"})


def parse_price(s: pd.Series) -> pd.Series:
    """Turn values like '€12,50', '12.5', None into floats."""
    cleaned = (
        s.astype("string")
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
   
    report: dict[str, int] = {"rows_in": len(df)}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    df["region"] = df["region"].str.strip().str.title()
    df["channel"] = df["channel"].str.strip().str.lower()
    df["region"] = df["region"].fillna("Unknown")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["unit_price"] = parse_price(df["unit_price"])

    missing = df["unit_price"].isna() | (df["unit_price"] <= 0)
    report["prices_imputed"] = int(missing.sum())
    df.loc[df["unit_price"] <= 0, "unit_price"] = pd.NA
    df["unit_price"] = df["unit_price"].fillna(
        df.groupby("product")["unit_price"].transform("median")
    )

    invalid = (df["quantity"] <= 0) | df["order_date"].isna() | df["unit_price"].isna()
    report["invalid_rows_dropped"] = int(invalid.sum())
    df = df.loc[~invalid]

    df = df.assign(
        revenue=lambda d: (d["quantity"] * d["unit_price"]).round(2),
        month=lambda d: d["order_date"].dt.to_period("M"),
        weekday=lambda d: d["order_date"].dt.day_name(),
    )
    for col in ("product", "category", "region", "channel"):
        df[col] = df[col].astype("category")

    report["rows_out"] = len(df)
    return df.sort_values("order_date").reset_index(drop=True), report
