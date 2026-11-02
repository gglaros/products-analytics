"""Generate a synthetic, intentionally *messy* e-shop sales dataset.

The mess (duplicates, missing values, inconsistent casing, bad prices)
is deliberate, so the cleaning step has real work to do.

Usage:
    python scripts/generate_data.py --rows 5000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = {
    # product: (category, base_price)
    "Laptop": ("Electronics", 850.0),
    "Headphones": ("Electronics", 79.0),
    "Smartphone": ("Electronics", 520.0),
    "Monitor": ("Electronics", 210.0),
    "Coffee Maker": ("Home", 95.0),
    "Desk Lamp": ("Home", 35.0),
    "Office Chair": ("Home", 180.0),
    "Backpack": ("Accessories", 45.0),
    "Water Bottle": ("Accessories", 15.0),
    "Notebook": ("Stationery", 4.5),
    "Pen Set": ("Stationery", 9.0),
}
REGIONS = ["Attica", "Central Macedonia", "Crete", "Thessaly", "Western Greece"]
CHANNELS = ["web", "mobile", "marketplace"]


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(PRODUCTS)

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, rows), unit="D"
    )
    product = rng.choice(names, rows, p=_weights(len(names), rng))
    # Seasonal boost in Nov/Dec -> more units
    month_boost = np.where(dates.month.isin([11, 12]), 1.6, 1.0)
    quantity = np.maximum(1, rng.poisson(1.4 * month_boost)).astype(int)
    base = np.array([PRODUCTS[p][1] for p in product])
    unit_price = np.round(base * rng.normal(1.0, 0.07, rows), 2)

    df = pd.DataFrame(
        {
            "order_id": np.arange(100_000, 100_000 + rows),
            "order_date": dates.strftime("%Y-%m-%d"),
            "customer_id": rng.integers(1, 1200, rows),
            "product": product,
            "category": [PRODUCTS[p][0] for p in product],
            "quantity": quantity,
            "unit_price": unit_price,
            "region": rng.choice(REGIONS, rows, p=[0.42, 0.22, 0.14, 0.12, 0.10]),
            "channel": rng.choice(CHANNELS, rows, p=[0.5, 0.35, 0.15]),
        }
    )
    return _add_noise(df, rng)


def _weights(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.uniform(0.5, 2.0, n)
    return w / w.sum()


def _add_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    def pick(frac: float) -> np.ndarray:
        return rng.choice(df.index, int(n * frac), replace=False)

    df["unit_price"] = df["unit_price"].astype(object)
    df.loc[pick(0.02), "unit_price"] = np.nan                 # missing prices
    df.loc[pick(0.005), "unit_price"] = -1                    # invalid prices
    idx = pick(0.03)                                          # prices as "€12,50" strings
    df.loc[idx, "unit_price"] = [f"€{float(v):.2f}".replace(".", ",")
                                 if pd.notna(v) and float(v) > 0 else v
                                 for v in df.loc[idx, "unit_price"]]
    df.loc[pick(0.03), "region"] = np.nan                     # missing region
    idx = pick(0.05)                                          # inconsistent casing / spaces
    df.loc[idx, "region"] = df.loc[idx, "region"].str.upper() + " "
    idx = pick(0.04)
    df.loc[idx, "channel"] = df.loc[idx, "channel"].str.capitalize()
    df.loc[pick(0.01), "quantity"] = 0                        # zero-quantity rows

    dupes = df.sample(frac=0.02, random_state=int(rng.integers(1e6)))  # duplicate rows
    return pd.concat([df, dupes]).sample(frac=1, random_state=1).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/sales.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
