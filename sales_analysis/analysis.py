"""Business metrics on the cleaned sales data."""
from __future__ import annotations

import pandas as pd


def kpis(df: pd.DataFrame) -> dict[str, float]:
    orders = df["order_id"].nunique()
    revenue = df["revenue"].sum()
    return {
        "total_revenue": round(revenue, 2),
        "orders": orders,
        "customers": df["customer_id"].nunique(),
        "units_sold": int(df["quantity"].sum()),
        "avg_order_value": round(revenue / orders, 2) if orders else 0.0,
    }


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("month", observed=True)
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )
    out["mom_growth_pct"] = (out["revenue"].pct_change() * 100).round(1).astype("float64")
    return out


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        df.groupby("product", observed=True)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
        .nlargest(n, "revenue")
        .reset_index()
    )


def revenue_share(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = df.groupby(by, observed=True)["revenue"].sum().sort_values(ascending=False)
    return pd.DataFrame(
        {"revenue": out.round(2), "share_pct": (out / out.sum() * 100).round(1)}
    ).reset_index()


def category_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="region",
        columns="category",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    ).round(0)


def repeat_customer_rate(df: pd.DataFrame) -> float:
    orders_per_customer = df.groupby("customer_id")["order_id"].nunique()
    return round((orders_per_customer > 1).mean() * 100, 1)
