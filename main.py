"""Run the full pipeline: load -> clean -> analyse -> charts + markdown report.

Usage:
    python main.py [--input data/raw/sales.csv] [--reports reports]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sales_analysis import analysis, plotting
from sales_analysis.cleaning import clean, load_raw


def build_report(df, cleaning_report: dict) -> str:
    k = analysis.kpis(df)
    monthly = analysis.monthly_revenue(df)
    best = monthly.loc[monthly["revenue"].idxmax()]

    lines = [
        "# Sales report",
        "",
        "## Data cleaning",
        *[f"- **{key.replace('_', ' ')}**: {val:,}" for key, val in cleaning_report.items()],
        "",
        "## KPIs",
        f"- Total revenue: **€{k['total_revenue']:,.2f}**",
        f"- Orders: **{k['orders']:,}** · Customers: **{k['customers']:,}** · Units: **{k['units_sold']:,}**",
        f"- Average order value: **€{k['avg_order_value']:,.2f}**",
        f"- Repeat-customer rate: **{analysis.repeat_customer_rate(df)}%**",
        f"- Best month: **{best['month']}** (€{best['revenue']:,.0f})",
        "",
        "## Monthly revenue",
        monthly.assign(
            mom_growth_pct=monthly["mom_growth_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%")
        ).to_markdown(index=False, floatfmt=",.1f"),
        "",
        "## Top 5 products",
        analysis.top_products(df).to_markdown(index=False, floatfmt=",.0f"),
        "",
        "## Revenue by channel",
        analysis.revenue_share(df, "channel").to_markdown(index=False, floatfmt=",.1f"),
        "",
        "## Category × region (€)",
        analysis.category_by_region(df).to_markdown(floatfmt=",.0f"),
        "",
        "![Monthly revenue](figures/monthly_revenue.png)",
        "![Top products](figures/top_products.png)",
        "![Region share](figures/region_share.png)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/sales.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    raw = load_raw(args.input)
    df, cleaning_report = clean(raw)

    processed = Path("data/processed/sales_clean.csv")
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed, index=False)

    figs = args.reports / "figures"
    plotting.plot_monthly(analysis.monthly_revenue(df), figs / "monthly_revenue.png")
    plotting.plot_top_products(analysis.top_products(df), figs / "top_products.png")
    plotting.plot_region_share(analysis.revenue_share(df, "region"), figs / "region_share.png")

    report_path = args.reports / "report.md"
    report_path.write_text(build_report(df, cleaning_report), encoding="utf-8")

    print(f"Cleaned {cleaning_report['rows_in']:,} -> {cleaning_report['rows_out']:,} rows")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
