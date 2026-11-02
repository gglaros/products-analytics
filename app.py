
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sales_analysis import analysis
from sales_analysis.cleaning import clean, load_raw

DATA_PATH = Path(__file__).parent / "data" / "raw" / "sales.csv"

st.set_page_config(page_title="Sales Analysis", page_icon="🛒", layout="wide")


@st.cache_data
def get_data(file) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = load_raw(file)
    df, report = clean(raw)
    return raw, df, report


st.title("🛒 E-shop Sales Analysis")
st.caption("pandas pipeline: Load → Clean → Analyze → Visualize")

uploaded = st.sidebar.file_uploader("Upload your own sales CSV (optional)", type="csv")
raw, df_all, cleaning_report = get_data(uploaded or DATA_PATH)

st.sidebar.header("Filters")
regions = st.sidebar.multiselect(
    "Region", sorted(df_all["region"].unique()), default=sorted(df_all["region"].unique())
)
channels = st.sidebar.multiselect(
    "Channel", sorted(df_all["channel"].unique()), default=sorted(df_all["channel"].unique())
)
min_d, max_d = df_all["order_date"].min().date(), df_all["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
start, end = (date_range if len(date_range) == 2 else (min_d, max_d))

df = df_all[
    df_all["region"].isin(regions)
    & df_all["channel"].isin(channels)
    & df_all["order_date"].between(pd.Timestamp(start), pd.Timestamp(end))
]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

tab_load, tab_clean, tab_analyze, tab_viz = st.tabs(
    ["1. Load", "2. Clean", "3. Analyze", "4. Visualize"]
)

with tab_load:
    st.subheader("Raw data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Duplicate rows", f"{raw.duplicated().sum():,}")
    st.dataframe(raw.head(50), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values per column**")
        st.dataframe(raw.isna().sum().rename("missing"), width="stretch")
    with right:
        st.markdown("**Column types**")
        st.dataframe(raw.dtypes.astype(str).rename("dtype"), width="stretch")

    col = st.selectbox("value_counts() for column", ["region", "channel", "product", "category"])
    st.dataframe(raw[col].value_counts(dropna=False), width="stretch")

# ---------- 2. Clean ----------
with tab_clean:
    st.subheader("What the cleaning step did")
    cols = st.columns(len(cleaning_report))
    for c, (key, val) in zip(cols, cleaning_report.items()):
        c.metric(key.replace("_", " ").capitalize(), f"{val:,}")

    st.markdown(
        """
- Removed exact **duplicate rows**
- Normalised text (`" ATTICA "` → `Attica`, `Web` → `web`); missing region → `Unknown`
- Parsed prices like `€12,50` → `12.5`; negative/missing prices → **median price of the same product**
- Dropped orders with **quantity ≤ 0**
- Added `revenue = quantity × unit_price`, `month`, `weekday`
"""
    )
    st.markdown("**Cleaned data**")
    st.dataframe(df_all.head(50), width="stretch")

with tab_analyze:
    k = analysis.kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue", f"€{k['total_revenue']:,.0f}")
    c2.metric("Orders", f"{k['orders']:,}")
    c3.metric("Customers", f"{k['customers']:,}")
    c4.metric("Avg order value", f"€{k['avg_order_value']:,.2f}")
    c5.metric("Repeat customers", f"{analysis.repeat_customer_rate(df)}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top 5 products**")
        st.dataframe(analysis.top_products(df), hide_index=True, width="stretch")
        st.markdown("**Revenue by channel**")
        st.dataframe(analysis.revenue_share(df, "channel"), hide_index=True, width="stretch")
    with right:
        st.markdown("**Monthly revenue**")
        monthly = analysis.monthly_revenue(df)
        st.dataframe(
            monthly.assign(month=monthly["month"].astype(str)),
            hide_index=True,
            width="stretch",
        )
    st.markdown("**Category × region (€)**")
    st.dataframe(analysis.category_by_region(df), width="stretch")


with tab_viz:
    monthly = analysis.monthly_revenue(df)
    st.markdown("**Monthly revenue**")
    st.line_chart(monthly.assign(month=monthly["month"].dt.to_timestamp()), x="month", y="revenue")

    left, right = st.columns(2)
    with left:
        st.markdown("**Revenue by product**")
        by_product = (
            df.groupby("product", observed=True)["revenue"].sum().sort_values(ascending=False)
        )
        st.bar_chart(by_product.reset_index(), x="product", y="revenue", horizontal=True, sort="-revenue")
    with right:
        st.markdown("**Revenue share by region (%)**")
        share = analysis.revenue_share(df, "region")
        st.bar_chart(share, x="region", y="share_pct", horizontal=True, sort="-share_pct")

    st.download_button(
        "⬇️ Download cleaned & filtered CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="sales_clean.csv",
        mime="text/csv",
    )
