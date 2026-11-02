import pandas as pd
import pytest

from sales_analysis import analysis


@pytest.fixture
def df() -> pd.DataFrame:
    d = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "order_date": pd.to_datetime(["2025-01-01", "2025-01-15", "2025-02-01", "2025-02-10"]),
            "customer_id": [1, 1, 2, 3],
            "product": ["A", "B", "A", "C"],
            "region": ["Attica", "Crete", "Attica", "Attica"],
            "channel": ["web", "web", "mobile", "web"],
            "quantity": [1, 2, 1, 4],
            "revenue": [100.0, 50.0, 100.0, 40.0],
        }
    )
    d["month"] = d["order_date"].dt.to_period("M")
    return d


def test_kpis(df):
    k = analysis.kpis(df)
    assert k["total_revenue"] == 290.0
    assert k["orders"] == 4
    assert k["customers"] == 3
    assert k["avg_order_value"] == 72.5


def test_monthly_revenue_growth(df):
    m = analysis.monthly_revenue(df)
    assert m["revenue"].tolist() == [150.0, 140.0]
    assert m["mom_growth_pct"].iloc[1] == pytest.approx(-6.7)


def test_top_products(df):
    top = analysis.top_products(df, n=2)
    assert top["product"].tolist() == ["A", "B"]


def test_revenue_share_sums_to_100(df):
    share = analysis.revenue_share(df, "region")
    assert share["share_pct"].sum() == pytest.approx(100, abs=0.2)


def test_repeat_customer_rate(df):
    assert analysis.repeat_customer_rate(df) == pytest.approx(33.3)
