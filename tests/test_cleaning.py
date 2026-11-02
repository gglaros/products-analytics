import pandas as pd
import pytest

from sales_analysis.cleaning import clean, parse_price


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 2, 3, 4, 5],
            "order_date": ["2025-01-05", "2025-01-06", "2025-01-06", "2025-02-01", "2025-02-02", "bad"],
            "customer_id": [10, 11, 11, 10, 12, 13],
            "product": ["Pen Set", "Pen Set", "Pen Set", "Pen Set", "Pen Set", "Pen Set"],
            "category": ["Stationery"] * 6,
            "quantity": [2, 1, 1, 0, 3, 1],
            "unit_price": ["10", "€8,00", "€8,00", "9", None, "9"],
            "region": [" ATTICA ", "Crete", "Crete", None, "attica", "Crete"],
            "channel": ["Web", "web", "web", "mobile", "MOBILE", "web"],
        }
    )


def test_parse_price():
    out = parse_price(pd.Series(["€12,50", "3.2", None, "abc"]))
    assert out.tolist()[:2] == [12.5, 3.2]
    assert out.isna().tolist()[2:] == [True, True]


def test_clean_removes_duplicates_and_invalid(raw):
    df, report = clean(raw)
    assert report["duplicates_removed"] == 1
    assert report["invalid_rows_dropped"] == 2  # quantity 0 + bad date
    assert len(df) == 3
    assert df["order_id"].is_unique


def test_clean_normalises_text(raw):
    df, _ = clean(raw)
    assert set(df["region"]) <= {"Attica", "Crete", "Unknown"}
    assert set(df["channel"]) <= {"web", "mobile"}


def test_missing_price_imputed_with_product_median(raw):
    df, _ = clean(raw)
    row = df.loc[df["order_id"] == 4].iloc[0]
    assert row["unit_price"] == pytest.approx(9.0)  # median of 10, 8, 9, 9
    assert row["revenue"] == pytest.approx(27.0)
