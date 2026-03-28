"""
Unit tests for Phase 1: data loading and Spark aggregation.
All tests use synthetic in-memory fixtures — no real dataset required.
"""
import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = {"view", "cart", "purchase", "remove_from_cart"}

SAMPLE_CSV = """event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2020-01-01 10:00:00 UTC,view,1001,101,electronics.phone,samsung,299.99,u1,s1
2020-01-01 10:05:00 UTC,cart,1001,101,electronics.phone,samsung,299.99,u1,s1
2020-01-01 10:10:00 UTC,purchase,1001,101,electronics.phone,samsung,299.99,u1,s1
2020-01-01 11:00:00 UTC,view,1002,102,electronics.laptop,apple,999.00,u2,s2
2020-01-01 11:05:00 UTC,view,1003,102,electronics.laptop,dell,799.00,u2,s2
2020-01-02 09:00:00 UTC,purchase,1002,102,electronics.laptop,apple,999.00,u3,s3
2020-01-02 09:30:00 UTC,view,1004,103,apparel.shoes,nike,89.99,u4,s4
"""

BAD_ROWS_CSV = """event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2020-01-01 10:00:00 UTC,view,1001,101,electronics,samsung,10.0,,s1
2020-01-01 10:01:00 UTC,view,,101,electronics,samsung,10.0,u2,s2
2020-01-01 10:02:00 UTC,view,1003,101,electronics,samsung,10.0,u3,s3
2020-01-01 10:03:00 UTC,INVALID_TYPE,1004,101,electronics,samsung,10.0,u4,s4
"""


@pytest.fixture
def sample_csv_path(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(SAMPLE_CSV)
    return tmp_path  # return directory, not file


@pytest.fixture
def bad_rows_csv_path(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(BAD_ROWS_CSV)
    return tmp_path


@pytest.fixture
def sample_df():
    """Clean DataFrame matching what load_events() would return."""
    from src.data.loader import load_events
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "events.csv"
        csv_path.write_text(SAMPLE_CSV)
        return load_events(raw_path=Path(d), save=False)


# ---------------------------------------------------------------------------
# Test 1: No null user_ids after cleaning
# ---------------------------------------------------------------------------

def test_no_null_user_ids_after_cleaning(sample_csv_path):
    from src.data.loader import load_events
    df = load_events(raw_path=sample_csv_path, save=False)
    assert df["user_id"].isna().sum() == 0, "Found null user_ids after cleaning"


# ---------------------------------------------------------------------------
# Test 2: All event_type values are valid
# ---------------------------------------------------------------------------

def test_event_types_are_valid(bad_rows_csv_path):
    from src.data.loader import load_events
    df = load_events(raw_path=bad_rows_csv_path, save=False)
    invalid = set(df["event_type"].unique()) - VALID_EVENT_TYPES
    assert not invalid, f"Found invalid event types: {invalid}"


# ---------------------------------------------------------------------------
# Test 3: SparkAggregator output has required columns
# ---------------------------------------------------------------------------

REQUIRED_USER_FEATURE_COLUMNS = {
    "user_id", "total_sessions", "total_views", "total_carts",
    "total_purchases", "total_revenue", "avg_session_length",
    "days_active", "first_seen", "last_seen",
    "favorite_category", "avg_price_viewed",
}


def test_user_features_has_required_columns(sample_df, tmp_path):
    from src.data.spark_aggregator import SparkAggregator

    parquet_path = tmp_path / "events.parquet"
    sample_df.to_parquet(parquet_path, index=False)

    agg = SparkAggregator(use_pandas=True)
    result = agg.compute_user_features(events_path=parquet_path, save=False)
    missing = REQUIRED_USER_FEATURE_COLUMNS - set(result.columns)
    assert not missing, f"Missing columns: {missing}"


# ---------------------------------------------------------------------------
# Test 4: No negative counts in user features
# ---------------------------------------------------------------------------

def test_user_features_no_negative_counts(sample_df, tmp_path):
    from src.data.spark_aggregator import SparkAggregator

    parquet_path = tmp_path / "events.parquet"
    sample_df.to_parquet(parquet_path, index=False)

    agg = SparkAggregator(use_pandas=True)
    result = agg.compute_user_features(events_path=parquet_path, save=False)
    for col in ["total_views", "total_carts", "total_purchases", "total_sessions"]:
        assert (result[col] >= 0).all(), f"Negative values found in {col}"


# ---------------------------------------------------------------------------
# Test 5: total_revenue is non-negative
# ---------------------------------------------------------------------------

def test_revenue_is_nonnegative(sample_df, tmp_path):
    from src.data.spark_aggregator import SparkAggregator

    parquet_path = tmp_path / "events.parquet"
    sample_df.to_parquet(parquet_path, index=False)

    agg = SparkAggregator(use_pandas=True)
    result = agg.compute_user_features(events_path=parquet_path, save=False)
    assert (result["total_revenue"] >= 0).all(), "Negative total_revenue values found"
