"""
Unit tests for Phase 2: metric computation.
"""
import pandas as pd
import pytest

from src.experiments.assignment import ExperimentAssigner
from src.experiments.metrics import MetricComputer

EXP_ID = "exp_A"
VARIANTS = ["control", "treatment"]


@pytest.fixture
def assigner():
    return ExperimentAssigner()


@pytest.fixture
def metric_computer():
    return MetricComputer()


@pytest.fixture
def synthetic_data(assigner):
    """
    Build a small synthetic events + assignments DataFrame for testing.

    200 users split ~50/50 across control/treatment.
    ~20% of users have a purchase event; ~40% have a cart event.
    """
    import random
    rng = random.Random(42)

    users = [str(i) for i in range(200)]
    assignments = assigner.assign_all_users(users, EXP_ID, VARIANTS)

    events = []
    for uid in users:
        # All users view something
        events.append({
            "user_id": uid, "event_type": "view",
            "price": rng.uniform(10, 500), "user_session": f"{uid}_s1",
        })
        # 40% add to cart
        if rng.random() < 0.4:
            events.append({
                "user_id": uid, "event_type": "cart",
                "price": rng.uniform(10, 500), "user_session": f"{uid}_s1",
            })
        # 20% purchase
        if rng.random() < 0.2:
            events.append({
                "user_id": uid, "event_type": "purchase",
                "price": rng.uniform(10, 500), "user_session": f"{uid}_s1",
            })

    events_df = pd.DataFrame(events)
    return events_df, assignments


# ---------------------------------------------------------------------------
# Test 1: Conversion rates are in [0, 1]
# ---------------------------------------------------------------------------

def test_conversion_rate_between_zero_and_one(metric_computer, synthetic_data):
    events_df, assignments_df = synthetic_data
    result = metric_computer.compute_conversion_rate(events_df, assignments_df, EXP_ID)
    assert len(result) == len(VARIANTS), "Should have one row per variant"
    assert (result["conversion_rate"] >= 0).all()
    assert (result["conversion_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# Test 2: Revenue per user is non-negative
# ---------------------------------------------------------------------------

def test_revenue_per_user_nonnegative(metric_computer, synthetic_data):
    events_df, assignments_df = synthetic_data
    result = metric_computer.compute_revenue_per_user(events_df, assignments_df, EXP_ID)
    assert (result["mean_revenue"] >= 0).all(), "Mean revenue per user must be >= 0"


# ---------------------------------------------------------------------------
# Test 3: compute_all_metrics returns all required keys
# ---------------------------------------------------------------------------

def test_compute_all_metrics_returns_all_keys(metric_computer, synthetic_data):
    events_df, assignments_df = synthetic_data
    result = metric_computer.compute_all_metrics(events_df, assignments_df, EXP_ID)
    assert "conversion_rate" in result
    assert "revenue_per_user" in result
    assert "add_to_cart_rate" in result
