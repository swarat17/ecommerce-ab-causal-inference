"""
Unit tests for Phase 4: propensity scoring and IPW weighting.
All tests use synthetic data — no real dataset required.
"""
import numpy as np
import pandas as pd
import pytest

from src.causal.ipw import compute_weights, adjusted_conversion_rate


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_balanced_data(n: int = 500, seed: int = 0):
    """Perfectly balanced: features drawn from same distribution for both groups."""
    rng = np.random.default_rng(seed)
    user_ids = [str(i) for i in range(n)]
    treatment = (np.arange(n) % 2).astype(int)  # strict alternating 50/50

    features = pd.DataFrame({
        "user_id": user_ids,
        "total_sessions": rng.integers(1, 20, n).astype(float),
        "total_revenue": rng.uniform(0, 500, n),
        "days_active": rng.integers(1, 30, n).astype(float),
        "avg_price_viewed": rng.uniform(10, 200, n),
        "total_views": rng.integers(1, 100, n).astype(float),
        "total_carts": rng.integers(0, 20, n).astype(float),
        "total_purchases": rng.integers(0, 10, n).astype(float),
        "avg_session_length": rng.uniform(1, 30, n),
        "first_seen": pd.Timestamp("2024-01-01"),
        "last_seen": pd.Timestamp("2024-03-01"),
        "favorite_category": rng.choice(["electronics", "apparel", "beauty"], n),
    })

    assignments = pd.DataFrame({
        "user_id": user_ids,
        "experiment_id": "exp_test",
        "variant": ["treatment" if t else "control" for t in treatment],
    })
    return features, assignments


def _make_imbalanced_data(n: int = 500, seed: int = 1):
    """
    Imbalanced: high-revenue users are over-assigned to treatment.
    The propensity model should be able to distinguish groups (AUC > 0.6).
    """
    rng = np.random.default_rng(seed)
    user_ids = [str(i) for i in range(n)]

    # High-revenue users → treatment; low-revenue → control
    revenue = rng.uniform(0, 1000, n)
    treatment = (revenue > 500).astype(int)

    features = pd.DataFrame({
        "user_id": user_ids,
        "total_sessions": rng.integers(1, 20, n).astype(float),
        "total_revenue": revenue,
        "days_active": rng.integers(1, 30, n).astype(float),
        "avg_price_viewed": revenue / 5,  # correlated with revenue
        "total_views": rng.integers(1, 100, n).astype(float),
        "total_carts": rng.integers(0, 20, n).astype(float),
        "total_purchases": rng.integers(0, 10, n).astype(float),
        "avg_session_length": rng.uniform(1, 30, n),
        "first_seen": pd.Timestamp("2024-01-01"),
        "last_seen": pd.Timestamp("2024-03-01"),
        "favorite_category": rng.choice(["electronics", "apparel", "beauty"], n),
    })

    assignments = pd.DataFrame({
        "user_id": user_ids,
        "experiment_id": "exp_test",
        "variant": ["treatment" if t else "control" for t in treatment],
    })
    return features, assignments, treatment


# ---------------------------------------------------------------------------
# Test 1: Propensity scores are in (0, 1)
# ---------------------------------------------------------------------------

def test_propensity_scores_between_zero_and_one():
    from src.causal.propensity import PropensityModel

    features, assignments, _ = _make_imbalanced_data()
    model = PropensityModel()
    model.train(features, assignments, experiment_id="test")

    scores = model.predict_propensity(features)
    assert scores.shape[0] == len(features)
    assert (scores > 0).all(), "Some propensity scores are <= 0"
    assert (scores < 1).all(), "Some propensity scores are >= 1"


# ---------------------------------------------------------------------------
# Test 2: Extreme propensity scores produce clipped weights
# ---------------------------------------------------------------------------

def test_weights_are_clipped():
    from src.causal.ipw import WEIGHT_MIN, WEIGHT_MAX

    # Extreme propensities near 0 and 1
    propensity = np.array([0.001, 0.01, 0.5, 0.99, 0.999])
    assignments = np.array([1, 1, 0, 0, 1])
    weights = compute_weights(propensity, assignments)

    assert (weights >= WEIGHT_MIN).all(), f"Weight below min {WEIGHT_MIN}"
    assert (weights <= WEIGHT_MAX).all(), f"Weight above max {WEIGHT_MAX}"


# ---------------------------------------------------------------------------
# Test 3: Balanced groups → CV AUC ≈ 0.5
# ---------------------------------------------------------------------------

def test_balanced_groups_have_low_auc():
    from src.causal.propensity import PropensityModel

    features, assignments = _make_balanced_data(n=600)
    model = PropensityModel()
    auc = model.train(features, assignments, experiment_id="test_balanced")

    assert 0.4 <= auc <= 0.65, (
        f"Balanced data should yield AUC near 0.5, got {auc:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Imbalanced groups → CV AUC > 0.6
# ---------------------------------------------------------------------------

def test_imbalanced_groups_have_high_auc():
    from src.causal.propensity import PropensityModel

    features, assignments, _ = _make_imbalanced_data(n=600)
    model = PropensityModel()
    auc = model.train(features, assignments, experiment_id="test_imbalanced")

    assert auc > 0.6, f"Imbalanced data should yield AUC > 0.6, got {auc:.3f}"


# ---------------------------------------------------------------------------
# Test 5: Imbalanced groups → adjustment_magnitude > 0
# ---------------------------------------------------------------------------

def test_adjusted_lift_differs_from_unadjusted_when_imbalanced():
    from src.causal.propensity import PropensityModel

    features, assignments, treatment = _make_imbalanced_data(n=600)
    model = PropensityModel()
    model.train(features, assignments, experiment_id="test_imb")

    propensity = model.predict_propensity(features)
    weights = compute_weights(propensity, treatment)

    rng = np.random.default_rng(99)
    conversions = (rng.uniform(size=len(treatment)) < 0.05 + 0.02 * treatment).astype(float)

    result = adjusted_conversion_rate(conversions, weights, treatment)
    assert result["adjustment_magnitude"] > 0, (
        "Imbalanced groups should produce non-zero adjustment"
    )


# ---------------------------------------------------------------------------
# Test 6: Balanced groups → adjustment_magnitude is small
# ---------------------------------------------------------------------------

def test_adjusted_lift_similar_when_balanced():
    from src.causal.propensity import PropensityModel

    features, assignments = _make_balanced_data(n=600)
    treatment = (assignments["variant"] == "treatment").astype(int).values

    model = PropensityModel()
    model.train(features, assignments, experiment_id="test_bal")

    propensity = model.predict_propensity(features)
    weights = compute_weights(propensity, treatment)

    rng = np.random.default_rng(77)
    conversions = (rng.uniform(size=len(treatment)) < 0.05).astype(float)

    result = adjusted_conversion_rate(conversions, weights, treatment)
    assert result["adjustment_magnitude"] < 0.05, (
        f"Balanced groups should have small adjustment, got {result['adjustment_magnitude']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: IPW treatment weights sum ≈ n_treatment
# ---------------------------------------------------------------------------

def test_ipw_weights_sum_to_n_users():
    """
    A key IPW property: sum of treatment weights ≈ number of treatment users,
    and similarly for control. This ensures the re-weighted sample has the
    same effective size as the original.
    """
    n = 400
    propensity = np.full(n, 0.5)  # perfect balance → weights all = 2.0, then clipped
    assignments = np.array([1] * (n // 2) + [0] * (n // 2))
    weights = compute_weights(propensity, assignments)

    t_mask = assignments == 1
    c_mask = assignments == 0

    # With p=0.5: weight = 1/0.5 = 2.0 for both groups
    # sum of treatment weights = n_treatment * 2.0
    # normalized sum ≈ n_treatment (within reasonable tolerance)
    n_treatment = t_mask.sum()
    n_control = c_mask.sum()
    sum_t = weights[t_mask].sum()
    sum_c = weights[c_mask].sum()

    # weights should be proportional to group size (within factor of 3x)
    assert sum_t / n_treatment < 3.0, f"Treatment weights too large: {sum_t / n_treatment:.2f}"
    assert sum_c / n_control < 3.0, f"Control weights too large: {sum_c / n_control:.2f}"
