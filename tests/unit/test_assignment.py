"""
Unit tests for Phase 2: experiment assignment and SRM detection.
"""
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.experiments.assignment import ExperimentAssigner


@pytest.fixture
def assigner():
    return ExperimentAssigner()


VARIANTS = ["control", "treatment"]
EXP_ID = "exp_A"

# ---------------------------------------------------------------------------
# Test 1: Assignment is deterministic
# ---------------------------------------------------------------------------

def test_assignment_is_deterministic(assigner):
    uid = "user_42"
    v1 = assigner.assign_variant(uid, EXP_ID, VARIANTS)
    v2 = assigner.assign_variant(uid, EXP_ID, VARIANTS)
    assert v1 == v2, "Same user+experiment must always return the same variant"


# ---------------------------------------------------------------------------
# Test 2: traffic_pct excludes ~50% of users
# ---------------------------------------------------------------------------

def test_traffic_pct_excludes_users(assigner):
    users = [str(i) for i in range(5000)]
    assignments = [assigner.assign_variant(u, EXP_ID, VARIANTS, traffic_pct=0.5) for u in users]
    none_count = sum(1 for a in assignments if a is None)
    # Expect roughly 50% excluded; allow ±5% tolerance
    assert 0.45 <= none_count / len(users) <= 0.55, (
        f"Expected ~50% excluded, got {none_count / len(users):.2%}"
    )


# ---------------------------------------------------------------------------
# Test 3: Variant split is approximately equal (chi-square passes)
# ---------------------------------------------------------------------------

def test_variant_split_is_approximately_equal(assigner):
    users = [str(i) for i in range(10_000)]
    df = assigner.assign_all_users(users, EXP_ID, VARIANTS)
    counts = df["variant"].value_counts()

    for variant in VARIANTS:
        pct = counts.get(variant, 0) / len(df)
        assert 0.45 <= pct <= 0.55, f"Variant '{variant}' got {pct:.2%} — not balanced"


# ---------------------------------------------------------------------------
# Test 4: SRM detected on a bad (skewed) split
# ---------------------------------------------------------------------------

def test_srm_detected_on_bad_split(assigner):
    # Manufacture a 90/10 split
    bad_df = pd.DataFrame({
        "user_id": [str(i) for i in range(1000)],
        "experiment_id": [EXP_ID] * 1000,
        "variant": ["control"] * 900 + ["treatment"] * 100,
    })
    result = assigner.check_srm(bad_df)
    assert result["srm_detected"] == True, "SRM should be detected on a 90/10 split"
    assert result["p_value"] < 0.01


# ---------------------------------------------------------------------------
# Test 5: SRM not detected on a good (equal) split
# ---------------------------------------------------------------------------

def test_srm_not_detected_on_good_split(assigner):
    users = [str(i) for i in range(10_000)]
    df = assigner.assign_all_users(users, EXP_ID, VARIANTS)
    result = assigner.check_srm(df)
    assert result["srm_detected"] == False, (
        f"SRM should NOT be detected on hash-balanced data (p={result['p_value']:.4f})"
    )
