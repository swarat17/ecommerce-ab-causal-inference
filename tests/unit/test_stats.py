"""
Unit tests for Phase 3: frequentist tests, Bayesian A/B, and corrections.
"""
import numpy as np
import pandas as pd
import pytest

from src.stats.frequentist import (
    two_proportion_z_test,
    two_sample_t_test,
    novelty_effect_test,
    required_sample_size,
)
from src.stats.corrections import benjamini_hochberg, bonferroni, apply_corrections


# ---------------------------------------------------------------------------
# Test 1: z-test detects large effect
# ---------------------------------------------------------------------------

def test_z_test_significant_large_effect():
    # 5% vs 8% with n=10,000 each — clearly significant
    result = two_proportion_z_test(
        n_control=10_000, conv_control=500,
        n_treatment=10_000, conv_treatment=800,
    )
    assert result["significant"] == True
    assert result["p_value"] < 0.05
    assert result["absolute_lift"] == pytest.approx(0.03, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2: z-test does NOT detect tiny effect with small n
# ---------------------------------------------------------------------------

def test_z_test_not_significant_small_effect():
    # 5% vs 5.1% with n=500 — underpowered
    result = two_proportion_z_test(
        n_control=500, conv_control=25,
        n_treatment=500, conv_treatment=26,
    )
    assert result["significant"] == False
    assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Test 3: t-test result contains cohens_d
# ---------------------------------------------------------------------------

def test_t_test_returns_cohens_d():
    rng = np.random.default_rng(42)
    control = rng.normal(loc=10.0, scale=3.0, size=200)
    treatment = rng.normal(loc=11.5, scale=3.0, size=200)
    result = two_sample_t_test(control, treatment)
    assert "cohens_d" in result
    assert isinstance(result["cohens_d"], float)


# ---------------------------------------------------------------------------
# Test 4: 95% CI covers the true effect
# ---------------------------------------------------------------------------

def test_ci_contains_true_effect():
    # True effect: treatment mean = 12, control mean = 10  → delta = 2
    rng = np.random.default_rng(0)
    control = rng.normal(loc=10.0, scale=2.0, size=1000)
    treatment = rng.normal(loc=12.0, scale=2.0, size=1000)
    result = two_sample_t_test(control, treatment)
    true_effect = 2.0
    assert result["ci_lower"] < true_effect < result["ci_upper"], (
        f"True effect {true_effect} not in CI [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
    )


# ---------------------------------------------------------------------------
# Test 5: Novelty effect detected on decaying conversion data
# ---------------------------------------------------------------------------

def test_novelty_effect_detected_on_decaying_data():
    # Construct daily rates that start high then decay
    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    early_rates = [0.15, 0.14, 0.15, 0.13, 0.14, 0.15, 0.14]  # 7 days high
    post_rates = [0.06] * 14  # decay after week 1
    df = pd.DataFrame({
        "date": dates,
        "variant": ["treatment"] * 21,
        "conversion_rate": early_rates + post_rates,
    })
    result = novelty_effect_test(df, experiment_id="test_exp", novelty_window_days=7)
    assert result["novelty_detected"] == True
    assert result["early_mean"] > result["post_early_mean"]


# ---------------------------------------------------------------------------
# Test 6: BH correction reduces rejections vs uncorrected
# ---------------------------------------------------------------------------

def test_bh_correction_reduces_rejections():
    # Mix of 3 truly significant and 7 noise p-values
    p_values = [0.001, 0.003, 0.01, 0.08, 0.12, 0.25, 0.40, 0.55, 0.70, 0.90]
    uncorrected = [p < 0.05 for p in p_values]
    corrected = benjamini_hochberg(p_values)
    assert sum(corrected) <= sum(uncorrected), (
        "BH should reject fewer (or equal) hypotheses than uncorrected"
    )
    # The 3 truly significant ones should still be detected
    assert corrected[0] == True
    assert corrected[1] == True
    assert corrected[2] == True


# ---------------------------------------------------------------------------
# Test 7: Bonferroni is more conservative than BH
# ---------------------------------------------------------------------------

def test_bonferroni_more_conservative_than_bh():
    p_values = [0.001, 0.01, 0.03, 0.04, 0.06, 0.10, 0.20, 0.40, 0.60, 0.80]
    bh_results = benjamini_hochberg(p_values)
    bonf_results = bonferroni(p_values)
    assert sum(bonf_results) <= sum(bh_results), (
        "Bonferroni should reject fewer (or equal) hypotheses than BH"
    )


# ---------------------------------------------------------------------------
# Test 8: Bayesian probabilities sum to ~1.0
# ---------------------------------------------------------------------------

def test_bayesian_probability_sums_correctly():
    from src.stats.bayesian import BayesianABTest

    model = BayesianABTest()
    model.fit_conversion(
        n_control=1000, conv_control=50,
        n_treatment=1000, conv_treatment=65,
        tune=300, draws=500, chains=2,
    )
    p_t_better = model.probability_treatment_better()
    p_c_better = 1.0 - p_t_better  # complement (ties are measure-zero)
    assert abs(p_t_better + p_c_better - 1.0) < 1e-9
    assert 0.0 <= p_t_better <= 1.0


# ---------------------------------------------------------------------------
# Test 9: Larger MDE → smaller required sample size
# ---------------------------------------------------------------------------

def test_required_sample_size_increases_with_smaller_mde():
    n_large_mde = required_sample_size(baseline_rate=0.05, mde=0.02)
    n_small_mde = required_sample_size(baseline_rate=0.05, mde=0.005)
    assert n_small_mde > n_large_mde, (
        f"Smaller MDE should need more users: got {n_small_mde} vs {n_large_mde}"
    )
