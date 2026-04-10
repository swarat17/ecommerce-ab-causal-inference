"""
Unit tests for Phase 6: Streamlit dashboard helper functions.
No Streamlit, no API calls — pure function tests.
"""
import pytest

from frontend.helpers import (
    days_to_significance,
    format_lift,
    significance_label,
    srm_banner,
)


# ---------------------------------------------------------------------------
# Test 1: format_lift positive
# ---------------------------------------------------------------------------

def test_format_lift_positive():
    assert format_lift(0.023) == "+2.3%"


# ---------------------------------------------------------------------------
# Test 2: format_lift negative
# ---------------------------------------------------------------------------

def test_format_lift_negative():
    assert format_lift(-0.015) == "-1.5%"


# ---------------------------------------------------------------------------
# Test 3: srm_banner color — detected
# ---------------------------------------------------------------------------

def test_srm_banner_color_detected():
    result = srm_banner(True)
    assert result["color"] == "red"


# ---------------------------------------------------------------------------
# Test 4: srm_banner color — not detected
# ---------------------------------------------------------------------------

def test_srm_banner_color_not_detected():
    result = srm_banner(False)
    assert result["color"] == "green"


# ---------------------------------------------------------------------------
# Test 5: days_to_significance returns plausible int
# ---------------------------------------------------------------------------

def test_days_to_significance():
    # 500 users/variant in 10 days → 50 users/day
    # Need 1000 users/variant → 500 more → 10 more days
    result = days_to_significance(
        required_n_per_variant=1000,
        current_n_per_variant=500,
        days_running=10,
    )
    assert isinstance(result, int)
    assert result == 10


def test_days_to_significance_already_sufficient():
    result = days_to_significance(
        required_n_per_variant=500,
        current_n_per_variant=1000,
        days_running=10,
    )
    assert result == 0


# ---------------------------------------------------------------------------
# Test 6: significance_label — corrected significant
# ---------------------------------------------------------------------------

def test_significance_label_corrected():
    label = significance_label(significant_corrected=True, correction_method="BH")
    assert "Significant" in label
    assert "BH" in label


# ---------------------------------------------------------------------------
# Test 7: significance_label — not significant
# ---------------------------------------------------------------------------

def test_significance_label_not_significant():
    label = significance_label(significant_corrected=False)
    assert "Not significant" in label
