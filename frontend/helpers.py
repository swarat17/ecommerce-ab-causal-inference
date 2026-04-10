"""
Pure helper functions for the Streamlit dashboard.

Extracted so they can be unit-tested without importing Streamlit.
"""
from __future__ import annotations

import math
from typing import Optional


def format_lift(lift: float) -> str:
    """Format a lift value as a percentage string with sign."""
    pct = lift * 100
    if pct >= 0:
        return f"+{pct:.1f}%"
    return f"{pct:.1f}%"


def srm_banner(srm_detected: bool) -> dict:
    """
    Return a dict with 'color' and 'message' for the SRM banner.

    color='red'   → SRM detected, experiment may be invalid
    color='green' → No SRM, randomization looks correct
    """
    if srm_detected:
        return {
            "color": "red",
            "message": "⚠️ Sample Ratio Mismatch detected! Randomization may be broken. Do NOT act on these results.",
        }
    return {
        "color": "green",
        "message": "✅ No Sample Ratio Mismatch — randomization looks correct.",
    }


def days_to_significance(
    required_n_per_variant: int,
    current_n_per_variant: int,
    days_running: int,
) -> int:
    """
    Estimate days until experiment reaches the required sample size.

    Uses the current daily enrollment rate inferred from current_n and days_running.
    Returns 0 if already sufficient.
    """
    if current_n_per_variant <= 0 or days_running <= 0:
        return 0
    daily_rate = current_n_per_variant / days_running
    if daily_rate <= 0:
        return 0
    remaining = max(required_n_per_variant - current_n_per_variant, 0)
    return math.ceil(remaining / daily_rate)


def significance_label(significant_corrected: bool, correction_method: str = "BH") -> str:
    """Return a human-readable significance label after multiple testing correction."""
    if significant_corrected:
        return f"✅ Significant ({correction_method})"
    return "❌ Not significant"


def format_p_value(p: float) -> str:
    """Format a p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def format_ci(lower: Optional[float], upper: Optional[float]) -> str:
    """Format a confidence interval as a string."""
    if lower is None or upper is None:
        return "N/A"
    return f"[{lower:+.4f}, {upper:+.4f}]"
