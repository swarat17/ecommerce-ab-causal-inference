"""
Inverse Probability Weighting (IPW) for causal effect estimation.

Weights re-balance the treatment and control groups on observed covariates,
correcting for any pre-experiment imbalance captured by the propensity model.

Weight formulas:
    treatment user:  w = 1 / p(x)
    control user:    w = 1 / (1 - p(x))

Weights are clipped to [0.1, 10.0] to prevent extreme values from dominating.
"""
import numpy as np
from statsmodels.stats.weightstats import ttest_ind as weighted_ttest_ind

from src.utils.logger import get_logger

logger = get_logger(__name__)

WEIGHT_MIN = 0.1
WEIGHT_MAX = 10.0


def compute_weights(
    propensity_scores: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """
    Compute IPW weights for each user.

    Args:
        propensity_scores: P(treatment | covariates), shape (n,)
        assignments: binary array, 1 = treatment, 0 = control, shape (n,)

    Returns clipped weight array of shape (n,).
    """
    propensity_scores = np.asarray(propensity_scores, dtype=float)
    assignments = np.asarray(assignments, dtype=int)

    weights = np.where(
        assignments == 1,
        1.0 / propensity_scores,
        1.0 / (1.0 - propensity_scores),
    )
    weights = np.clip(weights, WEIGHT_MIN, WEIGHT_MAX)
    return weights


def adjusted_conversion_rate(
    conversions: np.ndarray,
    weights: np.ndarray,
    assignments: np.ndarray,
) -> dict:
    """
    IPW-adjusted conversion rate for treatment and control.

    Args:
        conversions: binary outcome (1 = converted), shape (n,)
        weights: IPW weights from compute_weights(), shape (n,)
        assignments: 1 = treatment, 0 = control, shape (n,)

    Returns:
        {
            "control": float,           weighted conversion rate for control
            "treatment": float,         weighted conversion rate for treatment
            "adjusted_lift": float,     treatment - control (weighted)
            "unadjusted_lift": float,   treatment - control (raw)
            "adjustment_magnitude": float,  |adjusted - unadjusted|
        }
    """
    conversions = np.asarray(conversions, dtype=float)
    assignments = np.asarray(assignments, dtype=int)

    t_mask = assignments == 1
    c_mask = assignments == 0

    raw_t = float(conversions[t_mask].mean()) if t_mask.sum() > 0 else 0.0
    raw_c = float(conversions[c_mask].mean()) if c_mask.sum() > 0 else 0.0
    unadjusted_lift = raw_t - raw_c

    adj_t = float(np.average(conversions[t_mask], weights=weights[t_mask])) if t_mask.sum() > 0 else 0.0
    adj_c = float(np.average(conversions[c_mask], weights=weights[c_mask])) if c_mask.sum() > 0 else 0.0
    adjusted_lift = adj_t - adj_c

    adjustment_magnitude = abs(adjusted_lift - unadjusted_lift)

    logger.info(
        f"IPW adjustment: unadjusted lift = {unadjusted_lift:+.4f}, "
        f"adjusted lift = {adjusted_lift:+.4f}, "
        f"magnitude = {adjustment_magnitude:.4f}"
    )

    return {
        "control": adj_c,
        "treatment": adj_t,
        "adjusted_lift": adjusted_lift,
        "unadjusted_lift": unadjusted_lift,
        "adjustment_magnitude": adjustment_magnitude,
    }


def adjusted_t_test(
    metric_values: np.ndarray,
    weights: np.ndarray,
    assignments: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Weighted t-test for difference in means using statsmodels.

    Mirrors the interface of frequentist.two_sample_t_test for consistency.

    Args:
        metric_values: continuous outcome (e.g., revenue per user), shape (n,)
        weights: IPW weights from compute_weights(), shape (n,)
        assignments: 1 = treatment, 0 = control, shape (n,)
        alpha: significance level

    Returns dict with: t_stat, p_value, cohens_d, ci_lower, ci_upper, significant
    """
    from statsmodels.stats.weightstats import DescrStatsW

    metric_values = np.asarray(metric_values, dtype=float)
    assignments = np.asarray(assignments, dtype=int)

    t_mask = assignments == 1
    c_mask = assignments == 0

    d1 = DescrStatsW(metric_values[t_mask], weights=weights[t_mask], ddof=1)
    d2 = DescrStatsW(metric_values[c_mask], weights=weights[c_mask], ddof=1)

    t_stat, p_value, _ = d1.ttest_mean(d2.mean)

    # Cohen's d using weighted means and pooled std
    pooled_std = np.sqrt((d1.std ** 2 + d2.std ** 2) / 2)
    cohens_d = (d1.mean - d2.mean) / pooled_std if pooled_std > 0 else 0.0

    # 95% CI on weighted mean difference
    diff = d1.mean - d2.mean
    se = np.sqrt(d1.std ** 2 / d1.nobs + d2.std ** 2 / d2.nobs)
    from scipy import stats
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = diff - z_crit * se
    ci_upper = diff + z_crit * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(p_value < alpha),
    }
