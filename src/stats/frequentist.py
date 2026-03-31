"""
Frequentist statistical tests for A/B experiments.

Includes:
- Two-proportion z-test (conversion rate comparison)
- Two-sample Welch's t-test (continuous metrics like revenue)
- Minimum detectable effect (MDE) calculation
- Required sample size calculation
- Novelty effect detection
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
from statsmodels.stats.power import NormalIndPower

from src.utils.logger import get_logger

logger = get_logger(__name__)


def two_proportion_z_test(
    n_control: int,
    conv_control: int,
    n_treatment: int,
    conv_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """
    Test the difference between two conversion rates.

    Args:
        n_control: users in control
        conv_control: converters in control
        n_treatment: users in treatment
        conv_treatment: converters in treatment
        alpha: significance level

    Returns dict with:
        z_stat, p_value, relative_lift, absolute_lift,
        ci_lower, ci_upper (95% CI on absolute lift), significant
    """
    rate_c = conv_control / n_control
    rate_t = conv_treatment / n_treatment

    count = np.array([conv_treatment, conv_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    absolute_lift = rate_t - rate_c
    relative_lift = absolute_lift / rate_c if rate_c > 0 else 0.0

    # 95% CI on absolute lift using normal approximation
    se = np.sqrt(rate_c * (1 - rate_c) / n_control + rate_t * (1 - rate_t) / n_treatment)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = absolute_lift - z_crit * se
    ci_upper = absolute_lift + z_crit * se

    return {
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "relative_lift": float(relative_lift),
        "absolute_lift": float(absolute_lift),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(p_value < alpha),
    }


def two_sample_t_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Welch's t-test for difference in means (e.g., revenue per user).

    Args:
        control_values: array of metric values for control users
        treatment_values: array of metric values for treatment users
        alpha: significance level

    Returns dict with:
        t_stat, p_value, cohens_d, ci_lower, ci_upper, significant
    """
    control_values = np.asarray(control_values, dtype=float)
    treatment_values = np.asarray(treatment_values, dtype=float)

    t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(
        (np.std(control_values, ddof=1) ** 2 + np.std(treatment_values, ddof=1) ** 2) / 2
    )
    cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0.0

    # 95% CI on mean difference via t-distribution
    diff = np.mean(treatment_values) - np.mean(control_values)
    se = np.sqrt(
        np.var(control_values, ddof=1) / len(control_values)
        + np.var(treatment_values, ddof=1) / len(treatment_values)
    )
    df = len(control_values) + len(treatment_values) - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": bool(p_value < alpha),
    }


def minimum_detectable_effect(
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Smallest absolute effect size detectable given current sample sizes.

    Uses NormalIndPower to solve for the effect size that achieves the
    desired power at the given alpha with the observed sample sizes.
    """
    ratio = n_treatment / n_control
    analysis = NormalIndPower()
    # effect size in Cohen's h (arcsine transformation)
    effect_size_h = analysis.solve_power(
        nobs1=n_control,
        ratio=ratio,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    # Convert Cohen's h back to absolute rate difference
    p1 = baseline_rate
    # h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))
    # Solve for p2
    arcsin_p1 = np.arcsin(np.sqrt(p1))
    arcsin_p2 = arcsin_p1 + effect_size_h / 2
    p2 = np.sin(arcsin_p2) ** 2
    return float(abs(p2 - p1))


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Number of users per variant needed to detect a given MDE.

    Args:
        baseline_rate: control conversion rate (e.g., 0.05 for 5%)
        mde: minimum detectable effect (absolute, e.g., 0.01 for +1pp)
        alpha: significance level
        power: desired power (1 - beta)

    Returns required n per variant (integer).
    """
    effect_size = proportion_effectsize(baseline_rate, baseline_rate + mde)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def novelty_effect_test(
    daily_metrics_df: pd.DataFrame,
    experiment_id: str,
    novelty_window_days: int = 7,
) -> dict:
    """
    Detect novelty effect by comparing early vs. post-early conversion.

    Splits the experiment into:
    - Early period: first `novelty_window_days` days
    - Post-early period: remainder

    Runs Welch's t-test on daily treatment conversion rates between periods.
    novelty_detected=True if early conversion is significantly higher (one-sided).

    Args:
        daily_metrics_df: DataFrame with columns
            [date, variant, conversion_rate] sorted by date
        experiment_id: identifier (used for logging)
        novelty_window_days: length of early period to check

    Returns dict with:
        novelty_detected, p_value, early_mean, post_early_mean
    """
    treatment = daily_metrics_df[daily_metrics_df["variant"] != "control"].copy()
    treatment = treatment.sort_values("date").reset_index(drop=True)

    if len(treatment) <= novelty_window_days:
        logger.warning(
            f"[{experiment_id}] Not enough days ({len(treatment)}) for novelty check"
        )
        return {
            "novelty_detected": False,
            "p_value": 1.0,
            "early_mean": float(treatment["conversion_rate"].mean()),
            "post_early_mean": float("nan"),
        }

    early = treatment.iloc[:novelty_window_days]["conversion_rate"].values
    post_early = treatment.iloc[novelty_window_days:]["conversion_rate"].values

    # One-sided test: early > post-early
    t_stat, p_two_sided = stats.ttest_ind(early, post_early, equal_var=False)
    # Convert to one-sided p-value (early > post-early means t_stat > 0)
    p_value = p_two_sided / 2 if t_stat > 0 else 1.0

    novelty_detected = p_value < 0.05
    if novelty_detected:
        logger.warning(
            f"[{experiment_id}] Novelty effect detected! "
            f"Early mean={np.mean(early):.4f}, post-early mean={np.mean(post_early):.4f}"
        )

    return {
        "novelty_detected": novelty_detected,
        "p_value": float(p_value),
        "early_mean": float(np.mean(early)),
        "post_early_mean": float(np.mean(post_early)),
    }
