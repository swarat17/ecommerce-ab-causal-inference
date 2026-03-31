"""
Multiple testing correction methods.

When testing multiple metrics simultaneously (conversion AND revenue AND CTR),
the probability of at least one false positive grows. These corrections control
the False Discovery Rate (BH) or Family-Wise Error Rate (Bonferroni).
"""
from statsmodels.stats.multitest import multipletests

from src.utils.logger import get_logger

logger = get_logger(__name__)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg procedure for False Discovery Rate control.

    More powerful than Bonferroni — appropriate when testing correlated
    metrics within a single experiment (conversion, revenue, CTR).

    Returns list of booleans: True = hypothesis rejected (significant).
    """
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return list(reject)


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Bonferroni correction for Family-Wise Error Rate control.

    More conservative than BH — rejects fewer hypotheses.
    Appropriate when any false positive is costly.

    Returns list of booleans: True = hypothesis rejected (significant).
    """
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="bonferroni")
    return list(reject)


def apply_corrections(results: dict, method: str = "benjamini_hochberg") -> dict:
    """
    Apply multiple testing correction to a dict of metric results.

    Args:
        results: dict keyed by metric name, each value is a dict
                 containing at minimum a "p_value" key.
                 E.g. output of MetricComputer.compute_all_metrics() after
                 frequentist tests have been run.
        method: "benjamini_hochberg" or "bonferroni"

    Returns the same dict with a "significant_corrected" key added to
    each metric's result.
    """
    metrics = list(results.keys())
    p_values = [results[m]["p_value"] for m in metrics]

    if method == "benjamini_hochberg":
        corrected = benjamini_hochberg(p_values)
    elif method == "bonferroni":
        corrected = bonferroni(p_values)
    else:
        raise ValueError(f"Unknown correction method: {method!r}")

    updated = dict(results)
    for metric, significant in zip(metrics, corrected):
        updated[metric] = dict(results[metric])
        updated[metric]["significant_corrected"] = bool(significant)

    n_rejected = sum(corrected)
    logger.info(
        f"[{method}] {n_rejected}/{len(metrics)} metrics significant after correction"
    )
    return updated
