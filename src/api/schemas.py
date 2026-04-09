"""
Pydantic models for the A/B Testing Platform API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ExperimentRequest(BaseModel):
    experiment_id: str
    metrics: list[str] = Field(
        default=["conversion_rate", "revenue_per_user", "add_to_cart_rate"]
    )
    correction_method: str = Field(
        default="benjamini_hochberg",
        pattern="^(benjamini_hochberg|bonferroni)$",
    )
    use_propensity: bool = False


class PowerAnalysisRequest(BaseModel):
    metric: str = "conversion_rate"
    mde: float = Field(..., gt=0, lt=1, description="Minimum detectable effect (absolute)")

    @field_validator("mde")
    @classmethod
    def mde_must_be_positive(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError("mde must be in (0, 1)")
        return v


# ---------------------------------------------------------------------------
# Nested result models
# ---------------------------------------------------------------------------

class VariantMetrics(BaseModel):
    variant: str
    n_users: int
    conversion_rate: Optional[float] = None
    mean_revenue: Optional[float] = None
    add_to_cart_rate: Optional[float] = None


class FrequentistTestResult(BaseModel):
    metric: str
    p_value: float
    significant: bool
    significant_corrected: bool
    absolute_lift: Optional[float] = None
    relative_lift: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class BayesianResult(BaseModel):
    probability_treatment_better: float
    expected_loss: float
    credible_interval_lower: float
    credible_interval_upper: float


class PropensityResult(BaseModel):
    cv_auc: float
    imbalance_detected: bool
    adjustment_magnitude: float
    unadjusted_lift: float
    adjusted_lift: float


class SRMResult(BaseModel):
    srm_detected: bool
    p_value: float
    expected_ratio: float
    actual_counts: dict[str, int]


class NoveltyResult(BaseModel):
    novelty_detected: bool
    p_value: float
    early_mean: float
    post_early_mean: Optional[float] = None


class SampleSizeAnalysis(BaseModel):
    baseline_rate: float
    current_n_per_variant: int
    minimum_detectable_effect: float
    required_n_for_80pct_power: int


# ---------------------------------------------------------------------------
# Top-level result and summary models
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    experiment_id: str
    name: str
    status: str
    computed_at: datetime

    srm_check: Optional[SRMResult] = None
    variants: list[VariantMetrics] = []
    frequentist: list[FrequentistTestResult] = []
    bayesian: Optional[BayesianResult] = None
    propensity: Optional[PropensityResult] = None
    novelty_check: Optional[NoveltyResult] = None
    sample_size_analysis: Optional[SampleSizeAnalysis] = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    status: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    n_users: Optional[int] = None
