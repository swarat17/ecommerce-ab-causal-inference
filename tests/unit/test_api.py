"""
Unit tests for Phase 5: API schema validation.
No database or HTTP calls — pure Pydantic validation tests.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ExperimentRequest,
    ExperimentResult,
    PowerAnalysisRequest,
    SRMResult,
    BayesianResult,
    PropensityResult,
)
from datetime import datetime


# ---------------------------------------------------------------------------
# Test 1: ExperimentRequest requires experiment_id
# ---------------------------------------------------------------------------

def test_experiment_request_requires_experiment_id():
    with pytest.raises(ValidationError) as exc_info:
        ExperimentRequest()  # missing experiment_id
    errors = exc_info.value.errors()
    fields = [e["loc"][0] for e in errors]
    assert "experiment_id" in fields


# ---------------------------------------------------------------------------
# Test 2: ExperimentResult schema has all required sections
# ---------------------------------------------------------------------------

def test_result_schema_has_all_sections():
    result = ExperimentResult(
        experiment_id="exp_A",
        name="Test Experiment",
        status="running",
        computed_at=datetime.utcnow(),
        srm_check=SRMResult(
            srm_detected=False,
            p_value=0.95,
            expected_ratio=0.5,
            actual_counts={"control": 500, "treatment": 500},
        ),
        frequentist=[],
        bayesian=BayesianResult(
            probability_treatment_better=0.75,
            expected_loss=0.002,
            credible_interval_lower=-0.01,
            credible_interval_upper=0.03,
        ),
        propensity=PropensityResult(
            cv_auc=0.52,
            imbalance_detected=False,
            adjustment_magnitude=0.001,
            unadjusted_lift=0.02,
            adjusted_lift=0.019,
        ),
    )
    assert result.srm_check is not None
    assert result.frequentist is not None
    assert result.bayesian is not None
    assert result.propensity is not None


# ---------------------------------------------------------------------------
# Test 3: PowerAnalysisRequest validates mde is in (0, 1)
# ---------------------------------------------------------------------------

def test_power_analysis_params_validated():
    # mde = 0 should fail
    with pytest.raises(ValidationError):
        PowerAnalysisRequest(mde=0.0)

    # mde = 1 should fail
    with pytest.raises(ValidationError):
        PowerAnalysisRequest(mde=1.0)

    # mde = 1.5 should fail
    with pytest.raises(ValidationError):
        PowerAnalysisRequest(mde=1.5)

    # valid mde should pass
    req = PowerAnalysisRequest(mde=0.01)
    assert req.mde == 0.01
