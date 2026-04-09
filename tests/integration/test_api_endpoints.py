"""
Integration tests for Phase 5: live FastAPI endpoints.

Requires:
- PostgreSQL running (docker-compose up -d)
- scripts/setup_db.py executed
- Experiment 'exp_A' seeded in the DB

Run with: pytest tests/integration/ -v -m integration
"""
import os
import pytest
import pandas as pd
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ab_testing")


def _seed_experiment(engine):
    """Insert minimal experiment data for integration tests."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO experiments (experiment_id, name, description, start_date, status)
            VALUES ('exp_A', 'New Product Page Layout', 'Test new layout', NOW(), 'running')
            ON CONFLICT (experiment_id) DO NOTHING
        """))
        conn.execute(text("""
            INSERT INTO variants (variant_id, experiment_id, name, is_control)
            VALUES
                ('exp_A_control', 'exp_A', 'control', TRUE),
                ('exp_A_treatment', 'exp_A', 'treatment', FALSE)
            ON CONFLICT (variant_id) DO NOTHING
        """))
        # Seed 200 user assignments (100 per variant)
        for i in range(100):
            conn.execute(text("""
                INSERT INTO user_assignments (user_id, experiment_id, variant_id)
                VALUES (:uid, 'exp_A', 'exp_A_control')
                ON CONFLICT (user_id, experiment_id) DO NOTHING
            """), {"uid": f"ctrl_user_{i}"})
        for i in range(100):
            conn.execute(text("""
                INSERT INTO user_assignments (user_id, experiment_id, variant_id)
                VALUES (:uid, 'exp_A', 'exp_A_treatment')
                ON CONFLICT (user_id, experiment_id) DO NOTHING
            """), {"uid": f"trt_user_{i}"})


@pytest.fixture(scope="module")
def client():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _seed_experiment(engine)
        engine.dispose()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")

    from src.api.main import app
    return TestClient(app)


@pytest.mark.integration
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.integration
def test_analyze_returns_full_result(client):
    response = client.post(
        "/experiments/exp_A/analyze",
        json={
            "experiment_id": "exp_A",
            "metrics": ["conversion_rate", "revenue_per_user"],
            "correction_method": "benjamini_hochberg",
            "use_propensity": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "exp_A"
    assert data["frequentist"] is not None
    assert len(data["frequentist"]) > 0
    assert data["srm_check"] is not None


@pytest.mark.integration
def test_results_persisted_after_analyze(client):
    # First run analyze to populate results
    client.post(
        "/experiments/exp_A/analyze",
        json={"experiment_id": "exp_A", "correction_method": "benjamini_hochberg", "use_propensity": False},
    )
    # Then fetch cached results
    response = client.get("/experiments/exp_A/results")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "exp_A"
