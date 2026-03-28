"""
Creates the PostgreSQL schema for the A/B testing platform.
Run: python scripts/setup_db.py
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    start_date      TIMESTAMP,
    end_date        TIMESTAMP,
    status          VARCHAR(32) DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS variants (
    variant_id      VARCHAR(64) PRIMARY KEY,
    experiment_id   VARCHAR(64) NOT NULL REFERENCES experiments(experiment_id),
    name            VARCHAR(128) NOT NULL,
    is_control      BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_assignments (
    user_id         VARCHAR(64) NOT NULL,
    experiment_id   VARCHAR(64) NOT NULL REFERENCES experiments(experiment_id),
    variant_id      VARCHAR(64) NOT NULL REFERENCES variants(variant_id),
    assigned_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_results (
    id              SERIAL PRIMARY KEY,
    experiment_id   VARCHAR(64) NOT NULL REFERENCES experiments(experiment_id),
    variant_id      VARCHAR(64) NOT NULL REFERENCES variants(variant_id),
    metric_name     VARCHAR(128) NOT NULL,
    value           DOUBLE PRECISION,
    ci_lower        DOUBLE PRECISION,
    ci_upper        DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    computed_at     TIMESTAMP DEFAULT NOW()
);
"""


def setup_db(db_url: str | None = None) -> None:
    url = db_url or os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ab_testing")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("Schema created successfully.")
    engine.dispose()


if __name__ == "__main__":
    setup_db()
