"""
Seed Experiments A and B into PostgreSQL and assign all users to variants.

Steps:
  1. Load unique user list from data/processed/user_features.parquet
  2. Insert experiment + variant rows (idempotent — ON CONFLICT DO NOTHING)
  3. Hash-assign all users to control/treatment for each experiment
  4. Bulk-insert assignments via a staging table (avoids row-by-row iterrows)
  5. Run SRM check and print a summary

Run: python scripts/run_experiment.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine, text

from src.experiments.assignment import ExperimentAssigner
from src.utils.logger import get_logger

logger = get_logger(__name__)

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ab_testing")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "experiment_id": "exp_A",
        "name": "New Product Page Layout",
        "description": (
            "Test whether a redesigned product page increases purchase "
            "conversion rate. Control: old layout. Treatment: new layout."
        ),
        "variants": [
            {"variant_id": "exp_A_control",   "name": "control",   "is_control": True},
            {"variant_id": "exp_A_treatment",  "name": "treatment", "is_control": False},
        ],
    },
    {
        "experiment_id": "exp_B",
        "name": "Discount Banner",
        "description": (
            "Test whether showing a 10% discount banner increases revenue "
            "per user. Control: no banner. Treatment: discount banner shown."
        ),
        "variants": [
            {"variant_id": "exp_B_control",   "name": "control",   "is_control": True},
            {"variant_id": "exp_B_treatment",  "name": "treatment", "is_control": False},
        ],
    },
]


# ---------------------------------------------------------------------------
# Step 1 — Seed experiments + variants
# ---------------------------------------------------------------------------

def seed_experiments(engine) -> None:
    with engine.begin() as conn:
        for exp in EXPERIMENTS:
            conn.execute(text("""
                INSERT INTO experiments
                    (experiment_id, name, description, start_date, status)
                VALUES
                    (:eid, :name, :desc, NOW(), 'running')
                ON CONFLICT (experiment_id) DO NOTHING
            """), {"eid": exp["experiment_id"], "name": exp["name"], "desc": exp["description"]})

            for v in exp["variants"]:
                conn.execute(text("""
                    INSERT INTO variants (variant_id, experiment_id, name, is_control)
                    VALUES (:vid, :eid, :name, :is_control)
                    ON CONFLICT (variant_id) DO NOTHING
                """), {
                    "vid": v["variant_id"],
                    "eid": exp["experiment_id"],
                    "name": v["name"],
                    "is_control": v["is_control"],
                })

            logger.info(
                f"Seeded experiment '{exp['experiment_id']}' "
                f"with {len(exp['variants'])} variants"
            )


# ---------------------------------------------------------------------------
# Step 2 — Assign users (in-memory only, then bulk insert)
# ---------------------------------------------------------------------------

def assign_and_persist(users: list[str], engine) -> None:
    assigner = ExperimentAssigner()

    for exp in EXPERIMENTS:
        exp_id = exp["experiment_id"]
        variant_ids = [v["variant_id"] for v in exp["variants"]]

        logger.info(f"[{exp_id}] Assigning {len(users):,} users ...")

        # Compute assignments in memory — do NOT pass db_url (avoids slow iterrows)
        assignments_df = assigner.assign_all_users(
            users=users,
            experiment_id=exp_id,
            variants=variant_ids,
            traffic_pct=1.0,
            db_url=None,
        )

        # SRM check before writing
        srm = assigner.check_srm(assignments_df)
        counts_str = ", ".join(f"{k}: {v:,}" for k, v in srm["actual_counts"].items())
        srm_flag = "⚠ SRM DETECTED" if srm["srm_detected"] else "✓ no SRM"
        logger.info(
            f"[{exp_id}] SRM check — p={srm['p_value']:.4f} {srm_flag} | {counts_str}"
        )

        # Bulk insert via staging table
        # assignments_df has columns: user_id, experiment_id, variant (= variant_id here)
        bulk_df = assignments_df.rename(columns={"variant": "variant_id"})[
            ["user_id", "experiment_id", "variant_id"]
        ]

        logger.info(f"[{exp_id}] Bulk-inserting {len(bulk_df):,} assignments ...")

        staging = f"_staging_assignments_{exp_id}"

        # Write to a temp staging table (fast bulk load, no constraints)
        bulk_df.to_sql(staging, engine, if_exists="replace", index=False, method="multi", chunksize=10_000)

        # Move from staging → real table with conflict handling in one SQL statement
        with engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO user_assignments (user_id, experiment_id, variant_id)
                SELECT user_id, experiment_id, variant_id
                FROM "{staging}"
                ON CONFLICT (user_id, experiment_id) DO NOTHING
            """))
            conn.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))

        logger.info(f"[{exp_id}] Done — {len(bulk_df):,} assignments persisted.")
        del assignments_df, bulk_df  # free before next experiment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("run_experiment.py — seeding experiments A & B")
    logger.info("=" * 60)

    # Load user list (only user_id column — no need to load all 12 features)
    features_path = f"{PROCESSED_DATA_PATH}/user_features.parquet"
    logger.info(f"Loading users from {features_path} ...")
    users_df = pd.read_parquet(features_path, columns=["user_id"])
    users = users_df["user_id"].astype(str).tolist()
    logger.info(f"Found {len(users):,} unique users")

    engine = create_engine(DB_URL)

    try:
        # Verify DB connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection OK")
    except Exception as e:
        logger.error(f"Cannot connect to PostgreSQL: {e}")
        logger.error("Make sure docker-compose up -d postgres && python scripts/setup_db.py")
        sys.exit(1)

    seed_experiments(engine)
    assign_and_persist(users, engine)

    # Print final summary from DB
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                e.experiment_id,
                e.name,
                v.name        AS variant,
                COUNT(ua.user_id) AS n_users
            FROM experiments e
            JOIN variants v        ON e.experiment_id = v.experiment_id
            LEFT JOIN user_assignments ua
                ON ua.experiment_id = e.experiment_id
               AND ua.variant_id    = v.variant_id
            GROUP BY e.experiment_id, e.name, v.name
            ORDER BY e.experiment_id, v.name
        """)).fetchall()

    logger.info("\n--- Assignment Summary ---")
    for r in rows:
        logger.info(f"  {r.experiment_id} / {r.variant:<12} : {r.n_users:>10,} users")

    engine.dispose()
    logger.info("=" * 60)
    logger.info("run_experiment.py complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
