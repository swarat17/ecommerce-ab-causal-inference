"""
Deterministic user-to-variant assignment via hashing.

Given the same user_id + experiment_id, always returns the same variant.
This allows replaying historical data with consistent assignments.
"""
import hashlib
import os
from typing import Optional

import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentAssigner:

    def assign_variant(
        self,
        user_id: str,
        experiment_id: str,
        variants: list[str],
        traffic_pct: float = 1.0,
    ) -> Optional[str]:
        """
        Deterministically assign a user to a variant.

        Uses SHA-256 hash of "{user_id}:{experiment_id}" modulo 10000
        to produce a stable bucket in [0, 9999].

        - If bucket >= traffic_pct * 10000 → user is not in the experiment (None)
        - Otherwise → assigned to one of the variants by equal split

        Returns variant name or None.
        """
        digest = hashlib.sha256(f"{user_id}:{experiment_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 10000

        threshold = int(traffic_pct * 10000)
        if bucket >= threshold:
            return None

        n = len(variants)
        variant_idx = bucket % n
        return variants[variant_idx]

    def assign_all_users(
        self,
        users: list[str],
        experiment_id: str,
        variants: list[str],
        traffic_pct: float = 1.0,
        db_url: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Assign all users to variants and return a DataFrame.

        Drops users assigned to None (not in experiment).
        Optionally persists assignments to PostgreSQL user_assignments table.

        Returns DataFrame with columns: user_id, experiment_id, variant.
        """
        records = []
        for uid in users:
            v = self.assign_variant(uid, experiment_id, variants, traffic_pct)
            if v is not None:
                records.append({"user_id": uid, "experiment_id": experiment_id, "variant": v})

        df = pd.DataFrame(records)
        logger.info(
            f"Assigned {len(df)} / {len(users)} users to experiment '{experiment_id}'"
        )

        if db_url and len(df) > 0:
            self._persist_assignments(df, db_url)

        return df

    def _persist_assignments(self, assignments_df: pd.DataFrame, db_url: str) -> None:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url)
        rows = assignments_df.rename(columns={"variant": "variant_id"})[
            ["user_id", "experiment_id", "variant_id"]
        ]
        with engine.begin() as conn:
            for _, row in rows.iterrows():
                conn.execute(
                    text(
                        """
                        INSERT INTO user_assignments (user_id, experiment_id, variant_id)
                        VALUES (:user_id, :experiment_id, :variant_id)
                        ON CONFLICT (user_id, experiment_id) DO NOTHING
                        """
                    ),
                    row.to_dict(),
                )
        engine.dispose()
        logger.info(f"Persisted {len(rows)} assignments to PostgreSQL")

    def check_srm(self, assignments_df: pd.DataFrame) -> dict:
        """
        Sample Ratio Mismatch (SRM) check.

        Runs a chi-square goodness-of-fit test on observed variant counts
        against the expected equal split. SRM (p < 0.01) means randomization
        is broken — the experiment should be halted.

        Returns:
            {
                "srm_detected": bool,
                "p_value": float,
                "expected_ratio": float,
                "actual_counts": dict,
            }
        """
        counts = assignments_df["variant"].value_counts()
        n_variants = len(counts)
        total = counts.sum()
        expected = total / n_variants
        expected_ratio = 1.0 / n_variants

        # chi2 goodness-of-fit: observed vs. uniform expected
        observed = counts.values
        expected_counts = [expected] * n_variants
        _, p_value = chi2_contingency(
            [observed, expected_counts], correction=False
        )[:2]

        srm_detected = p_value < 0.01
        if srm_detected:
            logger.warning(
                f"SRM detected! p={p_value:.4f} — assignment is not balanced. "
                "Halt the experiment."
            )

        return {
            "srm_detected": srm_detected,
            "p_value": float(p_value),
            "expected_ratio": expected_ratio,
            "actual_counts": counts.to_dict(),
        }
