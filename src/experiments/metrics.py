"""
Metric computation for A/B experiments.

Joins events with user assignments and computes per-variant metrics:
- Conversion rate (purchase conversion)
- Revenue per user
- Add-to-cart rate
"""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricComputer:

    def compute_conversion_rate(
        self,
        events_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
        experiment_id: str,
    ) -> pd.DataFrame:
        """
        Compute purchase conversion rate per variant.

        A user "converts" if they have at least one purchase event.

        Returns DataFrame with columns:
            variant, n_users, n_converters, conversion_rate
        """
        exp_assignments = assignments_df[
            assignments_df["experiment_id"] == experiment_id
        ]

        purchases = (
            events_df[events_df["event_type"] == "purchase"]
            .groupby("user_id")
            .size()
            .reset_index(name="purchase_count")
        )

        merged = exp_assignments.merge(purchases, on="user_id", how="left")
        merged["converted"] = merged["purchase_count"].notna() & (merged["purchase_count"] > 0)

        result = (
            merged.groupby("variant")
            .agg(
                n_users=("user_id", "count"),
                n_converters=("converted", "sum"),
            )
            .reset_index()
        )
        result["conversion_rate"] = result["n_converters"] / result["n_users"]
        return result

    def compute_revenue_per_user(
        self,
        events_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
        experiment_id: str,
    ) -> pd.DataFrame:
        """
        Compute mean revenue per user per variant.

        Revenue = sum of purchase event prices per user (0 if no purchases).

        Returns DataFrame with columns:
            variant, n_users, mean_revenue, std_revenue
        """
        exp_assignments = assignments_df[
            assignments_df["experiment_id"] == experiment_id
        ]

        user_revenue = (
            events_df[events_df["event_type"] == "purchase"]
            .groupby("user_id")["price"]
            .sum()
            .reset_index(name="revenue")
        )

        merged = exp_assignments.merge(user_revenue, on="user_id", how="left")
        merged["revenue"] = merged["revenue"].fillna(0.0)

        result = (
            merged.groupby("variant")
            .agg(
                n_users=("user_id", "count"),
                mean_revenue=("revenue", "mean"),
                std_revenue=("revenue", "std"),
            )
            .reset_index()
        )
        result["std_revenue"] = result["std_revenue"].fillna(0.0)
        return result

    def compute_add_to_cart_rate(
        self,
        events_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
        experiment_id: str,
    ) -> pd.DataFrame:
        """
        Compute add-to-cart rate per variant.

        A user "carted" if they have at least one cart event.

        Returns DataFrame with columns:
            variant, n_users, n_carted, add_to_cart_rate
        """
        exp_assignments = assignments_df[
            assignments_df["experiment_id"] == experiment_id
        ]

        carted = (
            events_df[events_df["event_type"] == "cart"]
            .groupby("user_id")
            .size()
            .reset_index(name="cart_count")
        )

        merged = exp_assignments.merge(carted, on="user_id", how="left")
        merged["carted"] = merged["cart_count"].notna() & (merged["cart_count"] > 0)

        result = (
            merged.groupby("variant")
            .agg(
                n_users=("user_id", "count"),
                n_carted=("carted", "sum"),
            )
            .reset_index()
        )
        result["add_to_cart_rate"] = result["n_carted"] / result["n_users"]
        return result

    def compute_all_metrics(
        self,
        events_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
        experiment_id: str,
    ) -> dict:
        """
        Run all three metric computations.

        Returns dict keyed by metric name:
            {
                "conversion_rate": DataFrame,
                "revenue_per_user": DataFrame,
                "add_to_cart_rate": DataFrame,
            }
        """
        logger.info(f"Computing all metrics for experiment '{experiment_id}'")
        return {
            "conversion_rate": self.compute_conversion_rate(
                events_df, assignments_df, experiment_id
            ),
            "revenue_per_user": self.compute_revenue_per_user(
                events_df, assignments_df, experiment_id
            ),
            "add_to_cart_rate": self.compute_add_to_cart_rate(
                events_df, assignments_df, experiment_id
            ),
        }
