"""
PySpark-based user-level feature aggregation over the full events dataset.
Produces one row per user with 12 aggregate features.

Set use_pandas=True (or env var SPARK_USE_PANDAS=1) to run with pandas instead
of PySpark — useful for unit tests and Windows environments without winutils.
"""
import os
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))


def _aggregate_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure-pandas implementation of the user-level aggregation.
    Produces the same 12 columns as the Spark version.
    Used for unit tests and Windows environments without winutils.
    """
    sessions = df.groupby("user_id")["user_session"].nunique().rename("total_sessions")
    views_df = df[df["event_type"] == "view"]
    carts_df = df[df["event_type"] == "cart"]
    purchases_df = df[df["event_type"] == "purchase"]

    total_views = df[df["event_type"] == "view"].groupby("user_id").size().rename("total_views")
    total_carts = df[df["event_type"] == "cart"].groupby("user_id").size().rename("total_carts")
    total_purchases = purchases_df.groupby("user_id").size().rename("total_purchases")
    total_revenue = purchases_df.groupby("user_id")["price"].sum().rename("total_revenue")

    session_len = (
        df.groupby(["user_id", "user_session"]).size()
        .reset_index(name="n")
        .groupby("user_id")["n"].mean()
        .rename("avg_session_length")
    )

    activity = df.groupby("user_id").agg(
        days_active=("date", "nunique"),
        first_seen=("event_time", "min"),
        last_seen=("event_time", "max"),
    )

    fav = (
        views_df[views_df["category_code"].notna()]
        .groupby(["user_id", "category_code"]).size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("user_id")
        .set_index("user_id")["category_code"]
        .rename("favorite_category")
    )

    avg_price = views_df.groupby("user_id")["price"].mean().rename("avg_price_viewed")

    all_users = pd.DataFrame({"user_id": df["user_id"].unique()}).set_index("user_id")
    result = (
        all_users
        .join(sessions)
        .join(total_views)
        .join(total_carts)
        .join(total_purchases)
        .join(total_revenue)
        .join(session_len)
        .join(activity)
        .join(fav)
        .join(avg_price)
        .fillna({
            "total_sessions": 0, "total_views": 0, "total_carts": 0,
            "total_purchases": 0, "total_revenue": 0.0,
            "avg_session_length": 0.0, "days_active": 0, "avg_price_viewed": 0.0,
        })
        .reset_index()
    )
    return result


class SparkAggregator:
    """Computes user-level aggregates using PySpark (or pandas in test mode)."""

    def __init__(self, app_name: str = "ABTestingAggregator", use_pandas: bool = False):
        self._spark = None
        self.app_name = app_name
        self.use_pandas = use_pandas or os.getenv("SPARK_USE_PANDAS", "0") == "1"

    def _get_spark(self):
        if self._spark is None:
            from pyspark.sql import SparkSession
            self._spark = (
                SparkSession.builder
                .appName(self.app_name)
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.driver.memory", "4g")
                .getOrCreate()
            )
            self._spark.sparkContext.setLogLevel("WARN")
        return self._spark

    def compute_user_features(
        self,
        events_path: Path | None = None,
        save: bool = True,
    ):
        """
        Compute per-user aggregate features from the events parquet file.

        Output columns:
            user_id, total_sessions, total_views, total_carts, total_purchases,
            total_revenue, avg_session_length, days_active, first_seen, last_seen,
            favorite_category, avg_price_viewed

        Returns a pandas DataFrame.
        """
        events_path = events_path or (PROCESSED_DATA_PATH / "events.parquet")
        logger.info(f"Reading events from {events_path}")

        if self.use_pandas:
            df = pd.read_parquet(events_path)
            pdf = _aggregate_with_pandas(df)
            logger.info(f"User features shape: {pdf.shape}")
            if save:
                PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
                out = PROCESSED_DATA_PATH / "user_features.parquet"
                pdf.to_parquet(out, index=False)
                logger.info(f"Saved user features to {out}")
            return pdf

        from pyspark.sql import functions as F

        spark = self._get_spark()

        logger.info(f"Reading events from {events_path}")
        df = spark.read.parquet(str(events_path))

        # Sessions per user (distinct user_session values)
        sessions = df.groupBy("user_id").agg(
            F.countDistinct("user_session").alias("total_sessions")
        )

        # Event type counts
        views = df.filter(F.col("event_type") == "view").groupBy("user_id").agg(
            F.count("*").alias("total_views")
        )
        carts = df.filter(F.col("event_type") == "cart").groupBy("user_id").agg(
            F.count("*").alias("total_carts")
        )
        purchases = df.filter(F.col("event_type") == "purchase").groupBy("user_id").agg(
            F.count("*").alias("total_purchases"),
            F.sum("price").alias("total_revenue"),
        )

        # Session length: events per session, averaged per user
        session_lengths = (
            df.groupBy("user_id", "user_session")
            .agg(F.count("*").alias("session_events"))
            .groupBy("user_id")
            .agg(F.avg("session_events").alias("avg_session_length"))
        )

        # Activity window
        activity = df.groupBy("user_id").agg(
            F.countDistinct("date").alias("days_active"),
            F.min("event_time").alias("first_seen"),
            F.max("event_time").alias("last_seen"),
        )

        # Favorite category (most viewed)
        fav_category = (
            df.filter(F.col("event_type") == "view")
            .filter(F.col("category_code").isNotNull())
            .groupBy("user_id", "category_code")
            .agg(F.count("*").alias("n"))
            .orderBy("user_id", F.desc("n"))
            .dropDuplicates(["user_id"])
            .select("user_id", F.col("category_code").alias("favorite_category"))
        )

        # Avg price of viewed items
        avg_price = (
            df.filter(F.col("event_type") == "view")
            .groupBy("user_id")
            .agg(F.avg("price").alias("avg_price_viewed"))
        )

        # Join everything
        all_users = df.select("user_id").distinct()
        result = (
            all_users
            .join(sessions, "user_id", "left")
            .join(views, "user_id", "left")
            .join(carts, "user_id", "left")
            .join(purchases, "user_id", "left")
            .join(session_lengths, "user_id", "left")
            .join(activity, "user_id", "left")
            .join(fav_category, "user_id", "left")
            .join(avg_price, "user_id", "left")
            .fillna(0, subset=[
                "total_sessions", "total_views", "total_carts",
                "total_purchases", "total_revenue", "avg_session_length",
                "days_active", "avg_price_viewed"
            ])
        )

        pdf = result.toPandas()
        logger.info(f"User features shape: {pdf.shape}")

        if save:
            PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
            out = PROCESSED_DATA_PATH / "user_features.parquet"
            pdf.to_parquet(out, index=False)
            logger.info(f"Saved user features to {out}")

        return pdf

    def stop(self):
        if self._spark:
            self._spark.stop()
            self._spark = None
