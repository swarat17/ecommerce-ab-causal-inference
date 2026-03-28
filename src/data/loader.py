"""
Loads raw e-commerce events CSV, cleans it, and saves to parquet.
"""
import os
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

VALID_EVENT_TYPES = {"view", "cart", "purchase", "remove_from_cart"}


def load_events(raw_path: Path | None = None, save: bool = True) -> pd.DataFrame:
    """
    Load and clean the raw events CSV.

    Applies:
    - Parse event_time to datetime
    - Drop rows with null user_id or product_id
    - Add date column (date portion of event_time)
    - Add hour_of_day column (0-23)
    - Filter to valid event_type values

    Saves cleaned data to data/processed/events.parquet if save=True.
    Returns cleaned DataFrame.
    """
    raw_path = raw_path or RAW_DATA_PATH

    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    logger.info(f"Loading {len(csv_files)} CSV file(s) from {raw_path}")
    dfs = []
    for f in csv_files:
        logger.info(f"  Reading {f.name}")
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} raw rows")

    # Parse event_time
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    # Drop nulls
    before = len(df)
    df = df.dropna(subset=["user_id", "product_id"])
    logger.info(f"Dropped {before - len(df):,} rows with null user_id or product_id")

    # Filter valid event types
    df = df[df["event_type"].isin(VALID_EVENT_TYPES)]

    # Add derived columns
    df["date"] = df["event_time"].dt.date
    df["hour_of_day"] = df["event_time"].dt.hour

    # Ensure correct types
    df["user_id"] = df["user_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    logger.info(f"Final cleaned shape: {df.shape}")

    if save:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DATA_PATH / "events.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved to {out}")

    return df
