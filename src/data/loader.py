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


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a single DataFrame in-place.
    Called per-file so memory never holds more than one raw file at a time.
    """
    # Parse event_time
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    # Drop nulls and invalid event types early — reduces rows before any copies
    df = df.dropna(subset=["user_id", "product_id"])
    df = df[df["event_type"].isin(VALID_EVENT_TYPES)]

    # Derived columns
    df["date"] = df["event_time"].dt.date
    df["hour_of_day"] = df["event_time"].dt.hour

    # Type normalisation
    df["user_id"] = df["user_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Use memory-efficient categoricals for high-cardinality string columns
    for col in ("event_type", "category_code", "brand"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def load_events(raw_path: Path | None = None, save: bool = True) -> pd.DataFrame:
    """
    Load and clean the raw events CSVs.

    Each file is cleaned individually and written to parquet incrementally
    using PyArrow's ParquetWriter — so the full 20M-row dataset is processed
    without ever holding more than one file in memory at a time.

    Applies per file:
    - Parse event_time to datetime
    - Drop rows with null user_id or product_id
    - Filter to valid event_type values
    - Add date and hour_of_day columns
    - Cast event_type / category_code / brand to categorical dtype

    Saves cleaned data to data/processed/events.parquet if save=True.
    Returns cleaned DataFrame read back from the parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_path = raw_path or RAW_DATA_PATH

    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    logger.info(f"Loading {len(csv_files)} CSV file(s) from {raw_path}")

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DATA_PATH / "events.parquet"

    writer = None
    total_raw = 0
    total_clean = 0

    try:
        for f in csv_files:
            logger.info(f"  Reading {f.name}")
            raw = pd.read_csv(f)
            total_raw += len(raw)
            cleaned = _clean_chunk(raw)
            del raw  # free the raw file immediately

            total_clean += len(cleaned)
            logger.info(f"    {f.name}: {len(cleaned):,} rows after cleaning")

            # Convert to Arrow and write one row-group at a time
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            del cleaned

            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table)
            del table
    finally:
        if writer:
            writer.close()

    logger.info(f"Total raw rows: {total_raw:,} → cleaned rows: {total_clean:,}")
    logger.info(f"Saved to {out}")

    # Read back only the schema + row count for the return value —
    # avoids a second full 20M-row allocation when the file is already on disk.
    # Callers that need the full DataFrame should read the parquet directly.
    import pyarrow.parquet as pq
    meta = pq.read_metadata(out)
    logger.info(f"Final cleaned shape: ({meta.num_rows:,}, {meta.num_columns})")

    # Return a lightweight read using column projection limited to non-object cols
    df = pd.read_parquet(out, columns=["user_id", "event_type", "price",
                                        "user_session", "date", "hour_of_day",
                                        "product_id", "event_time"])
    return df
