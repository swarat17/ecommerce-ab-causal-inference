"""
One-shot script: clean events → user features.
Run: python scripts/build_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["SPARK_USE_PANDAS"] = "1"

from src.data.loader import load_events
from src.data.spark_aggregator import SparkAggregator

print("Loading events...")
df = load_events()
print(f"Events shape: {df.shape}")

print("Computing user features...")
agg = SparkAggregator(use_pandas=True)
features = agg.compute_user_features()
print(f"User features shape: {features.shape}")

print("Done.")
