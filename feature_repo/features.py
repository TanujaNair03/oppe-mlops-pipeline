"""
Feast feature definitions for the processed stock dataset.

This file is defensive: it locates the repo root and tries to read the processed parquet
to infer numeric feature columns. If reading fails at import time (for example when Feast
imports the module from inside `feature_repo`), it falls back to a small default schema.

Entity: 'symbol'
FeatureView: stock_minute_features
"""
import os
from datetime import timedelta
from typing import List
import pandas as pd

from feast import Entity, FeatureView, Field, FileSource
from feast.types import ValueType

def repo_root() -> str:
    # feature_repo/ is sibling of repo root, so root is parent of feature_repo dir
    this_file = os.path.abspath(__file__)
    feature_repo_dir = os.path.dirname(this_file)
    root = os.path.dirname(feature_repo_dir)
    return root

def discover_numeric_columns(parquet_path: str, max_cols: int = 20) -> List[str]:
    """
    Try to read parquet and return numeric columns except target.
    If reading/parquet access fails, return a sensible fallback list.
    """
    try:
        df = pd.read_parquet(parquet_path)
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        nums = [c for c in nums if c != "target"]
        if len(nums) == 0:
            # fallback: common OHLCV names
            return ["open", "high", "low", "close", "volume"]
        return nums[:max_cols]
    except Exception as e:
        print(f"[WARN] Could not read parquet at {parquet_path}: {e}")
        print("[WARN] Falling back to default numeric feature names: open, high, low, close, volume")
        return ["open", "high", "low", "close", "volume"]

# Build absolute processed path (safe regardless of cwd)
ROOT = repo_root()
processed_parquet = os.path.join(ROOT, "data", "processed", "processed_data.parquet")

# Create FileSource for Feast (Feast will use this path)
processed_source = FileSource(
    path=processed_parquet,
    timestamp_field="timestamp",
)

# Define entity
symbol = Entity(name="symbol", value_type=ValueType.STRING, description="Stock symbol")

# Discover numeric columns (safe)
numeric_cols = discover_numeric_columns(processed_parquet)
fields = [Field(name=c, dtype=ValueType.DOUBLE) for c in numeric_cols]

# Create FeatureView
stock_minute_features = FeatureView(
    name="stock_minute_features",
    entities=["symbol"],
    ttl=timedelta(days=7),
    schema=fields,
    source=processed_source,
    tags={},
)