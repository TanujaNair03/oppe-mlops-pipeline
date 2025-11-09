#!/usr/bin/env python3
"""
Fallback materialization:
- Reads data/processed/processed_data.parquet
- Writes feature_repo/offline_features.parquet (full features)
- Writes feature_repo/online_store.db (SQLite table 'features' with latest row per symbol)
"""
import os
import pandas as pd
import sqlite3

PROC = os.path.join("data", "processed", "processed_data.parquet")
OFF = os.path.join("feature_repo", "offline_features.parquet")
DB = os.path.join("feature_repo", "online_store.db")

print("[INFO] Reading processed parquet:", PROC)
df = pd.read_parquet(PROC)
print(f"[INFO] Data shape: {df.shape}")

# Ensure expected columns exist
if "timestamp" not in df.columns:
    raise SystemExit("[ERROR] timestamp column not found in processed parquet")
if "symbol" not in df.columns:
    df["symbol"] = "UNKNOWN"

# Write offline features (full dataset)
os.makedirs(os.path.dirname(OFF), exist_ok=True)
df.to_parquet(OFF, index=False)
print("[INFO] Wrote offline features to:", OFF)

# Create a simple online-store sqlite DB with latest-per-symbol snapshot
conn = sqlite3.connect(DB)
latest = df.sort_values(["symbol", "timestamp"]).groupby("symbol").tail(1)
latest.to_sql("features", conn, if_exists="replace", index=False)
conn.close()
print("[INFO] Wrote online store snapshot to:", DB)

print("[INFO] Fallback materialization complete.")
