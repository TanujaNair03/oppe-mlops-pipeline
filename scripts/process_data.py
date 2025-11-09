#!/usr/bin/env python3
"""
Process raw CSV files into a single processed parquet for modeling.

- Reads all CSV files from data/raw/
- Parses timestamp (column name provided by --timestamp_col)
- Sorts by timestamp, fills missing values, creates simple lag features
- Ensures an entity column 'symbol' exists (Feast needs an entity)
- Outputs data/processed/processed_data.parquet
- Prints informative logs for examiners
"""
import os
import sys
import glob
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/raw", help="raw CSV directory")
    p.add_argument("--output_file", default="data/processed/processed_data.parquet", help="processed parquet output")
    p.add_argument("--timestamp_col", default="timestamp", help="timestamp column name")
    p.add_argument("--target_col", default="target", help="target column name")
    return p.parse_args()

def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Read {path} -> shape {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        raise

def ensure_timestamp(df, ts_col):
    if ts_col not in df.columns:
        raise RuntimeError(f"Timestamp column '{ts_col}' not found in columns: {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    if df[ts_col].isnull().any():
        n_null = df[ts_col].isnull().sum()
        print(f"[WARN] {n_null} timestamps could not be parsed; dropping those rows.")
        df = df.dropna(subset=[ts_col])
    return df

def main():
    args = parse_args()
    input_dir = args.input_dir
    out_file = args.output_file
    ts_col = args.timestamp_col
    target_col = args.target_col

    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        print(f"[ERROR] No CSV files found in {input_dir}. Place your CSVs into this folder and re-run.")
        sys.exit(2)

    dfs = []
    for f in files:
        df = safe_read_csv(f)
        df['__source_file'] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"[INFO] Concatenated {len(files)} files -> shape {df.shape}")

    df = ensure_timestamp(df, ts_col)
    df = df.sort_values(by=ts_col).reset_index(drop=True)

    if 'symbol' not in df.columns:
        print("[WARN] 'symbol' column not found. Adding default symbol 'UNKNOWN'.")
        df['symbol'] = 'UNKNOWN'

    # numeric and categorical detection
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude timestamp if pandas left it as dtype datetime64
    if ts_col in num_cols:
        num_cols = [c for c in num_cols if c != ts_col]
    cat_cols = [c for c in df.columns if c not in num_cols and c != ts_col and c not in ['__source_file']]

    print(f"[INFO] Numeric columns: {num_cols}")
    print(f"[INFO] Categorical columns: {cat_cols}")

    # Fill numeric NAs with median
    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Fill categorical NAs with 'NA'
    for c in cat_cols:
        df[c] = df[c].fillna("NA")

    # Create simple lag features per symbol for numeric columns
    LAGS = [1, 2]
    for c in num_cols:
        if c == target_col:
            continue
        for lag in LAGS:
            newcol = f"{c}_lag{lag}"
            df[newcol] = df.groupby('symbol')[c].shift(lag)
            df[newcol] = df[newcol].fillna(df[c].median())

    print("[INFO] Created lag features for numeric columns (if applicable)")

    # Label encode low-cardinality categorical columns
    try:
        from sklearn.preprocessing import LabelEncoder
        for c in cat_cols:
            if df[c].nunique() < 200:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                print(f"[INFO] Label-encoded column: {c}")
    except Exception as e:
        print(f"[WARN] sklearn not available for label encoding: {e}")

    # Ensure target column exists
    if target_col not in df.columns:
        print(f"[WARN] target column '{target_col}' not found. Creating dummy column with zeros.")
        df[target_col] = 0

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_parquet(out_file, index=False)
    print(f"[INFO] Saved processed data to {out_file} with shape {df.shape}")

if __name__ == "__main__":
    main()
