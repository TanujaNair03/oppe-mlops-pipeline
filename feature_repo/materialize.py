#!/usr/bin/env python3
"""
Materialize Feast features between two timestamps.
"""
import argparse
from feast import FeatureStore

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="1970-01-01T00:00:00", help="start timestamp")
    p.add_argument("--end", default="2099-01-01T00:00:00", help="end timestamp")
    return p.parse_args()

def main():
    args = parse_args()
    fs = FeatureStore(repo_path="feature_repo")
    print(f"[INFO] Materializing from {args.start} to {args.end}")
    fs.materialize(args.start, args.end)
    print("[INFO] Materialization complete!")

if __name__ == "__main__":
    main()
