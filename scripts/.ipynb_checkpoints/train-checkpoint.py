#!/usr/bin/env python3
"""
Train a RandomForest on processed features, with Hyperopt tuning and MLflow logging.

Fixes:
- Drops timestamp and other non-numeric columns from X before scaling.
- Ensures y is numeric.
- Uses quniform ranges and casts to int where needed.
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_file", required=True, help="Path to processed parquet file")
    p.add_argument("--target_col", default="target")
    p.add_argument("--timestamp_col", default="timestamp")
    p.add_argument("--mlflow_uri", default="http://127.0.0.1:5000")
    p.add_argument("--max_evals", type=int, default=10)
    return p.parse_args()

def load_data(path, target_col, ts_col):
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded data shape: {df.shape}")
    # Ensure target exists and is numeric
    if target_col not in df.columns:
        raise SystemExit(f"[ERROR] target column '{target_col}' not found in {path}")
    y = df[target_col].astype(int)

    # Drop helper columns and timestamps from features
    X = df.drop(columns=[target_col, "__source_file"], errors="ignore")

    # Drop timestamp column if present
    if ts_col in X.columns:
        print(f"[INFO] Dropping timestamp column '{ts_col}' from features")
        X = X.drop(columns=[ts_col])

    # Keep numeric columns only (drop any object / datetime columns)
    X_numeric = X.select_dtypes(include=[np.number])
    print(f"[INFO] Using numeric feature columns: {len(X_numeric.columns)} cols")
    # Fill NaNs
    X_numeric = X_numeric.fillna(0)
    return X_numeric, y

def objective_factory(X_train, X_val, y_train, y_val):
    def objective(params):
        # Cast hyperopt floats to ints where necessary
        n_estimators = int(params["n_estimators"])
        max_depth = int(params["max_depth"])
        min_samples_split = int(params["min_samples_split"])

        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            })

            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, zero_division=0)

            mlflow.log_metric("val_accuracy", float(acc))
            mlflow.log_metric("val_f1", float(f1))

        # minimize loss (-accuracy so higher accuracy -> lower loss)
        return {"loss": -float(acc), "status": STATUS_OK}
    return objective

def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("exam_experiment")

    X, y = load_data(args.processed_file, args.target_col, args.timestamp_col)
    # sanity: ensure no non-numeric columns remain
    non_num = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_num:
        print(f"[WARN] Non-numeric columns still present: {non_num}. They will be dropped.")
        X = X.select_dtypes(include=[np.number])

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    print(f"[INFO] X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Hyperopt search space (quniform gives floats -> cast to int)
    search_space = {
        "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
        "max_depth": hp.quniform("max_depth", 3, 30, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    }

    print("[INFO] Starting hyperparameter search with Hyperopt...")
    trials = Trials()
    objective = objective_factory(X_train, X_val, y_train, y_val)
    best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials)
    print(f"[INFO] Hyperopt best (raw): {best}")

    # Map best (quniform floats) to integers
    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_split": int(best["min_samples_split"])
    }
    print(f"[INFO] Best params (int): {best_params}")

    # Retrain on full train+val (combine)
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    final_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=42,
            n_jobs=-1
        ))
    ])

    final_pipe.fit(X_full, y_full)
    # evaluate on holdout (we'll reuse X_val as a proxy here)
    preds = final_pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, zero_division=0)
    print("[INFO] Final evaluation on validation set:")
    print(f"       val_accuracy = {acc:.4f}, val_f1 = {f1:.4f}")
    print(classification_report(y_val, preds, zero_division=0))

    with mlflow.start_run(run_name="final_random_forest"):
        mlflow.log_params(best_params)
        mlflow.log_metric("val_accuracy", float(acc))
        mlflow.log_metric("val_f1", float(f1))
        mlflow.sklearn.log_model(final_pipe, artifact_path="model", registered_model_name="exam_random_forest")
        print("[INFO] Final model logged and registered (if registry enabled).")

if __name__ == "__main__":
    main()
