"""
MLflow-instrumented training script (Phase 4).
Wraps train.py with full experiment tracking, model registration,
and automated promotion to 'Production' stage when metrics improve.

Usage:
    python -m pipeline.train_mlflow \
        --bihar   data/MSCDATA_Bhiharcsv.csv \
        --rajasthan "data/MSCDATA_40459 Rajasthan CSV.csv" \
        --suspects_bihar   data/Suspectdata_Bihar.csv \
        --suspects_rajasthan data/Suspects_Rajasthan.csv

Env vars:
    MLFLOW_TRACKING_URI  : http://localhost:5000 (or RDS-backed URI)
    MLFLOW_EXPERIMENT    : FMCC-FraudDetection (default)
    PROMOTE_THRESHOLD_F1 : 0.80  — auto-promote if F1 exceeds this
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.train import train as _train, load_cdrs, load_suspects
from app.features import FEATURE_COLS, build_1day_features

EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT", "FMCC-FraudDetection")
MODEL_REGISTRY   = "fmcc-fraud-voting-ensemble"
PROMOTE_F1       = float(os.getenv("PROMOTE_THRESHOLD_F1", "0.80"))
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def run(args):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="voting-ensemble-1day") as run:
        # ── Log data params ───────────────────────────────────────────────────
        mlflow.log_params({
            "feature_window": "1-day",
            "split_strategy": "GroupShuffleSplit",
            "test_size": 0.20,
            "random_state": 42,
            "n_features": len(FEATURE_COLS),
            "ensemble": "RF+GBM+LR",
            "rf_n_estimators": 200,
            "rf_max_depth": 20,
            "gb_n_estimators": 100,
            "gb_max_depth": 5,
        })

        # ── Train ─────────────────────────────────────────────────────────────
        artifact_path, metrics = _train(args)

        # ── Log metrics ───────────────────────────────────────────────────────
        mlflow.log_metrics({
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "roc_auc":   metrics["roc_auc"],
        })

        # ── Log model to registry ─────────────────────────────────────────────
        import joblib
        artifact = joblib.load(artifact_path)
        model_obj = artifact["model"]
        mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY,
            input_example=None,
        )

        # ── Log feature list as artifact ──────────────────────────────────────
        feat_path = "/tmp/feature_cols.txt"
        with open(feat_path, "w") as f:
            f.write("\n".join(FEATURE_COLS))
        mlflow.log_artifact(feat_path, artifact_path="metadata")

        # ── Auto-promote if F1 beats threshold ────────────────────────────────
        if metrics["f1"] >= PROMOTE_F1:
            client = MlflowClient()
            # Get latest version just registered
            versions = client.get_latest_versions(MODEL_REGISTRY, stages=["None"])
            if versions:
                latest_ver = versions[0].version
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY,
                    version=latest_ver,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(f"[MLflow] Auto-promoted version {latest_ver} to Production (F1={metrics['f1']:.4f})")
                mlflow.set_tag("promoted_to_production", "true")
        else:
            print(f"[MLflow] F1={metrics['f1']:.4f} below threshold {PROMOTE_F1}. Staying in Staging.")
            mlflow.set_tag("promoted_to_production", "false")

        print(f"[MLflow] Run ID: {run.info.run_id}")
        print(f"[MLflow] Experiment: {EXPERIMENT_NAME}")
        print(f"[MLflow] Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bihar", required=True)
    parser.add_argument("--rajasthan", required=True)
    parser.add_argument("--suspects_bihar", required=True)
    parser.add_argument("--suspects_rajasthan", required=True)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    run(args)
