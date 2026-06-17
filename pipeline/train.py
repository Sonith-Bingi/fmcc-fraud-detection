"""
Training script — mirrors the FMCC_1Day.ipynb pipeline end-to-end.
Reads raw CDR CSVs, engineers 1-day features, trains Voting Ensemble,
saves artifact to models/<version>.pkl.

Usage:
    python -m pipeline.train \
        --bihar   /path/to/MSCDATA_Bhiharcsv.csv \
        --rajasthan /path/to/MSCDATA_40459\ Rajasthan\ CSV.csv \
        --suspects_bihar /path/to/Suspectdata_Bihar.csv \
        --suspects_rajasthan /path/to/Suspects_Rajasthan.csv \
        --output  models/
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# Allow running from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.features import FEATURE_COLS, build_1day_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CDR_COLS = [
    "CALLINGMSISDN", "CALLEDMSISDN", "CALLINGPARTYIMSI",
    "IMEI", "GLOBALCELLID", "DURATION", "DATETIMEOFCALL", "MSCID",
    "MOCALLFLAG", "MTCALLFLAG", "POSTPAIDFLAG", "PREPAIDFLAG",
    "NATIONALINROAMERFLAG", "INTERNATIONALINROAMERFLAG", "FORWARDINGFLAG",
    "CALLCONFFLAG", "SMSFLAG", "VOICEFLAG", "CAUSEOFTERM", "DATASTITCHFLAG",
]


def load_cdrs(bihar_path: str, rajasthan_path: str) -> pd.DataFrame:
    logger.info("Loading Bihar CDRs from %s", bihar_path)
    bih = pd.read_csv(bihar_path, usecols=lambda c: c in CDR_COLS, low_memory=False)
    logger.info("  Bihar: %d rows", len(bih))

    logger.info("Loading Rajasthan CDRs from %s", rajasthan_path)
    raj = pd.read_csv(rajasthan_path, usecols=lambda c: c in CDR_COLS, low_memory=False)
    logger.info("  Rajasthan: %d rows", len(raj))

    data = pd.concat([bih, raj], ignore_index=True)
    data = data.dropna().drop_duplicates()
    logger.info("Combined: %d rows after clean", len(data))
    return data


def load_suspects(bihar_path: str, rajasthan_path: str) -> pd.Series:
    s1 = pd.read_csv(bihar_path)
    s2 = pd.read_csv(rajasthan_path)
    suspects = pd.concat([s1, s2], ignore_index=True)
    logger.info("Suspect MSISDNs: %d", len(suspects))
    return suspects["MSISDN"]


def train(args):
    # ── Data loading ──────────────────────────────────────────────────────────
    data = load_cdrs(args.bihar, args.rajasthan)
    suspects = load_suspects(args.suspects_bihar, args.suspects_rajasthan)

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building 1-day features...")
    features = build_1day_features(data)
    suspects = suspects.astype(features["MSISDN"].dtype)
    features["isFraud"] = features["MSISDN"].isin(suspects).astype(int)
    logger.info(
        "Feature matrix: %d rows | Fraud: %d (%.2f%%)",
        len(features),
        features["isFraud"].sum(),
        features["isFraud"].mean() * 100,
    )

    # ── Train / test split (group by MSISDN — no leakage) ────────────────────
    X = features[FEATURE_COLS]
    y = features["isFraud"]
    groups = features["MSISDN"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    logger.info(
        "Train: %d rows (%d fraud) | Test: %d rows (%d fraud)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum(),
    )

    # ── Model: Voting Ensemble ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    # VotingClassifier needs all estimators to see the same X; LR needs scaled
    # We wrap LR with a pipeline so VotingClassifier stays homogeneous
    from sklearn.pipeline import Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", lr)])

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr_pipe)],
        voting="soft",
        n_jobs=-1,
    )
    logger.info("Training Voting Ensemble (RF + GBM + LR)...")
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0, 4),
    }
    logger.info("Metrics: %s", metrics)

    # ── Find optimal threshold ────────────────────────────────────────────────
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_threshold = float(thresholds[np.argmax(f1s)])
    logger.info("Optimal threshold: %.4f", best_threshold)

    # ── Save artifact ─────────────────────────────────────────────────────────
    version = f"voting_ensemble_1day_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "version": version,
        "threshold": best_threshold,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }
    artifact_path = output_dir / f"{version}.pkl"
    joblib.dump(artifact, artifact_path)
    logger.info("Saved model artifact: %s", artifact_path)

    # Also write metrics JSON for MLflow / CI checks
    metrics_path = output_dir / f"{version}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "threshold": best_threshold, "version": version}, f, indent=2)
    logger.info("Saved metrics: %s", metrics_path)

    return str(artifact_path), metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FMCC fraud detection model")
    parser.add_argument("--bihar", required=True)
    parser.add_argument("--rajasthan", required=True)
    parser.add_argument("--suspects_bihar", required=True)
    parser.add_argument("--suspects_rajasthan", required=True)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train(args)
