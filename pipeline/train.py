"""
Training script — mirrors FMCC_1Day_Corrected.ipynb pipeline end-to-end.
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
import warnings
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def load_suspects(bihar_path: str, rajasthan_path: str) -> pd.DataFrame:
    """Return full suspect DataFrame — we need both MSISDN and REPORTDATE."""
    s1 = pd.read_csv(bihar_path)
    s2 = pd.read_csv(rajasthan_path)
    suspects = pd.concat([s1, s2], ignore_index=True)
    suspects["REPORTDATE"] = pd.to_datetime(suspects["REPORTDATE"], format="mixed").dt.date
    logger.info("Suspect records: %d", len(suspects))
    return suspects


def label_features(features: pd.DataFrame, suspects: pd.DataFrame) -> pd.DataFrame:
    """
    Cell 37 (corrected): label on (MSISDN, DATE) pair, not MSISDN alone.
    Only the specific date a MSISDN was flagged becomes isFraud=1.
    This prevents label smearing across all days of a suspect's activity.
    """
    suspect_pairs = set(zip(suspects["MSISDN"], suspects["REPORTDATE"]))
    features["isFraud"] = features.apply(
        lambda row: 1 if (row["MSISDN"], row["DATE"]) in suspect_pairs else 0,
        axis=1,
    )
    return features


def train(args):
    # ── Data loading ──────────────────────────────────────────────────────────
    data = load_cdrs(args.bihar, args.rajasthan)
    suspects = load_suspects(args.suspects_bihar, args.suspects_rajasthan)

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building 1-day features...")
    features = build_1day_features(data)

    # ── Labeling: (MSISDN, DATE) pair match — corrected notebook approach ─────
    features = label_features(features, suspects)
    logger.info(
        "Feature matrix: %d rows | Fraud: %d (%.2f%%)",
        len(features),
        features["isFraud"].sum(),
        features["isFraud"].mean() * 100,
    )

    # ── Out-of-time train/test split (Cell 38 corrected) ─────────────────────
    # Sort dates, use earliest 70% for train and latest 30% for test.
    # Prevents temporal leakage: model never sees future dates during training.
    X = features[FEATURE_COLS]
    y = features["isFraud"]

    all_dates = sorted(features["DATE"].unique())
    split_point = int(len(all_dates) * 0.7)
    train_dates = set(all_dates[:split_point])
    test_dates = set(all_dates[split_point:])

    train_mask = features["DATE"].isin(train_dates)
    test_mask = features["DATE"].isin(test_dates)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    logger.info(
        "Out-of-time split: train dates %s→%s (%d rows, %d fraud) | test dates %s→%s (%d rows, %d fraud)",
        min(train_dates), max(train_dates), len(X_train), int(y_train.sum()),
        min(test_dates),  max(test_dates),  len(X_test),  int(y_test.sum()),
    )

    # ── Model: Voting Ensemble — matches Cell 45 exactly ──────────────────────
    # Notebook feeds raw (unscaled) features to all three estimators
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
    )
    logger.info("Training Voting Ensemble (RF + GBM + LR)...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
