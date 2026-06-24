"""
Phase 5 — Daily drift monitor.

Compares today's feature distribution (from prediction_log) against the
training baseline. Uses Evidently AI. Publishes metric to CloudWatch.
Sends SNS alert if dataset drift is detected.

Can be run standalone (cron / Airflow) or imported.
"""
import json
import logging
import os
from datetime import date, timedelta
from typing import Optional

import boto3
import pandas as pd
import psycopg2
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine

from app.features import FEATURE_COLS

logger = logging.getLogger(__name__)

POSTGRES_URL  = os.getenv("POSTGRES_URL", "postgresql://fmcc:fmcc_secure_2025@172.31.18.63:5432/fmcc_fraud")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET     = os.getenv("FMCC_S3_ARTIFACTS", "fmcc-fraud-results-635863384351")
BASELINE_PATH = os.getenv("BASELINE_PATH", "baseline/training_features.parquet")


def load_baseline() -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    local = "/tmp/fmcc_baseline.parquet"
    s3.download_file(S3_BUCKET, BASELINE_PATH, local)
    return pd.read_parquet(local)[FEATURE_COLS]


def load_current(run_date: date) -> pd.DataFrame:
    """Pull today's feature snapshot from the prediction log."""
    engine = create_engine(POSTGRES_URL)
    query = f"SELECT fraud_probability FROM prediction_log WHERE date = '{run_date}'"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def load_current_from_s3(run_date: date) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    local = f"/tmp/fmcc_features_{run_date}.parquet"
    key = f"{run_date}/features.parquet"
    s3.download_file(S3_BUCKET, key, local)
    df = pd.read_parquet(local)
    shared = [c for c in FEATURE_COLS if c in df.columns]
    return df[shared]


def run_drift_report(run_date: Optional[date] = None) -> dict:
    run_date = run_date or date.today() - timedelta(days=1)
    logger.info("Running drift report for %s", run_date)

    baseline = load_baseline()

    try:
        current = load_current_from_s3(run_date)
    except Exception as e:
        logger.warning("S3 feature load failed (%s), falling back to prediction log", e)
        current = load_current(run_date)

    if current.empty:
        logger.warning("No current data for %s — skipping drift check", run_date)
        return {"drift_detected": False, "reason": "no_data"}

    # Align columns
    shared_cols = [c for c in FEATURE_COLS if c in current.columns]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline[shared_cols], current_data=current[shared_cols])

    result = report.as_dict()
    drift_metrics = result["metrics"][0]["result"]
    drift_detected = drift_metrics["dataset_drift"]
    drift_share    = drift_metrics.get("share_of_drifted_columns", 0.0)

    logger.info("Drift detected: %s (share=%.2f)", drift_detected, drift_share)

    # ── Publish to CloudWatch ──────────────────────────────────────────────────
    cw = boto3.client("cloudwatch", region_name=AWS_REGION)
    cw.put_metric_data(
        Namespace="FMCC/ModelMonitoring",
        MetricData=[
            {
                "MetricName": "FeatureDriftShare",
                "Value": drift_share,
                "Unit": "None",
                "Dimensions": [{"Name": "Date", "Value": str(run_date)}],
            },
            {
                "MetricName": "DriftDetected",
                "Value": 1 if drift_detected else 0,
                "Unit": "Count",
            },
        ],
    )

    # ── Write HTML report to S3 ───────────────────────────────────────────────
    html_path = f"/tmp/drift_{run_date}.html"
    report.save_html(html_path)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(html_path, S3_BUCKET, f"drift-reports/{run_date}/report.html")
    logger.info("Drift report uploaded to s3://%s/drift-reports/%s/report.html", S3_BUCKET, run_date)

    # ── Store per-feature drift in Postgres ───────────────────────────────────
    rows = []
    for col_result in drift_metrics.get("drift_by_columns", {}).values():
        rows.append({
            "report_date": run_date,
            "feature":     col_result.get("column_name", "unknown"),
            "drift_score": col_result.get("stattest_threshold", None),
            "drift_detected": col_result.get("drift_detected", False),
        })
    if rows:
        engine = create_engine(POSTGRES_URL)
        with engine.connect() as conn:
            pd.DataFrame(rows).to_sql("drift_report", conn, if_exists="append", index=False)

    # ── SNS alert ─────────────────────────────────────────────────────────────
    if drift_detected and SNS_TOPIC_ARN:
        drifted_cols = [
            c for c, v in drift_metrics.get("drift_by_columns", {}).items()
            if v.get("drift_detected")
        ]
        sns = boto3.client("sns", region_name=AWS_REGION)
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[FMCC ALERT] Feature drift detected — {run_date}",
            Message=(
                f"Date: {run_date}\n"
                f"Drifted columns ({len(drifted_cols)}): {', '.join(drifted_cols)}\n"
                f"Drift share: {drift_share:.1%}\n\n"
                f"Full report: s3://{S3_BUCKET}/drift-reports/{run_date}/report.html"
            ),
        )
        logger.info("SNS alert sent to %s", SNS_TOPIC_ARN)

    return {
        "date": str(run_date),
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_columns": [
            c for c, v in drift_metrics.get("drift_by_columns", {}).items()
            if v.get("drift_detected")
        ],
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_date_arg = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run_drift_report(run_date_arg)
    print(json.dumps(result, indent=2))
