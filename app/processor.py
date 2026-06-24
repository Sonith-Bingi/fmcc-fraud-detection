"""
Batch CDR processor — called by the /process endpoint.

Flow:
  1. Read raw CDR CSV from S3
  2. Call stitching + 1-day feature engineering
  3. Score every MSISDN via the loaded model
  4. Write results to PostgreSQL prediction_log
  5. Return summary stats
"""
import logging
import os
from io import StringIO
from datetime import datetime

import boto3
import pandas as pd
import psycopg2

from app.features import build_1day_features, FEATURE_COLS
from app import model as model_module

logger = logging.getLogger(__name__)

RAW_BUCKET     = os.getenv("RAW_BUCKET", "fmcc-raw-cdrs-635863384351")
RESULTS_BUCKET = os.getenv("RESULTS_BUCKET", "fmcc-fraud-results-635863384351")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")

DB_HOST = os.getenv("DB_HOST", "172.31.18.63")
DB_NAME = os.getenv("DB_NAME", "fmcc_fraud")
DB_USER = os.getenv("DB_USER", "fmcc")
DB_PASS = os.getenv("DB_PASS", "fmcc_secure_2025")
DB_PORT = int(os.getenv("DB_PORT", "5432"))


def process_s3_key(s3_key: str) -> dict:
    """
    Full pipeline for one day's CDR file.
    s3_key example: cdrs/2025-07-01.csv
    Returns summary dict with counts and fraud rate.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # 1. Read CDR file from S3
    logger.info("Reading s3://%s/%s", RAW_BUCKET, s3_key)
    obj = s3.get_object(Bucket=RAW_BUCKET, Key=s3_key)
    raw = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")), low_memory=False)
    logger.info("Loaded %d CDR rows", len(raw))

    # 2. Feature engineering
    features = build_1day_features(raw)
    logger.info("Engineered features for %d MSISDNs", len(features))

    # 3. Score via loaded model
    m = model_module.get_model()
    threshold = model_module.get_threshold()
    version = model_module.get_model_version()

    X = features[FEATURE_COLS]
    probas = model_module.predict(features)

    features = features.copy()
    features["fraud_probability"] = probas
    features["is_fraud"] = probas >= threshold
    features["risk_tier"] = [model_module.risk_tier(float(p), threshold) for p in probas]
    features["model_version"] = version
    features["threshold"] = threshold

    # 4. Save results CSV to S3
    date_str = s3_key.replace("cdrs/", "").replace(".csv", "")
    result_key = f"results/{date_str}_fraud_report.csv"
    result_cols = ["MSISDN", "DATE", "fraud_probability", "is_fraud", "risk_tier", "model_version", "threshold"]
    buf = StringIO()
    features[result_cols].to_csv(buf, index=False)
    s3.put_object(Bucket=RESULTS_BUCKET, Key=result_key, Body=buf.getvalue())
    logger.info("Results saved to s3://%s/%s", RESULTS_BUCKET, result_key)

    # 5. Save features parquet to S3 (used by drift monitor)
    feat_parquet_path = f"/tmp/features_{date_str}.parquet"
    features[FEATURE_COLS].to_parquet(feat_parquet_path, index=False)
    s3.upload_file(feat_parquet_path, RESULTS_BUCKET, f"{date_str}/features.parquet")
    logger.info("Features saved to s3://%s/%s/features.parquet", RESULTS_BUCKET, date_str)

    # 6. Write to PostgreSQL
    inserted = _write_to_postgres(features, date_str, version, threshold)

    # 7. Run drift detection
    drift_result = {}
    try:
        from monitoring.drift_monitor import run_drift_report
        from datetime import date as date_type
        drift_result = run_drift_report(date_type.fromisoformat(date_str))
        logger.info("Drift result: %s", drift_result)
    except Exception as e:
        logger.warning("Drift monitor failed (non-fatal): %s", e)

    fraud_count = int(features["is_fraud"].sum())
    total = len(features)
    fraud_rate = round(fraud_count / total * 100, 2) if total > 0 else 0

    # 8. Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://172.31.18.63:5001"))
        mlflow.set_experiment("FMCC-FraudDetection")
        with mlflow.start_run(run_name=f"daily-{date_str}"):
            mlflow.log_param("date", date_str)
            mlflow.log_param("model_version", version)
            mlflow.log_metrics({
                "msisdns_scored": total,
                "fraud_flagged": fraud_count,
                "fraud_rate_pct": fraud_rate,
                "drift_detected": 1 if drift_result.get("drift_detected") else 0,
                "drift_share": drift_result.get("drift_share") or 0.0,
            })
        logger.info("MLflow run logged for %s", date_str)
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)

    summary = {
        "date": date_str,
        "msisdns_scored": total,
        "fraud_flagged": fraud_count,
        "fraud_rate_pct": fraud_rate,
        "model_version": version,
        "rows_inserted_to_db": inserted,
        "drift_detected": drift_result.get("drift_detected"),
        "drift_share": drift_result.get("drift_share"),
    }
    logger.info("Summary: %s", summary)
    return summary


def _write_to_postgres(features: pd.DataFrame, date_str: str, version: str, threshold: float) -> int:
    try:
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME, user=DB_USER,
            password=DB_PASS, port=DB_PORT, connect_timeout=10
        )
        cur = conn.cursor()
        inserted = 0
        for _, row in features.iterrows():
            cur.execute("""
                INSERT INTO prediction_log
                    (msisdn, date, fraud_probability, is_fraud, risk_tier, model_version, threshold)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                int(row["MSISDN"]),
                date_str,
                float(row["fraud_probability"]),
                bool(row["is_fraud"]),
                str(row["risk_tier"]),
                version,
                threshold,
            ))
            inserted += 1
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Inserted %d rows into prediction_log", inserted)
        return inserted
    except Exception as e:
        logger.error("PostgreSQL write failed: %s", e)
        return 0
