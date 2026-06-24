"""
FMCC Daily Fraud Detection Pipeline — Ingestion Layer

Airflow's job: upload one day's CDR sample to S3 each day.
Lambda's job: detect the new file and trigger processing.

Schedule: 1am UTC daily (after CDRs are available)
Date range: 2026-06-23 to 2026-07-07 (simulation window)
"""
from datetime import datetime, timedelta, date
from io import StringIO

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3
import pandas as pd

RAW_BUCKET     = "fmcc-raw-cdrs-635863384351"
BIHAR_PATH     = "/opt/airflow/fmcc_data/MSCDATA_Bhiharcsv.csv"
RAJASTHAN_PATH = "/opt/airflow/fmcc_data/MSCDATA_40459 Rajasthan CSV.csv"
ROWS_PER_DAY   = 10_000
AWS_REGION     = "us-east-1"

# Simulation maps real execution dates → actual CDR data dates
SIM_START  = date(2026, 6, 23)   # day 0 of simulation
DATA_START = date(2025, 7, 1)    # first day of actual CDR data

CDR_COLS = [
    "CALLINGMSISDN", "CALLEDMSISDN", "CALLINGPARTYIMSI",
    "IMEI", "GLOBALCELLID", "DURATION", "DATETIMEOFCALL",
    "MOCALLFLAG", "MTCALLFLAG", "POSTPAIDFLAG", "PREPAIDFLAG",
    "NATIONALINROAMERFLAG", "INTERNATIONALINROAMERFLAG", "FORWARDINGFLAG",
    "CALLCONFFLAG", "SMSFLAG", "VOICEFLAG", "CAUSEOFTERM", "DATASTITCHFLAG",
]

default_args = {
    "owner": "fmcc",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def upload_cdrs_to_s3(ds, **kwargs):
    import logging
    logger = logging.getLogger(__name__)

    # Map execution date to CDR data date
    exec_date = date.fromisoformat(ds)
    day_index = (exec_date - SIM_START).days
    data_date = (DATA_START + timedelta(days=day_index)).isoformat()
    logger.info("Execution date %s → CDR data date %s (day %d)", ds, data_date, day_index)

    bih = pd.read_csv(BIHAR_PATH, usecols=lambda c: c in CDR_COLS, low_memory=False)
    raj = pd.read_csv(RAJASTHAN_PATH, usecols=lambda c: c in CDR_COLS, low_memory=False)

    data = pd.concat([bih, raj], ignore_index=True).dropna().drop_duplicates()
    data["DATETIMEOFCALL"] = pd.to_datetime(data["DATETIMEOFCALL"], format="mixed")
    data["DATE"] = data["DATETIMEOFCALL"].dt.date.astype(str)

    day_data = data[data["DATE"] == data_date].drop(columns=["DATE"])

    if len(day_data) == 0:
        logger.warning("No CDR data found for data_date=%s (exec=%s) — skipping", data_date, ds)
        return

    sample = day_data.sample(n=min(ROWS_PER_DAY, len(day_data)), random_state=42)
    logger.info("Sampled %d of %d rows for data_date=%s", len(sample), len(day_data), data_date)

    s3_key = f"cdrs/{ds}.csv"
    buf = StringIO()
    sample.to_csv(buf, index=False)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=RAW_BUCKET, Key=s3_key, Body=buf.getvalue())
    logger.info("Uploaded to s3://%s/%s — Lambda will trigger automatically", RAW_BUCKET, s3_key)


with DAG(
    dag_id="fmcc_daily_fraud_pipeline",
    default_args=default_args,
    description="Daily CDR upload to S3 — Lambda triggers processing automatically",
    schedule_interval="0 1 * * *",
    start_date=datetime(2026, 6, 23),
    end_date=datetime(2026, 7, 7),
    catchup=False,
    tags=["fmcc", "fraud", "production"],
) as dag:

    upload = PythonOperator(
        task_id="upload_cdrs_to_s3",
        python_callable=upload_cdrs_to_s3,
        provide_context=True,
    )
