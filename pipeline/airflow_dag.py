"""
FMCC Daily CDR Fraud Detection Pipeline — Airflow DAG

Runs daily at 02:00 UTC:
  1. Wait for CDR file in S3
  2. Trigger EMR job (PySpark feature engineering)
  3. Poll EMR until complete
  4. Run retraining check — retrain if feature drift detected
  5. Update model in S3 + notify API to reload
  6. Run Evidently drift report

Set these Airflow Variables:
  fmcc_s3_raw_bucket   : s3://your-raw-cdr-bucket
  fmcc_s3_feature_store: s3://your-feature-store-bucket
  fmcc_emr_cluster_id  : j-XXXXXXXXXX (or leave blank to create ad-hoc)
  fmcc_api_url         : https://your-api-gateway-url
  fmcc_sns_topic_arn   : arn:aws:sns:us-east-1:XXXX:fmcc-alerts

Set these Airflow Connections:
  aws_default          : AWS credentials
  fmcc_postgres        : PostgreSQL prediction log DB
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.amazon.aws.operators.emr import (
    EmrAddStepsOperator,
    EmrCreateJobFlowOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.utils.trigger_rule import TriggerRule

# ── Config pulled from Airflow Variables ──────────────────────────────────────
S3_RAW       = Variable.get("fmcc_s3_raw_bucket",    default_var="s3://fmcc-raw-cdrs")
S3_FEATURES  = Variable.get("fmcc_s3_feature_store", default_var="s3://fmcc-features")
S3_ARTIFACTS = Variable.get("fmcc_s3_artifacts",     default_var="s3://fmcc-artifacts")
API_URL      = Variable.get("fmcc_api_url",           default_var="http://localhost:8000")
SNS_TOPIC    = Variable.get("fmcc_sns_topic_arn",     default_var="")

EMR_JOB_FLOW_OVERRIDES = {
    "Name": "fmcc-feature-engineering-{{ ds }}",
    "ReleaseLabel": "emr-7.2.0",
    "Applications": [{"Name": "Spark"}],
    "Instances": {
        "InstanceGroups": [
            {"Name": "Master", "Market": "ON_DEMAND", "InstanceRole": "MASTER",
             "InstanceType": "m5.xlarge", "InstanceCount": 1},
            {"Name": "Core",   "Market": "SPOT",      "InstanceRole": "CORE",
             "InstanceType": "m5.xlarge", "InstanceCount": 2},
        ],
        "KeepJobFlowAliveWhenNoSteps": False,
        "TerminationProtected": False,
    },
    "JobFlowRole": "EMR_EC2_DefaultRole",
    "ServiceRole": "EMR_DefaultRole",
    "LogUri": f"{S3_ARTIFACTS}/emr-logs/",
    "Tags": [{"Key": "Project", "Value": "FMCC"}, {"Key": "Env", "Value": "prod"}],
}

EMR_SPARK_STEP = [
    {
        "Name": "FMCC 1-Day Feature Engineering",
        "ActionOnFailure": "CONTINUE",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--conf", "spark.sql.shuffle.partitions=200",
                f"{S3_ARTIFACTS}/scripts/pyspark_features.py",
                "--input",  f"{S3_RAW}/{{{{ ds }}}}/",
                "--output", f"{S3_FEATURES}/{{{{ ds }}}}/",
                "--date",   "{{ ds }}",
            ],
        },
    }
]

default_args = {
    "owner": "fmcc-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="fmcc_daily_cdr_pipeline",
    description="Daily CDR ingestion → PySpark features → fraud scoring",
    schedule="0 2 * * *",
    start_date=datetime(2025, 7, 1),
    catchup=False,
    default_args=default_args,
    tags=["fmcc", "fraud", "production"],
    doc_md=__doc__,
) as dag:

    # 1. Wait for today's CDR file to land in S3
    wait_for_cdr = S3KeySensor(
        task_id="wait_for_cdr_file",
        bucket_name="fmcc-raw-cdrs",
        bucket_key="{{ ds }}/CDR_*.csv",
        wildcard_match=True,
        aws_conn_id="aws_default",
        timeout=3600,
        poke_interval=300,
        mode="reschedule",
    )

    # 2. Create EMR cluster
    create_emr = EmrCreateJobFlowOperator(
        task_id="create_emr_cluster",
        job_flow_overrides=EMR_JOB_FLOW_OVERRIDES,
        aws_conn_id="aws_default",
    )

    # 3. Submit PySpark feature engineering step
    submit_spark = EmrAddStepsOperator(
        task_id="submit_spark_step",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster', key='return_value') }}",
        steps=EMR_SPARK_STEP,
        aws_conn_id="aws_default",
    )

    # 4. Wait for Spark step to finish
    wait_spark = EmrStepSensor(
        task_id="wait_for_spark_step",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster', key='return_value') }}",
        step_id="{{ task_instance.xcom_pull('submit_spark_step', key='return_value')[0] }}",
        aws_conn_id="aws_default",
        poke_interval=60,
        timeout=7200,
    )

    # 5. Terminate EMR cluster
    terminate_emr = EmrTerminateJobFlowOperator(
        task_id="terminate_emr_cluster",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster', key='return_value') }}",
        aws_conn_id="aws_default",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # 6. Run daily drift check + optional retrain
    def run_drift_and_score(**context):
        """Pull today's features from S3, score via API, log to Postgres."""
        import boto3, json, requests, psycopg2
        from datetime import date

        ds = context["ds"]
        s3 = boto3.client("s3")

        # Download today's feature parquet
        local_path = f"/tmp/features_{ds}.parquet"
        bucket = "fmcc-features"
        key = f"{ds}/features.parquet"
        s3.download_file(bucket, key, local_path)

        import pandas as pd
        features = pd.read_parquet(local_path)

        # Batch score via API
        records = features.rename(columns={
            "outgoing_duration_1day": "outgoing_duration",
            "incoming_duration_1day": "incoming_duration",
            "unique_recipients_1day": "unique_recipients",
            "total_calls_1day": "total_calls",
            "unique_cell_ids_1day": "unique_cell_ids",
            "imei_count_1day": "imei_count",
            "short_call_count_1day": "short_call_count",
        }).assign(date=ds).to_dict(orient="records")

        # Score in batches of 500
        results = []
        for i in range(0, len(records), 500):
            batch = records[i:i+500]
            r = requests.post(f"{API_URL}/predict", json={"records": batch}, timeout=30)
            r.raise_for_status()
            results.extend(r.json()["results"])

        # Write to Postgres
        conn_str = Variable.get("fmcc_postgres_url")
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        for res in results:
            cur.execute(
                """INSERT INTO prediction_log
                   (msisdn, date, fraud_probability, is_fraud, risk_tier, model_version, threshold)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (res["msisdn"], res["date"], res["fraud_probability"],
                 res["is_fraud"], res["risk_tier"], res["model_version"], res["threshold_used"]),
            )
        conn.commit()
        cur.close()
        conn.close()

        fraud_count = sum(1 for r in results if r["is_fraud"])
        context["task_instance"].xcom_push(key="fraud_count", value=fraud_count)
        print(f"Scored {len(results)} MSISDNs | Fraud flagged: {fraud_count}")

    score_and_log = PythonOperator(
        task_id="score_and_log_predictions",
        python_callable=run_drift_and_score,
    )

    # 7. Run Evidently drift report
    def run_evidently_drift(**context):
        import boto3, json
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ds = context["ds"]
        s3 = boto3.client("s3")

        # Load today vs baseline (training distribution)
        current = pd.read_parquet(f"/tmp/features_{ds}.parquet")
        baseline_key = "baseline/training_features.parquet"
        s3.download_file("fmcc-features", baseline_key, "/tmp/baseline.parquet")
        baseline = pd.read_parquet("/tmp/baseline.parquet")

        from app.features import FEATURE_COLS
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline[FEATURE_COLS], current_data=current[FEATURE_COLS])

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]

        # Upload HTML report to S3
        report_path = f"/tmp/drift_report_{ds}.html"
        report.save_html(report_path)
        s3.upload_file(report_path, "fmcc-artifacts", f"drift-reports/{ds}/report.html")

        # Alert via SNS if drift detected
        if drift_detected and SNS_TOPIC:
            sns = boto3.client("sns", region_name="us-east-1")
            sns.publish(
                TopicArn=SNS_TOPIC,
                Subject=f"[FMCC] Feature drift detected — {ds}",
                Message=json.dumps(result["metrics"][0]["result"], indent=2),
            )
            print(f"SNS alert sent: drift detected on {ds}")

        context["task_instance"].xcom_push(key="drift_detected", value=drift_detected)
        print(f"Drift detected: {drift_detected}")

    evidently_drift = PythonOperator(
        task_id="run_evidently_drift_report",
        python_callable=run_evidently_drift,
    )

    # ── DAG wiring ────────────────────────────────────────────────────────────
    wait_for_cdr >> create_emr >> submit_spark >> wait_spark >> terminate_emr
    wait_spark >> score_and_log >> evidently_drift
