"""
PySpark 1-Day Feature Engineering Job
Runs on AWS EMR. Reads raw CDR CSVs from S3, produces per-MSISDN-DATE
feature parquet written back to S3 feature store.

Usage (spark-submit):
    spark-submit pyspark_features.py \
        --input  s3://fmcc-raw-cdrs/2025-07-01/ \
        --output s3://fmcc-features/2025-07-01/ \
        --date   2025-07-01
"""
import argparse
import sys
from datetime import datetime

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType

CDR_COLS = [
    "CALLINGMSISDN", "CALLEDMSISDN", "IMEI", "GLOBALCELLID",
    "DURATION", "DATETIMEOFCALL",
]


def build_session(app_name: str = "FMCC-FeatureEngineering") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def load_cdrs(spark: SparkSession, input_path: str, run_date: str):
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{input_path}*.csv")
        .select(*CDR_COLS)
        .dropna()
        .dropDuplicates()
    )

    df = df.withColumn(
        "DATETIMEOFCALL",
        F.to_timestamp("DATETIMEOFCALL", "dd-MM-yyyy HH:mm")
    )
    df = df.withColumn("DATE", F.to_date("DATETIMEOFCALL"))
    df = df.withColumn("DURATION", df["DURATION"].cast(DoubleType()))
    df = df.withColumn("CALLINGMSISDN", df["CALLINGMSISDN"].cast(LongType()))
    df = df.withColumn("CALLEDMSISDN",  df["CALLEDMSISDN"].cast(LongType()))

    # Filter to the run date only (1-day window)
    df = df.filter(F.col("DATE") == F.lit(run_date))
    return df


def build_features(df):
    # ── Short call flag ───────────────────────────────────────────────────────
    df = df.withColumn("is_short_call", (F.col("DURATION") <= 15).cast(IntegerType()))
    df = df.withColumn("hour", F.hour("DATETIMEOFCALL"))
    df = df.withColumn(
        "is_night",
        ((F.col("hour") >= 22) | (F.col("hour") < 6)).cast(IntegerType())
    )

    # ── Call gap (seconds between consecutive calls per MSISDN) ──────────────
    w = Window.partitionBy("CALLINGMSISDN").orderBy("DATETIMEOFCALL")
    df = df.withColumn(
        "call_gap_seconds",
        (F.col("DATETIMEOFCALL").cast("long") -
         F.lag("DATETIMEOFCALL").over(w).cast("long"))
    ).fillna({"call_gap_seconds": 0})

    # ── Outgoing aggregations ─────────────────────────────────────────────────
    outgoing = df.groupBy("CALLINGMSISDN", "DATE").agg(
        F.sum("DURATION").alias("outgoing_duration_1day"),
        F.count("CALLEDMSISDN").alias("total_calls_1day"),
        F.countDistinct("CALLEDMSISDN").alias("unique_recipients_1day"),
        F.countDistinct("GLOBALCELLID").alias("unique_cell_ids_1day"),
        F.sum("is_short_call").alias("short_call_count_1day"),
        F.mean("is_night").alias("night_call_ratio_1day"),
        F.mean("call_gap_seconds").alias("avg_call_gap_seconds_1day"),
    ).withColumnRenamed("CALLINGMSISDN", "MSISDN")

    # ── Incoming aggregations ─────────────────────────────────────────────────
    incoming = df.groupBy("CALLEDMSISDN", "DATE").agg(
        F.sum("DURATION").alias("incoming_duration_1day"),
        F.countDistinct("IMEI").alias("imei_count_1day"),
    ).withColumnRenamed("CALLEDMSISDN", "MSISDN")

    # ── Burst ratio: max calls in any 1-hour bucket ───────────────────────────
    hourly = (
        df.groupBy("CALLINGMSISDN", "DATE", "hour")
        .count()
        .groupBy("CALLINGMSISDN", "DATE")
        .agg(F.max("count").alias("max_hourly_calls"))
        .withColumnRenamed("CALLINGMSISDN", "MSISDN")
    )

    # ── Merge ─────────────────────────────────────────────────────────────────
    features = (
        outgoing
        .join(incoming, on=["MSISDN", "DATE"], how="outer")
        .join(hourly,   on=["MSISDN", "DATE"], how="left")
        .fillna(0)
    )

    total_dur = F.col("outgoing_duration_1day") + F.col("incoming_duration_1day")

    features = (
        features
        .withColumn("outgoing_pct_1day",
            F.when(total_dur > 0, F.col("outgoing_duration_1day") / total_dur * 100).otherwise(0))
        .withColumn("short_call_ratio_1day",
            F.when(F.col("total_calls_1day") > 0,
                   F.col("short_call_count_1day") / F.col("total_calls_1day")).otherwise(0))
        .withColumn("call_burst_ratio_1day",
            F.when(F.col("total_calls_1day") > 0,
                   F.col("max_hourly_calls") / F.col("total_calls_1day")).otherwise(0))
        .withColumn("call_to_unique_ratio",
            F.col("total_calls_1day") / (F.col("unique_recipients_1day") + 1))
        .withColumn("duration_per_call",
            F.when(F.col("total_calls_1day") > 0, total_dur / F.col("total_calls_1day")).otherwise(0))
        .withColumn("cell_id_density",
            F.col("unique_cell_ids_1day") / (F.col("total_calls_1day") + 1))
        .drop("max_hourly_calls")
    )

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="S3 path to raw CDR CSVs")
    parser.add_argument("--output", required=True, help="S3 path for feature parquet output")
    parser.add_argument("--date",   required=True, help="Processing date YYYY-MM-DD")
    args = parser.parse_args()

    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[FMCC] Loading CDRs from {args.input} for date {args.date}")
    df = load_cdrs(spark, args.input, args.date)
    record_count = df.count()
    print(f"[FMCC] Loaded {record_count:,} CDR records")

    print("[FMCC] Building 1-day features...")
    features = build_features(df)
    msisdn_count = features.count()
    print(f"[FMCC] Feature matrix: {msisdn_count:,} unique MSISDNs")

    output_path = f"{args.output}features.parquet"
    print(f"[FMCC] Writing features to {output_path}")
    features.write.mode("overwrite").parquet(output_path)

    print(f"[FMCC] Done. {msisdn_count:,} MSISDNs processed from {record_count:,} CDRs.")
    spark.stop()


if __name__ == "__main__":
    main()
