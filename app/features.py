"""
1-day feature engineering — exact replication of FMCC_1Day_Corrected.ipynb.

Pipeline:
  1. Call stitching  (Cell 13): merge consecutive same-pair calls
  2. Aggregation     (Cell 37): per-MSISDN-DATE outgoing + received
  3. Derived ratios  (Cell 37): recipient_uniqueness_ratio, short_call_percent, etc.
  4. Advanced feats  (Cell 48): call_to_unique_ratio, duration_to_calls_ratio, etc.

Column naming follows the corrected notebook exactly:
  - received_duration_within_dataset_1day  (NOT incoming_duration)
  - received_calls_within_dataset_1day     (NOT incoming_calls)
  - received_duration_percent_within_dataset
  - received_calls_percent_within_dataset
  - received_outgoing_interaction          (NOT incoming_outgoing_interaction)
"""
import pandas as pd
import numpy as np

# Exact feature columns matching FMCC_1Day_Corrected.ipynb (Cells 37 + 48)
FEATURE_COLS = [
    # --- Cell 37 outgoing base features ---
    "outgoing_duration_1day",
    "avg_call_duration_1day",
    "call_count_1day",
    "max_call_duration_1day",
    "unique_recipients_1day",
    "cell_diversity_1day",
    "device_diversity_1day",
    "short_calls_1day",
    "active_hours_1day",
    # --- Cell 37 derived ratios ---
    "recipient_uniqueness_ratio",
    "short_call_percent",
    "calls_per_active_hour",
    # --- Cell 37 received (incoming within dataset) ---
    "received_duration_within_dataset_1day",
    "received_calls_within_dataset_1day",
    "received_duration_percent_within_dataset",
    "received_calls_percent_within_dataset",
    # --- Cell 48 interaction features ---
    "call_to_unique_ratio",
    "duration_to_calls_ratio",
    "cell_diversity",
    "received_outgoing_interaction",
    "high_activity_flag",
]


def stitch_calls(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cell 13: merge consecutive calls between the same MSISDN pair.
    Groups back-to-back calls (where next call starts exactly when previous ends)
    into a single stitched record with summed DURATION.
    """
    df = data.copy()
    df["DATETIMEOFCALL"] = pd.to_datetime(df["DATETIMEOFCALL"], format="mixed")
    df["DATE"] = df["DATETIMEOFCALL"].dt.date

    df = df.sort_values(["CALLINGMSISDN", "CALLEDMSISDN", "DATETIMEOFCALL"])

    df["expected_next_start"] = df["DATETIMEOFCALL"] + pd.to_timedelta(df["DURATION"], unit="s")
    df["prev_expected_end"] = df.groupby(["CALLINGMSISDN", "CALLEDMSISDN"])["expected_next_start"].shift()
    df["new_group"] = (df["DATETIMEOFCALL"] != df["prev_expected_end"]).astype(int)
    df["group_id"] = df.groupby(["CALLINGMSISDN", "CALLEDMSISDN", "DATE"])["new_group"].cumsum()

    merged = (
        df.groupby(["CALLINGMSISDN", "CALLEDMSISDN", "DATE", "group_id"])
        .agg(
            DATETIMEOFCALL=("DATETIMEOFCALL", "first"),
            DURATION=("DURATION", "sum"),
            GLOBALCELLID=("GLOBALCELLID", "first"),
            IMEI=("IMEI", "first"),
            CALLINGPARTYIMSI=("CALLINGPARTYIMSI", "first"),
        )
        .reset_index()
    )
    return merged


def build_1day_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline matching FMCC_1Day_Corrected.ipynb end-to-end.
    Input:  raw CDR DataFrame.
    Output: per-(MSISDN, DATE) feature DataFrame with all 21 FEATURE_COLS.
    """
    # Step 1: Call stitching (Cell 13)
    df = stitch_calls(data)
    df["is_short_call"] = (df["DURATION"] <= 15).astype(int)
    df["hour_of_day"] = df["DATETIMEOFCALL"].dt.hour

    # Step 2: Outgoing aggregations per MSISDN-DATE (Cell 37)
    outgoing = (
        df.groupby(["CALLINGMSISDN", "DATE"])
        .agg(
            outgoing_duration_1day=("DURATION", "sum"),
            avg_call_duration_1day=("DURATION", "mean"),
            call_count_1day=("DURATION", "count"),
            max_call_duration_1day=("DURATION", "max"),
            unique_recipients_1day=("CALLEDMSISDN", "nunique"),
            cell_diversity_1day=("GLOBALCELLID", "nunique"),
            device_diversity_1day=("IMEI", "nunique"),
            short_calls_1day=("is_short_call", "sum"),
            active_hours_1day=("hour_of_day", "nunique"),
        )
        .reset_index()
        .rename(columns={"CALLINGMSISDN": "MSISDN"})
    )

    # Step 3: Derived ratios (Cell 37)
    outgoing["recipient_uniqueness_ratio"] = (
        outgoing["unique_recipients_1day"] / (outgoing["call_count_1day"] + 1e-5)
    )
    outgoing["short_call_percent"] = (
        outgoing["short_calls_1day"] / (outgoing["call_count_1day"] + 1e-5)
    ) * 100
    outgoing["calls_per_active_hour"] = (
        outgoing["call_count_1day"] / (outgoing["active_hours_1day"] + 1e-5)
    )

    # Step 4: Received calls within dataset (Cell 37) — named exactly as notebook
    received = (
        df.groupby(["CALLEDMSISDN", "DATE"])["DURATION"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={
            "CALLEDMSISDN": "MSISDN",
            "sum": "received_duration_within_dataset_1day",
            "count": "received_calls_within_dataset_1day",
        })
    )

    features = outgoing.merge(received, on=["MSISDN", "DATE"], how="left").fillna(0)

    # Step 5: Received vs outgoing imbalance ratios (Cell 37)
    total_dur = features["outgoing_duration_1day"] + features["received_duration_within_dataset_1day"]
    total_calls = features["call_count_1day"] + features["received_calls_within_dataset_1day"]

    features["received_duration_percent_within_dataset"] = np.where(
        total_dur > 0,
        (features["received_duration_within_dataset_1day"] / total_dur) * 100,
        0,
    )
    features["received_calls_percent_within_dataset"] = np.where(
        total_calls > 0,
        (features["received_calls_within_dataset_1day"] / total_calls) * 100,
        0,
    )

    # Step 6: Advanced interaction features (Cell 48)
    features["call_to_unique_ratio"] = (
        features["call_count_1day"] / (features["unique_recipients_1day"] + 1)
    )
    features["duration_to_calls_ratio"] = (
        (features["outgoing_duration_1day"] + features["received_duration_within_dataset_1day"])
        / (features["call_count_1day"] + 1)
    )
    features["cell_diversity"] = (
        features["cell_diversity_1day"] / (features["call_count_1day"] + 1)
    )
    features["received_outgoing_interaction"] = (
        features["received_duration_within_dataset_1day"] * features["outgoing_duration_1day"]
    )
    features["high_activity_flag"] = (
        (features["call_count_1day"] > features["call_count_1day"].quantile(0.75)) &
        (features["outgoing_duration_1day"] > features["outgoing_duration_1day"].quantile(0.75))
    ).astype(int)

    return features[["MSISDN", "DATE"] + FEATURE_COLS]


def features_from_request_records(records: list[dict]) -> pd.DataFrame:
    """Convert pre-aggregated API request records into the feature matrix."""
    rows = []
    for r in records:
        og_dur = r["outgoing_duration"]
        rec_dur = r.get("received_duration", 0)
        og_calls = r["total_calls"]
        rec_calls = r.get("received_calls", 0)
        total_dur = og_dur + rec_dur
        total_calls_all = og_calls + rec_calls

        rows.append({
            "MSISDN": r["msisdn"],
            "DATE": r["date"],
            "outgoing_duration_1day":                  og_dur,
            "avg_call_duration_1day":                  r.get("avg_call_duration", og_dur / og_calls if og_calls else 0),
            "call_count_1day":                         og_calls,
            "max_call_duration_1day":                  r.get("max_call_duration", 0),
            "unique_recipients_1day":                  r["unique_recipients"],
            "cell_diversity_1day":                     r["unique_cell_ids"],
            "device_diversity_1day":                   r["imei_count"],
            "short_calls_1day":                        r["short_call_count"],
            "active_hours_1day":                       r.get("active_hours", 1),
            "recipient_uniqueness_ratio":               r["unique_recipients"] / (og_calls + 1e-5),
            "short_call_percent":                      r["short_call_count"] / (og_calls + 1e-5) * 100,
            "calls_per_active_hour":                   og_calls / (r.get("active_hours", 1) + 1e-5),
            "received_duration_within_dataset_1day":   rec_dur,
            "received_calls_within_dataset_1day":      rec_calls,
            "received_duration_percent_within_dataset": (rec_dur / total_dur * 100) if total_dur > 0 else 0,
            "received_calls_percent_within_dataset":   (rec_calls / total_calls_all * 100) if total_calls_all > 0 else 0,
            "call_to_unique_ratio":                    og_calls / (r["unique_recipients"] + 1),
            "duration_to_calls_ratio":                 total_dur / (og_calls + 1),
            "cell_diversity":                          r["unique_cell_ids"] / (og_calls + 1),
            "received_outgoing_interaction":           rec_dur * og_dur,
            "high_activity_flag":                      r.get("high_activity_flag", 0),
        })
    return pd.DataFrame(rows)
