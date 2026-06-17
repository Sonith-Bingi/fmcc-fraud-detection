"""
1-day feature engineering — mirrors the notebook logic.
Accepts raw CDR DataFrame, returns aggregated per-MSISDN-DATE feature DataFrame.
"""
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "outgoing_duration_1day",
    "incoming_duration_1day",
    "outgoing_pct_1day",
    "unique_recipients_1day",
    "total_calls_1day",
    "unique_cell_ids_1day",
    "imei_count_1day",
    "short_call_count_1day",
    "short_call_ratio_1day",
    "night_call_ratio_1day",
    "call_burst_ratio_1day",
    "avg_call_gap_seconds_1day",
    "call_to_unique_ratio",
    "duration_per_call",
    "cell_id_density",
]


def build_1day_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw CDR DataFrame with columns from the notebook.
    Output: per-(MSISDN, DATE) feature DataFrame ready for model input.
    """
    df = data.copy()
    df["DATETIMEOFCALL"] = pd.to_datetime(df["DATETIMEOFCALL"], format="mixed")
    df["DATE"] = df["DATETIMEOFCALL"].dt.date
    df["is_short_call"] = (df["DURATION"] <= 15).astype(int)
    df["hour"] = df["DATETIMEOFCALL"].dt.hour
    df["is_night"] = df["hour"].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    # Sort for gap calculation
    df = df.sort_values(["CALLINGMSISDN", "DATETIMEOFCALL"])
    df["call_gap_seconds"] = (
        df.groupby("CALLINGMSISDN")["DATETIMEOFCALL"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    # --- Outgoing aggregations ---
    out = df.groupby(["CALLINGMSISDN", "DATE"]).agg(
        outgoing_duration_1day=("DURATION", "sum"),
        total_calls_1day=("CALLEDMSISDN", "count"),
        unique_recipients_1day=("CALLEDMSISDN", "nunique"),
        unique_cell_ids_1day=("GLOBALCELLID", "nunique"),
        short_call_count_1day=("is_short_call", "sum"),
        night_call_ratio_1day=("is_night", "mean"),
        avg_call_gap_seconds_1day=("call_gap_seconds", "mean"),
    ).reset_index().rename(columns={"CALLINGMSISDN": "MSISDN"})

    # --- Incoming aggregations ---
    inc = df.groupby(["CALLEDMSISDN", "DATE"]).agg(
        incoming_duration_1day=("DURATION", "sum"),
        imei_count_1day=("IMEI", "nunique"),
    ).reset_index().rename(columns={"CALLEDMSISDN": "MSISDN"})

    # --- Burst ratio: max calls in any 1-hour bucket / total ---
    def burst_ratio(sub_df):
        if len(sub_df) == 0:
            return 0.0
        counts = sub_df.groupby(sub_df["DATETIMEOFCALL"].dt.hour).size()
        return counts.max() / len(sub_df)

    burst = (
        df.groupby(["CALLINGMSISDN", "DATE"])
        .apply(burst_ratio)
        .reset_index(name="call_burst_ratio_1day")
        .rename(columns={"CALLINGMSISDN": "MSISDN"})
    )

    # --- Merge ---
    features = out.merge(inc, on=["MSISDN", "DATE"], how="outer").fillna(0)
    features = features.merge(burst, on=["MSISDN", "DATE"], how="left").fillna(0)

    total_dur = features["outgoing_duration_1day"] + features["incoming_duration_1day"]
    features["outgoing_pct_1day"] = np.where(
        total_dur > 0, features["outgoing_duration_1day"] / total_dur * 100, 0
    )
    features["short_call_ratio_1day"] = np.where(
        features["total_calls_1day"] > 0,
        features["short_call_count_1day"] / features["total_calls_1day"],
        0,
    )
    features["call_to_unique_ratio"] = (
        features["total_calls_1day"] / (features["unique_recipients_1day"] + 1)
    )
    features["duration_per_call"] = np.where(
        features["total_calls_1day"] > 0,
        total_dur / features["total_calls_1day"],
        0,
    )
    features["cell_id_density"] = (
        features["unique_cell_ids_1day"] / (features["total_calls_1day"] + 1)
    )

    return features[["MSISDN", "DATE"] + FEATURE_COLS]


def features_from_request_records(records: list[dict]) -> pd.DataFrame:
    """Convert pre-aggregated API request records into a feature matrix."""
    rows = []
    for r in records:
        total_dur = r["outgoing_duration"] + r["incoming_duration"]
        rows.append({
            "MSISDN": r["msisdn"],
            "DATE": r["date"],
            "outgoing_duration_1day": r["outgoing_duration"],
            "incoming_duration_1day": r["incoming_duration"],
            "outgoing_pct_1day": (r["outgoing_duration"] / total_dur * 100) if total_dur > 0 else 0,
            "unique_recipients_1day": r["unique_recipients"],
            "total_calls_1day": r["total_calls"],
            "unique_cell_ids_1day": r["unique_cell_ids"],
            "imei_count_1day": r["imei_count"],
            "short_call_count_1day": r["short_call_count"],
            "short_call_ratio_1day": (r["short_call_count"] / r["total_calls"]) if r["total_calls"] > 0 else 0,
            "night_call_ratio_1day": r["night_call_ratio"],
            "call_burst_ratio_1day": r["call_burst_ratio"],
            "avg_call_gap_seconds_1day": r["avg_call_gap_seconds"],
            "call_to_unique_ratio": r["total_calls"] / (r["unique_recipients"] + 1),
            "duration_per_call": total_dur / r["total_calls"] if r["total_calls"] > 0 else 0,
            "cell_id_density": r["unique_cell_ids"] / (r["total_calls"] + 1),
        })
    return pd.DataFrame(rows)
