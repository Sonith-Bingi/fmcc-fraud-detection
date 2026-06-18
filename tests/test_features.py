"""Tests for feature engineering — exact match to FMCC_1Day_Corrected.ipynb."""
import pandas as pd
import numpy as np
import pytest
from app.features import features_from_request_records, FEATURE_COLS


def make_record(**overrides):
    base = dict(
        msisdn=9876543210,
        date="2025-07-01",
        outgoing_duration=14000.0,
        avg_call_duration=161.0,
        total_calls=87,
        max_call_duration=600.0,
        unique_recipients=82,
        unique_cell_ids=2,
        imei_count=1,
        short_call_count=4,
        active_hours=8,
        received_duration=50.0,
        received_calls=0,
        high_activity_flag=1,
    )
    return {**base, **overrides}


def test_all_feature_columns_present():
    df = features_from_request_records([make_record()])
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_feature_count():
    assert len(FEATURE_COLS) == 21


def test_received_column_names_correct():
    """Notebook uses received_*_within_dataset, not incoming_*."""
    df = features_from_request_records([make_record()])
    assert "received_duration_within_dataset_1day" in df.columns
    assert "received_calls_within_dataset_1day" in df.columns
    assert "received_duration_percent_within_dataset" in df.columns
    assert "received_calls_percent_within_dataset" in df.columns
    assert "received_outgoing_interaction" in df.columns
    # These old names must NOT exist
    assert "incoming_duration_1day" not in df.columns
    assert "incoming_outgoing_interaction" not in df.columns


def test_recipient_uniqueness_ratio():
    rec = make_record(unique_recipients=80, total_calls=100)
    df = features_from_request_records([rec])
    expected = 80 / (100 + 1e-5)
    assert abs(df["recipient_uniqueness_ratio"].iloc[0] - expected) < 0.001


def test_short_call_percent():
    rec = make_record(short_call_count=30, total_calls=100)
    df = features_from_request_records([rec])
    expected = 30 / (100 + 1e-5) * 100
    assert abs(df["short_call_percent"].iloc[0] - expected) < 0.01


def test_received_duration_percent():
    rec = make_record(outgoing_duration=900, received_duration=300)
    df = features_from_request_records([rec])
    expected = 300 / 1200 * 100
    assert abs(df["received_duration_percent_within_dataset"].iloc[0] - expected) < 0.01


def test_received_calls_percent():
    rec = make_record(total_calls=80, received_calls=20)
    df = features_from_request_records([rec])
    expected = 20 / 100 * 100
    assert abs(df["received_calls_percent_within_dataset"].iloc[0] - expected) < 0.01


def test_zero_division_safety():
    rec = make_record(outgoing_duration=0, received_duration=0, total_calls=0,
                      unique_recipients=0, received_calls=0, active_hours=1)
    df = features_from_request_records([rec])
    assert not df[FEATURE_COLS].isnull().any().any()
    assert not np.isinf(df[FEATURE_COLS].values).any()


def test_received_outgoing_interaction():
    rec = make_record(outgoing_duration=1000, received_duration=200)
    df = features_from_request_records([rec])
    assert df["received_outgoing_interaction"].iloc[0] == 1000 * 200


def test_call_to_unique_ratio():
    rec = make_record(total_calls=90, unique_recipients=45)
    df = features_from_request_records([rec])
    expected = 90 / (45 + 1)
    assert abs(df["call_to_unique_ratio"].iloc[0] - expected) < 0.001


def test_batch_records():
    records = [make_record(msisdn=i) for i in range(50)]
    df = features_from_request_records(records)
    assert len(df) == 50
    assert df[FEATURE_COLS].notna().all().all()


def test_sim_box_profile():
    """Classic SIM-box: near-100% outgoing, many unique recipients, single cell."""
    rec = make_record(
        outgoing_duration=14638,
        received_duration=0,
        total_calls=86,
        unique_recipients=82,
        unique_cell_ids=1,
        short_call_count=3,
        received_calls=0,
    )
    df = features_from_request_records([rec])
    assert df["received_duration_percent_within_dataset"].iloc[0] == 0.0
    assert df["recipient_uniqueness_ratio"].iloc[0] > 0.9
    assert df["cell_diversity_1day"].iloc[0] == 1
