"""Tests for feature engineering — no model needed."""
import pandas as pd
import numpy as np
import pytest
from app.features import features_from_request_records, FEATURE_COLS


def make_record(**overrides):
    base = dict(
        msisdn=9876543210,
        date="2025-07-01",
        outgoing_duration=3600.0,
        incoming_duration=600.0,
        unique_recipients=80,
        total_calls=120,
        unique_cell_ids=2,
        imei_count=1,
        short_call_count=30,
        night_call_ratio=0.1,
        call_burst_ratio=0.4,
        avg_call_gap_seconds=45.0,
    )
    return {**base, **overrides}


def test_feature_columns_present():
    df = features_from_request_records([make_record()])
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_outgoing_pct_calculation():
    rec = make_record(outgoing_duration=900, incoming_duration=300)
    df = features_from_request_records([rec])
    expected = 900 / (900 + 300) * 100
    assert abs(df["outgoing_pct_1day"].iloc[0] - expected) < 0.01


def test_zero_division_safety():
    rec = make_record(outgoing_duration=0, incoming_duration=0, total_calls=0, unique_recipients=0)
    df = features_from_request_records([rec])
    assert not df[FEATURE_COLS].isnull().any().any()
    assert not np.isinf(df[FEATURE_COLS].values).any()


def test_short_call_ratio():
    rec = make_record(short_call_count=40, total_calls=100)
    df = features_from_request_records([rec])
    assert abs(df["short_call_ratio_1day"].iloc[0] - 0.4) < 0.001


def test_batch_records():
    records = [make_record(msisdn=i) for i in range(50)]
    df = features_from_request_records(records)
    assert len(df) == 50
    assert df[FEATURE_COLS].notna().all().all()


def test_high_risk_profile():
    """A SIM-box profile should show extreme outgoing dominance."""
    rec = make_record(
        outgoing_duration=14000,
        incoming_duration=50,
        unique_recipients=90,
        total_calls=95,
        unique_cell_ids=1,
        short_call_count=5,
    )
    df = features_from_request_records([rec])
    assert df["outgoing_pct_1day"].iloc[0] > 95
    assert df["call_to_unique_ratio"].iloc[0] > 1.0
