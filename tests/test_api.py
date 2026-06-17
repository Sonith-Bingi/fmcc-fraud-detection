"""API integration tests using TestClient (no real model needed)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

SAMPLE_RECORD = {
    "msisdn": 9876543210,
    "date": "2025-07-01",
    "outgoing_duration": 14000.0,
    "incoming_duration": 50.0,
    "unique_recipients": 90,
    "total_calls": 95,
    "unique_cell_ids": 1,
    "imei_count": 1,
    "short_call_count": 5,
    "night_call_ratio": 0.05,
    "call_burst_ratio": 0.6,
    "avg_call_gap_seconds": 30.0,
}

def _mock_predict_proba(X):
    n = len(X)
    return np.column_stack([np.full(n, 0.08), np.full(n, 0.92)])

MOCK_MODEL = MagicMock()
MOCK_MODEL.predict_proba.side_effect = _mock_predict_proba


@pytest.fixture(autouse=True)
def mock_model():
    with (
        patch("app.model._MODEL_CACHE", MOCK_MODEL),
        patch("app.model._MODEL_VERSION", "test-v1"),
        patch("app.model._THRESHOLD", 0.5),
    ):
        yield


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single_high_risk():
    r = client.post("/predict", json={"records": [SAMPLE_RECORD]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["msisdn"] == 9876543210
    assert result["is_fraud"] is True
    assert result["risk_tier"] == "HIGH"
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_batch():
    records = [{**SAMPLE_RECORD, "msisdn": 9000000000 + i} for i in range(10)]
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 10


def test_predict_empty_body():
    r = client.post("/predict", json={"records": []})
    assert r.status_code == 422


def test_predict_missing_field():
    bad = {k: v for k, v in SAMPLE_RECORD.items() if k != "total_calls"}
    r = client.post("/predict", json={"records": [bad]})
    # Should still succeed — total_calls has a default of 0
    assert r.status_code == 200


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "FMCC" in r.json()["service"]


def test_response_time_header():
    r = client.post("/predict", json={"records": [SAMPLE_RECORD]})
    assert "x-response-time-ms" in r.headers
