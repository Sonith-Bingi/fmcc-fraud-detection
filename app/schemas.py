from pydantic import BaseModel, Field
from typing import Optional


class CDRRecord(BaseModel):
    """
    Pre-aggregated 1-day stats for a single MSISDN.
    The caller computes these from raw CDRs (after call stitching).
    """
    msisdn: int = Field(..., description="Subscriber number")
    date: str = Field(..., description="Activity date (YYYY-MM-DD)")

    # Outgoing (Cell 37)
    outgoing_duration: float = Field(0.0, ge=0, description="Total outgoing call seconds")
    avg_call_duration: float = Field(0.0, ge=0, description="Mean outgoing call duration (s)")
    total_calls: int = Field(0, ge=0, description="Outgoing call count")
    max_call_duration: float = Field(0.0, ge=0, description="Longest single outgoing call (s)")
    unique_recipients: int = Field(0, ge=0, description="Distinct numbers called")
    unique_cell_ids: int = Field(0, ge=0, description="Distinct cell towers used")
    imei_count: int = Field(0, ge=0, description="Distinct IMEIs observed")
    short_call_count: int = Field(0, ge=0, description="Calls <= 15 seconds")
    active_hours: int = Field(1, ge=1, description="Number of distinct hours with activity")

    # Received within dataset (Cell 37) — calls where this MSISDN is the CALLEDMSISDN
    received_duration: float = Field(0.0, ge=0, description="Total received call seconds (within dataset)")
    received_calls: int = Field(0, ge=0, description="Received call count (within dataset)")

    # Cell 48 — high_activity_flag must be pre-computed by the caller
    # (requires population-level quantiles, so can't compute per-record)
    high_activity_flag: int = Field(0, ge=0, le=1)


class PredictionRequest(BaseModel):
    records: list[CDRRecord]


class PredictionResult(BaseModel):
    msisdn: int
    date: str
    fraud_probability: float
    is_fraud: bool
    risk_tier: str  # LOW / MEDIUM / HIGH


class PredictionResponse(BaseModel):
    results: list[PredictionResult]
    model_version: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    feature_count: int
