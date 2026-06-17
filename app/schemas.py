from pydantic import BaseModel, Field
from typing import Optional


class CDRRecord(BaseModel):
    """Single CDR record for real-time feature computation."""
    msisdn: int = Field(..., description="Subscriber number")
    date: str = Field(..., description="Date of activity (YYYY-MM-DD)")
    outgoing_duration: float = Field(0.0, ge=0)
    incoming_duration: float = Field(0.0, ge=0)
    unique_recipients: int = Field(0, ge=0)
    total_calls: int = Field(0, ge=0)
    unique_cell_ids: int = Field(0, ge=0)
    imei_count: int = Field(0, ge=0)
    short_call_count: int = Field(0, ge=0, description="Calls <= 15s")
    night_call_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of calls 22:00-06:00")
    call_burst_ratio: float = Field(0.0, ge=0.0, description="Max calls in any 1-hour window / total calls")
    avg_call_gap_seconds: float = Field(0.0, ge=0)


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
