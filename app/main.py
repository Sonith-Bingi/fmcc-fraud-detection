"""
FMCC Fraud Detection API
FastAPI application serving the 1-day window Voting Ensemble model.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import model as model_module
from app.features import FEATURE_COLS, features_from_request_records
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model...")
    try:
        model_module.load_model()
        logger.info("Model ready: %s", model_module.get_model_version())
    except FileNotFoundError as e:
        logger.warning("Model not found at startup: %s", e)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="FMCC Fraud Detection API",
    description="Real-time telecom fraud scoring using 1-day CDR features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info("%s %s → %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    try:
        m = model_module.get_model()
        loaded = m is not None
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=model_module.get_model_version(),
        feature_count=len(FEATURE_COLS),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest):
    if not request.records:
        raise HTTPException(status_code=422, detail="records list is empty")

    records_dicts = [r.model_dump() for r in request.records]
    features_df = features_from_request_records(records_dicts)

    try:
        probas = model_module.predict(features_df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    threshold = model_module.get_threshold()
    results = [
        PredictionResult(
            msisdn=rec["msisdn"],
            date=rec["date"],
            fraud_probability=round(float(p), 4),
            is_fraud=bool(p >= threshold),
            risk_tier=model_module.risk_tier(float(p), threshold),
        )
        for rec, p in zip(records_dicts, probas)
    ]

    return PredictionResponse(
        results=results,
        model_version=model_module.get_model_version(),
        threshold_used=threshold,
    )


@app.get("/", tags=["ops"])
def root():
    return {"service": "FMCC Fraud Detection API", "docs": "/docs"}
