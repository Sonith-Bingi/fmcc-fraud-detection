"""
Model loading, caching, and inference.
Looks for the artifact at MODEL_PATH env var, falls back to models/latest.pkl.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from app.features import FEATURE_COLS

logger = logging.getLogger(__name__)

_MODEL_CACHE: Optional[object] = None
_MODEL_VERSION: str = "unknown"
_THRESHOLD: float = 0.5


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)
    # Look for the most recent pkl in models/
    models_dir = Path(__file__).parent.parent / "models"
    pkls = sorted(models_dir.glob("*.pkl"), reverse=True)
    if pkls:
        return pkls[0]
    raise FileNotFoundError(
        "No model artifact found. Set MODEL_PATH or place a .pkl in models/."
    )


def load_model() -> None:
    global _MODEL_CACHE, _MODEL_VERSION, _THRESHOLD

    path = _resolve_model_path()
    artifact = joblib.load(path)

    # Artifact can be a plain model or a dict with metadata
    if isinstance(artifact, dict):
        _MODEL_CACHE = artifact["model"]
        _MODEL_VERSION = artifact.get("version", path.stem)
        _THRESHOLD = artifact.get("threshold", 0.5)
    else:
        _MODEL_CACHE = artifact
        _MODEL_VERSION = path.stem
        _THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

    logger.info("Loaded model %s from %s (threshold=%.2f)", _MODEL_VERSION, path, _THRESHOLD)


def get_model():
    if _MODEL_CACHE is None:
        load_model()
    return _MODEL_CACHE


def predict(features_df: pd.DataFrame) -> np.ndarray:
    model = get_model()
    X = features_df[FEATURE_COLS].values
    return model.predict_proba(X)[:, 1]


def get_model_version() -> str:
    return _MODEL_VERSION


def get_threshold() -> float:
    return _THRESHOLD


def risk_tier(prob: float, threshold: float) -> str:
    if prob < threshold * 0.6:
        return "LOW"
    if prob < threshold:
        return "MEDIUM"
    return "HIGH"
