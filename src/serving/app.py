"""
src/serving/app.py
-------------------
FastAPI application for the Fraud Detection MLOps Pipeline.

Endpoints
---------
POST /predict     Validate → preprocess → infer → log to SQLite → return result
GET  /health      Liveness + model info + cumulative prediction counts
GET  /drift-log   Returns parsed drift_log.jsonl entries (Phase 4); empty list
                  if the file does not yet exist (Phase 2 compatible)

Startup behaviour
-----------------
1. Loads the 'Production'-aliased fraud_detector model from the local MLflow
   registry (sqlite:///mlruns.db).
2. Downloads the preprocessor.pkl artifact stored in the same MLflow run.
3. Initialises predictions.db (SQLite).

Inference notes
---------------
The model was fitted on numpy arrays (no pandas feature names), so the
ColumnTransformer output (numpy ndarray) is passed directly to predict_proba.
The mlflow pyfunc wraps sklearn models in _SklearnModelWrapper which exposes:
  - .sklearn_model  →  the raw RandomForestClassifier
  - .predict_proba  →  convenience proxy

Environment variables
---------------------
MLFLOW_TRACKING_URI   default: sqlite:///mlruns.db
PREDICTIONS_DB_PATH   default: predictions.db
DRIFT_LOG_PATH        default: reports/drift_log.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.serving.db import count_predictions, init_db, insert_prediction
from src.serving.schemas import (
    DriftLogEntry,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGISTRY_MODEL_NAME = "fraud_detector"
DRIFT_LOG_PATH = Path(os.getenv("DRIFT_LOG_PATH", "reports/drift_log.jsonl"))

# Feature column order — must match ColumnTransformer in engineer.py
# (Time + Amount are scaled; V1-V28 are passed through unchanged)
FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


# ---------------------------------------------------------------------------
# Application state (populated at startup, never mutated after)
# ---------------------------------------------------------------------------

class _ModelState:
    pyfunc_model: mlflow.pyfunc.PyFuncModel | None = None
    preprocessor: Any = None          # fitted sklearn ColumnTransformer
    model_version: str = "unknown"
    model_name: str = REGISTRY_MODEL_NAME
    start_time: float = 0.0


_state = _ModelState()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + preprocessor once at startup; nothing to teardown."""
    _state.start_time = time.monotonic()

    logger.info("Fetching production model pointer from S3...")
    
    # Download pointer file from S3
    import boto3
    s3 = boto3.client("s3")
    s3_bucket = "nikhilsh10-fraud-mlflow-artifacts"
    try:
        response = s3.get_object(Bucket=s3_bucket, Key="production_model.json")
        pointer = json.loads(response["Body"].read().decode('utf-8'))
    except Exception as e:
        logger.error("Failed to fetch production_model.json from S3: %s", e)
        raise

    _state.model_version = str(pointer.get("version", "unknown"))
    s3_model_uri = pointer["s3_model_uri"]
    s3_preprocessor_uri = pointer["s3_preprocessor_uri"]

    logger.info(
        "Production model pointer fetched → version=%s  run_id=%s", 
        _state.model_version, pointer.get("run_id", "unknown")[:8]
    )

    # Load pyfunc model directly from S3 URI
    logger.info("Loading model from %s", s3_model_uri)
    _state.pyfunc_model = mlflow.pyfunc.load_model(s3_model_uri)
    logger.info("Model loaded successfully.")

    # Download preprocessor.pkl from S3 URI via mlflow artifact downloader
    logger.info("Downloading preprocessor from %s", s3_preprocessor_uri)
    local_prep_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=s3_preprocessor_uri,
        )
    )
    with open(local_prep_path, "rb") as fh:
        _state.preprocessor = pickle.load(fh)
    logger.info("Preprocessor loaded from %s", local_prep_path)

    # Initialise predictions database
    db_path = init_db()
    logger.info("Predictions DB initialised at %s", db_path)

    logger.info("✅ Startup complete — API is ready.")
    yield
    # No teardown needed


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "MLOps-grade fraud scoring service backed by the Production model "
        "from the local MLflow registry. Scores transactions in real time "
        "and logs every prediction to a SQLite database."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(req: TransactionRequest) -> np.ndarray:
    """
    Convert a TransactionRequest → ordered DataFrame → preprocessed ndarray.
    The ColumnTransformer scales Time & Amount; V1-V28 pass through unchanged.
    We pass a numpy array to avoid the sklearn feature-names warning (the RF
    was fitted on numpy arrays, not DataFrames).
    """
    row = req.to_feature_dict()
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return _state.preprocessor.transform(df)   # shape (1, 30)


def _get_fraud_probability(X: np.ndarray) -> float:
    """
    Extract class-1 (fraud) probability from the mlflow pyfunc model.

    The loaded model is an instance of mlflow.sklearn._SklearnModelWrapper
    which exposes:
      - .sklearn_model  → the raw RandomForestClassifier
      - .predict_proba  → proxies sklearn_model.predict_proba

    The RF was fitted on raw numpy arrays, so we pass X (ndarray) directly.
    """
    wrapper = _state.pyfunc_model._model_impl   # _SklearnModelWrapper
    probs = wrapper.predict_proba(X)             # shape (1, 2)
    return float(probs[0, 1])                   # class-1 probability


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction for fraud",
    tags=["inference"],
)
def predict(req: TransactionRequest) -> PredictionResponse:
    """
    Accept a 30-feature transaction JSON body, run inference through the
    Production fraud model, persist the result to SQLite, and return the
    fraud probability + binary label.

    **Threshold**: 0.5 (fraud if `fraud_probability >= 0.5`).
    """
    if _state.pyfunc_model is None or _state.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet — please retry in a moment.",
        )

    try:
        X = _preprocess(req)
        proba = _get_fraud_probability(X)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    is_fraud = proba >= 0.5
    ts = datetime.now(timezone.utc).isoformat()

    prediction_id = insert_prediction(
        ts=ts,
        fraud_prob=proba,
        is_fraud=is_fraud,
        model_version=_state.model_version,
        amount=req.Amount,
        time_feature=req.Time,
    )

    logger.info(
        "Prediction #%d — prob=%.4f  is_fraud=%s  amount=%.2f",
        prediction_id, proba, is_fraud, req.Amount,
    )

    return PredictionResponse(
        is_fraud=bool(is_fraud),
        fraud_probability=round(proba, 6),
        model_name=_state.model_name,
        model_version=_state.model_version,
        prediction_id=prediction_id,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health / readiness check",
    tags=["ops"],
)
def health() -> HealthResponse:
    """Return API liveness, loaded model version, and cumulative stats."""
    if _state.pyfunc_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    total, fraud = count_predictions()
    return HealthResponse(
        status="ok",
        model_name=_state.model_name,
        model_version=_state.model_version,
        uptime_seconds=round(time.monotonic() - _state.start_time, 2),
        total_predictions=total,
        fraud_predictions=fraud,
    )


@app.get(
    "/drift-log",
    response_model=list[DriftLogEntry],
    status_code=status.HTTP_200_OK,
    summary="Drift detection history (Phase 4)",
    tags=["monitoring"],
)
def drift_log() -> list[DriftLogEntry]:
    """
    Return parsed drift log entries from `reports/drift_log.jsonl`.
    Returns an empty list if the file does not yet exist (Phase 2 safe).
    """
    if not DRIFT_LOG_PATH.exists():
        return []
    entries: list[DriftLogEntry] = []
    with open(DRIFT_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(DriftLogEntry(**json.loads(line)))
                except Exception:  # noqa: BLE001
                    pass
    return entries
