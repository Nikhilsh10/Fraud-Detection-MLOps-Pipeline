"""
src/serving/schemas.py
-----------------------
Pydantic v2 request / response models for the Fraud Detection API.

The 30-feature layout mirrors the Kaggle Credit Card Fraud dataset:
    Time   — seconds elapsed since first transaction in the dataset
    V1-V28 — PCA-transformed anonymised features
    Amount — transaction amount in USD

Pydantic aliases are used so callers can POST plain JSON with keys
like "V1", "V28" etc. without needing Python-safe variable names.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """
    A single credit-card transaction with all 30 model features.
    All V-features are already PCA-transformed by the dataset provider.
    """

    # Timing / amount
    Time: float = Field(..., description="Seconds elapsed since first transaction (≥0)")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in USD (≥0)")

    # PCA components V1-V28
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @model_validator(mode="after")
    def check_time_non_negative(self) -> "TransactionRequest":
        if self.Time < 0:
            raise ValueError("Time must be ≥ 0")
        return self

    def to_feature_dict(self) -> dict[str, float]:
        """Return features in the order expected by the preprocessor."""
        return self.model_dump()

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 406.0,
            "Amount": 149.62,
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530534538215,
        }
    }}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Fraud prediction result for a single transaction."""
    model_config = {"protected_namespaces": ()}

    is_fraud: bool = Field(..., description="True if the model predicts fraud")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted probability that the transaction is fraudulent"
    )
    model_name: str = Field(..., description="MLflow registered model name")
    model_version: str = Field(..., description="MLflow model version number")
    prediction_id: int = Field(..., description="Row ID in the predictions log")


class HealthResponse(BaseModel):
    """API health / readiness information."""
    model_config = {"protected_namespaces": ()}

    status: str = Field(..., description="'ok' when the service is ready")
    model_name: str
    model_version: str
    uptime_seconds: float = Field(..., description="Seconds since the server started")
    total_predictions: int = Field(..., description="Cumulative predictions served")
    fraud_predictions: int = Field(..., description="Cumulative fraud predictions")


class DriftLogEntry(BaseModel):
    """One entry from the drift log produced by Phase-4 monitoring."""
    timestamp: str
    drift_share: float
    drifted_features: list[str]
    retrain_triggered: bool
