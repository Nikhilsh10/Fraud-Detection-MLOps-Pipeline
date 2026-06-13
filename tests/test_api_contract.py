"""
tests/test_api_contract.py
---------------------------
Contract tests for the Fraud Detection FastAPI serving layer.

Uses FastAPI TestClient (synchronous, no real server needed).

Test cases
----------
health_200          GET /health returns 200 + valid HealthResponse schema
predict_200_non_fraud POST /predict with a typical non-fraud-like row → 200
predict_200_fraud     POST /predict with a fraud-like row → 200 (prob may vary)
predict_422_missing   POST /predict with missing required field → 422
predict_422_bad_type  POST /predict with wrong type → 422
predict_422_negative_amount  POST /predict with Amount < 0 → 422
drift_log_200       GET /drift-log returns 200 + list (empty OK in Phase 2)

Isolation
---------
We patch mlflow.pyfunc.load_model and mlflow.artifacts.download_artifacts
so no real MLflow or SQLite is required. A temporary predictions.db is used.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def _make_preprocessor() -> ColumnTransformer:
    """Build and fit a real ColumnTransformer on synthetic data."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    pca_cols = [f"V{i}" for i in range(1, 29)]
    ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount"]),
            ("passthrough", "passthrough", pca_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    rng = np.random.default_rng(42)
    n = 200
    data = {
        "Time": rng.uniform(0, 172792, n),
        "Amount": rng.uniform(0, 500, n),
    }
    for col in pca_cols:
        data[col] = rng.standard_normal(n)
    df = pd.DataFrame(data)
    ct.fit(df)
    return ct


def _make_mock_pyfunc(fraud_prob: float = 0.05):
    """
    Create a mock mlflow.pyfunc.PyFuncModel whose _model_impl.predict_proba
    returns [[1-p, p]] for the given fraud probability.
    """
    mock_wrapper = MagicMock()
    mock_wrapper.predict_proba.return_value = np.array([[1 - fraud_prob, fraud_prob]])

    mock_model = MagicMock()
    mock_model._model_impl = mock_wrapper
    return mock_model


def _non_fraud_row() -> dict:
    """A realistic non-fraud transaction row (30 features)."""
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row.update({"Time": 406.0, "Amount": 149.62})
    return row


def _fraud_row() -> dict:
    """Row that the mock will score as fraud (prob injected via fixture)."""
    return _non_fraud_row()  # values don't matter — mock controls probability


@pytest.fixture()
def client(tmp_path):
    """
    TestClient with:
    - mlflow load_model → mock pyfunc (non-fraud by default)
    - mlflow download_artifacts → real temporary preprocessor.pkl
    - predictions.db → temp directory
    """
    # Write a real preprocessor to a temp file
    preprocessor = _make_preprocessor()
    prep_pkl = tmp_path / "preprocessor.pkl"
    with open(prep_pkl, "wb") as fh:
        pickle.dump(preprocessor, fh)

    # Mock MLflow client
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_mv.run_id = "76e8fb115ee74e4aa070637ba91480d8"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv

    env_overrides = {
        "PREDICTIONS_DB_PATH": str(tmp_path / "predictions.db"),
        "MLFLOW_TRACKING_URI": "sqlite:///mlruns.db",
    }

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.MlflowClient", return_value=mock_client),
        patch("mlflow.pyfunc.load_model", return_value=_make_mock_pyfunc(0.05)),
        patch(
            "mlflow.artifacts.download_artifacts",
            return_value=str(prep_pkl),
        ),
        patch.dict(os.environ, env_overrides),
    ):
        # Import app AFTER patches are active so lifespan picks them up
        import importlib
        import src.serving.app as app_module
        import src.serving.db as db_module

        # Re-patch DB path before importing TestClient
        importlib.reload(db_module)
        importlib.reload(app_module)

        with TestClient(app_module.app, raise_server_exceptions=True) as tc:
            yield tc


@pytest.fixture()
def fraud_client(tmp_path):
    """Same as client but mock returns high fraud probability."""
    preprocessor = _make_preprocessor()
    prep_pkl = tmp_path / "preprocessor.pkl"
    with open(prep_pkl, "wb") as fh:
        pickle.dump(preprocessor, fh)

    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_mv.run_id = "76e8fb115ee74e4aa070637ba91480d8"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv

    env_overrides = {"PREDICTIONS_DB_PATH": str(tmp_path / "predictions.db")}

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.MlflowClient", return_value=mock_client),
        patch("mlflow.pyfunc.load_model", return_value=_make_mock_pyfunc(0.92)),
        patch("mlflow.artifacts.download_artifacts", return_value=str(prep_pkl)),
        patch.dict(os.environ, env_overrides),
    ):
        import importlib
        import src.serving.app as app_module
        import src.serving.db as db_module

        importlib.reload(db_module)
        importlib.reload(app_module)

        with TestClient(app_module.app, raise_server_exceptions=True) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_200(self, client):
        """GET /health must return 200 with the expected schema."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_name"] == "fraud_detector"
        assert body["model_version"] == "1"
        assert isinstance(body["uptime_seconds"], float)
        assert body["uptime_seconds"] >= 0
        assert isinstance(body["total_predictions"], int)
        assert isinstance(body["fraud_predictions"], int)


class TestPredict:
    def test_predict_200_non_fraud(self, client):
        """POST /predict with a valid row and low-fraud mock → 200, is_fraud=False."""
        resp = client.post("/predict", json=_non_fraud_row())
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_fraud"] is False
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["model_name"] == "fraud_detector"
        assert body["model_version"] == "1"
        assert isinstance(body["prediction_id"], int)
        assert body["prediction_id"] >= 1

    def test_predict_200_fraud(self, fraud_client):
        """POST /predict with high-fraud mock → 200, is_fraud=True."""
        resp = fraud_client.post("/predict", json=_fraud_row())
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_fraud"] is True
        assert body["fraud_probability"] >= 0.5

    def test_predict_increments_prediction_id(self, client):
        """Two successive predictions should have consecutive IDs."""
        r1 = client.post("/predict", json=_non_fraud_row()).json()
        r2 = client.post("/predict", json=_non_fraud_row()).json()
        assert r2["prediction_id"] == r1["prediction_id"] + 1

    def test_predict_422_missing_field(self, client):
        """POST /predict with a missing required field → 422 Unprocessable Entity."""
        row = _non_fraud_row()
        del row["Amount"]
        resp = client.post("/predict", json=row)
        assert resp.status_code == 422

    def test_predict_422_wrong_type(self, client):
        """POST /predict with a string where float is expected → 422."""
        row = _non_fraud_row()
        row["V1"] = "not-a-number"
        resp = client.post("/predict", json=row)
        assert resp.status_code == 422

    def test_predict_422_negative_amount(self, client):
        """POST /predict with Amount < 0 → 422 (ge=0 constraint)."""
        row = _non_fraud_row()
        row["Amount"] = -1.0
        resp = client.post("/predict", json=row)
        assert resp.status_code == 422

    def test_predict_422_negative_time(self, client):
        """POST /predict with Time < 0 → 422 (model_validator)."""
        row = _non_fraud_row()
        row["Time"] = -100.0
        resp = client.post("/predict", json=row)
        assert resp.status_code == 422


class TestDriftLog:
    def test_drift_log_200_empty(self, client):
        """GET /drift-log returns 200 + empty list when file doesn't exist."""
        resp = client.get("/drift-log")
        assert resp.status_code == 200
        assert resp.json() == []
