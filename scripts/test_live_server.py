"""
In-process live integration test for the FastAPI serving layer.
Uses FastAPI TestClient (httpx transport) with the REAL:
  - MLflow registry (sqlite:///mlruns.db)
  - Production model loaded via mlflow.pyfunc
  - Preprocessor downloaded from run artifacts
  - Temporary SQLite predictions DB
  - Real feature row from data/raw/creditcard.csv

This is the definitive Phase-2 end-to-end validation.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point to temp predictions DB so we don't corrupt the real one
with tempfile.TemporaryDirectory() as tmp:
    os.environ["PREDICTIONS_DB_PATH"] = str(Path(tmp) / "test_predictions.db")

    # Import the real app (triggers mlflow connection at import, not startup)
    from fastapi.testclient import TestClient
    import src.serving.app as app_module

    print("Starting TestClient with real MLflow + real model …")
    with TestClient(app_module.app, raise_server_exceptions=True) as client:
        # ── 1. GET /health ───────────────────────────────────────────────
        print("\n=== GET /health ===")
        resp = client.get("/health")
        print(f"Status: {resp.status_code}")
        h = resp.json()
        print(json.dumps(h, indent=2))
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert h["status"] == "ok"
        assert h["model_version"] == "1", f"Expected version '1', got {h['model_version']}"
        assert h["model_name"] == "fraud_detector"
        print("PASS: /health returned 200 with model_version='1'")

        # ── 2. POST /predict with real CSV row ───────────────────────────
        import pandas as pd
        csv_path = Path("data/raw/creditcard.csv")
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        FEAT = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        row = df[FEAT].iloc[0].to_dict()
        true_class = int(df["Class"].iloc[0])
        print(f"\nUsing row 0: Time={row['Time']}, Amount={row['Amount']}, true_class={true_class}")

        print("\n=== POST /predict (real row from CSV) ===")
        resp = client.post("/predict", json=row)
        print(f"Status: {resp.status_code}")
        p = resp.json()
        print(json.dumps(p, indent=2))
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}\n{resp.text}"
        assert "is_fraud" in p
        assert "fraud_probability" in p
        assert 0.0 <= p["fraud_probability"] <= 1.0
        assert p["model_version"] == "1", f"Expected version '1', got {p['model_version']}"
        assert p["model_name"] == "fraud_detector"
        assert isinstance(p["prediction_id"], int) and p["prediction_id"] >= 1
        print(f"PASS: /predict returned 200 with model_version='1', prediction_id={p['prediction_id']}")

        # ── 3. GET /health again — confirm total_predictions incremented ──
        resp2 = client.get("/health")
        h2 = resp2.json()
        print(f"\n/health after predict: total_predictions={h2['total_predictions']}")
        assert h2["total_predictions"] >= 1

        # ── 4. GET /drift-log → empty list (Phase 2) ─────────────────────
        resp3 = client.get("/drift-log")
        assert resp3.status_code == 200
        assert resp3.json() == []
        print("\nPASS: /drift-log returned [] (Phase 2 expected)")

        # ── 5. Validate 422 on bad input ─────────────────────────────────
        bad_row = dict(row)
        bad_row["Amount"] = -5.0
        resp4 = client.post("/predict", json=bad_row)
        assert resp4.status_code == 422
        print("PASS: Amount=-5 returned 422")

print("\n✅ ALL LIVE IN-PROCESS CHECKS PASSED")
