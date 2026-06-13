"""Check how mlflow pyfunc exposes the sklearn model internals."""
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias("fraud_detector", "Production")
run_id = mv.run_id
print(f"run_id={run_id[:8]} version={mv.version}")

# Load model
model_uri = "models:/fraud_detector@Production"
m = mlflow.pyfunc.load_model(model_uri)
print(f"type(m._model_impl) = {type(m._model_impl)}")
print(f"dir(m._model_impl) = {[x for x in dir(m._model_impl) if not x.startswith('__')]}")

# Check attributes
impl = m._model_impl
for attr in ['sklearn_model', 'python_model', 'predict_proba']:
    print(f"  hasattr({attr}) = {hasattr(impl, attr)}")

# Load preprocessor
prep_path = Path(mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="preprocessor/preprocessor.pkl",
    tracking_uri="sqlite:///mlruns.db",
))
with open(prep_path, "rb") as f:
    preprocessor = pickle.load(f)

# Build a dummy row
FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
row = {col: 0.0 for col in FEATURE_COLUMNS}
row["Time"] = 406.0
row["Amount"] = 149.62
df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
X = preprocessor.transform(df)
print(f"X shape: {X.shape}")

# Try predict_proba via sklearn_model
try:
    sk = impl.sklearn_model
    print(f"sklearn_model type: {type(sk)}")
    probs = sk.predict_proba(X)
    print(f"predict_proba result: {probs}")
except Exception as e:
    print(f"sklearn_model failed: {e}")

# Try direct predict
result = m.predict(pd.DataFrame(X, columns=FEATURE_COLUMNS))
print(f"pyfunc.predict result: {result}")
print(f"type: {type(result)}")
