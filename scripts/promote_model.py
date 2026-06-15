"""
Promote the best fraud_detector model version to the 'Production' alias.
Best = highest avg_precision → version 1 (run 76e8fb11, avg_precision=1.0)
"""
import os
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
client = mlflow.MlflowClient()

model_name = "fraud_detector"

versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    raise ValueError(f"No versions found for model {model_name}")

latest_version = str(max(int(v.version) for v in versions))

# Set 'Production' alias on the latest version
client.set_registered_model_alias(model_name, "Production", latest_version)
print(f"Set alias 'Production' -> version {latest_version}")

# Confirm
mv = client.get_model_version_by_alias(model_name, "Production")
print(f"Production -> version={mv.version} run_id={mv.run_id[:8]} source={mv.source}")

# Print model URI
print(f"Model URI: models:/{model_name}@Production")
