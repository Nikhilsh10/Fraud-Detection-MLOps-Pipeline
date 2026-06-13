"""
Promote the best fraud_detector model version to the 'Production' alias.
Best = highest avg_precision → version 1 (run 76e8fb11, avg_precision=1.0)
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = mlflow.MlflowClient()

# Set 'Production' alias on version 1
client.set_registered_model_alias("fraud_detector", "Production", "1")
print("Set alias 'Production' -> version 1")

# Confirm
mv = client.get_model_version_by_alias("fraud_detector", "Production")
print(f"Production -> version={mv.version} run_id={mv.run_id[:8]} source={mv.source}")

# Print model URI
print(f"Model URI: models:/fraud_detector@Production")
