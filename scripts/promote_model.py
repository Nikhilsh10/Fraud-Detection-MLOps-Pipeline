"""
Promote the best fraud_detector model version to the 'Production' alias.
Best = highest avg_precision → version 1 (run 76e8fb11, avg_precision=1.0)
"""
import os
import json
import subprocess
import mlflow
import boto3

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

# Get the run info
mv = client.get_model_version_by_alias(model_name, "Production")
run = client.get_run(mv.run_id)
local_artifact_uri = run.info.artifact_uri.replace("file://", "")

# Sync the whole artifact directory to S3
s3_bucket = "nikhilsh10-fraud-mlflow-artifacts"
s3_prefix = f"experiments/runs/{mv.run_id}/artifacts"
s3_artifact_uri = f"s3://{s3_bucket}/{s3_prefix}"

print(f"Uploading artifacts from {local_artifact_uri} to {s3_artifact_uri}...")
subprocess.run(["aws", "s3", "sync", local_artifact_uri, s3_artifact_uri], check=True)

# Use MLflow to resolve the model URI to a local path (handles models:/ URIs automatically)
import mlflow.artifacts
print(f"Resolving model artifacts from {mv.source} ...")
local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=mv.source)
s3_model_uri = f"{s3_artifact_uri}/model"
print(f"Uploading model from {local_model_path} to {s3_model_uri}...")
subprocess.run(["aws", "s3", "sync", local_model_path, s3_model_uri], check=True)

# Write pointer file
pointer_data = {
    "run_id": mv.run_id,
    "version": mv.version,
    "s3_model_uri": s3_model_uri,
    "s3_preprocessor_uri": f"{s3_artifact_uri}/preprocessor/preprocessor.pkl"
}

s3 = boto3.client("s3")
s3.put_object(
    Bucket=s3_bucket,
    Key="production_model.json",
    Body=json.dumps(pointer_data)
)
print("Uploaded production_model.json to S3.")
