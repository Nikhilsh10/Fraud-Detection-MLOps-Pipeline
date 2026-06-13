"""Inspect artifacts for the Production model run."""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = mlflow.MlflowClient()

mv = client.get_model_version_by_alias("fraud_detector", "Production")
run_id = mv.run_id
print(f"Run ID: {run_id}")

artifacts = client.list_artifacts(run_id)
for a in artifacts:
    print(f"  {a.path} (dir={a.is_dir})")
    if a.is_dir:
        sub = client.list_artifacts(run_id, a.path)
        for s in sub:
            print(f"    {s.path}")
