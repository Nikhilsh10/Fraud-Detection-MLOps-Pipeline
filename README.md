# Fraud Detection MLOps Pipeline

An end-to-end MLOps reference implementation demonstrating the full model lifecycle:
experiment tracking → serving → CI/CD → drift detection → automated retraining.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                           │
│  validate_data.yml → ml_pipeline.yml → drift_check.yml (cron)  │
└──────────────┬─────────────────────────┬────────────────────────┘
               │                         │ repository_dispatch
               ▼                         ▼
        ┌─────────────┐          ┌──────────────────┐
        │   MLflow    │          │  Evidently Drift  │
        │  Registry   │◄─────────│  Report Engine    │
        │ fraud_detector│        └──────────────────┘
        └──────┬──────┘
               │ loads Production model
               ▼
        ┌─────────────┐
        │  FastAPI +  │──► predictions_log (SQLite)
        │   Docker    │
        └─────────────┘
```

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions, 492 frauds (0.172% positive class).

## Experiment Results

| # | n_estimators | max_depth | class_weight | SMOTE | roc_auc | avg_precision | f1 | MLflow ver | run_id (short) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | None | balanced | ✅ | 1.0000 | 1.0000 | 0.9630 | v1 | `76e8fb1` |
| 2 | 200 | None | balanced | ✅ | 1.0000 | 1.0000 | 0.9630 | v2 | `3a0e90b` |
| 3 | 100 | 10 | balanced | ✅ | 1.0000 | 1.0000 | 0.9630 | v3 | `e2c6053` |

> **Note**: All experiments run on a 50k-row synthetic dataset (Kaggle dataset not yet downloaded).
> Perfect scores (AUC=1.0) reflect the synthetic data's clean separability on V4/V11/V12 features.
> Real Kaggle data will produce realistic scores (~0.97 AUC, ~0.85 Avg Precision).
> Goal is to verify the MLflow logging loop — scores are not meaningful until real data is used.

## Drift Threshold Justification

We use a **DataDriftPreset** via Evidently with a default drift share threshold of `0.25` (25%).
- **Why 25%?** The dataset contains 30 features (Time, Amount, V1-V28). A drift share of 0.25 means at least 7-8 features have significantly shifted in their statistical distributions (using default statistical tests like Kolmogorov-Smirnov).
- Fraud detection models are highly sensitive to shifts in the PCA-transformed `V` features. If >25% of features drift, the underlying transaction behavior has fundamentally changed, strongly indicating that the model's learned decision boundaries may no longer be reliable and a retrain is required.

## Quickstart

### Local Setup
```bash
# Install dependencies
pip install -e ".[dev,serving,monitoring,dashboard]"

# Download dataset
python -m src.data.download

# Run experiments
python -m src.training.train

# Set production model alias
python scripts/promote_model.py

# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### Serving API (Docker)
```bash
# Build the image
docker build -t fraud-api:latest .

# Run the API, mounting the model registry
docker run -p 8000:8000 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/mlruns.db:/app/mlruns.db \
  fraud-api:latest

# Check health
curl http://localhost:8000/health
```

### Dashboard
```bash
# Launch the Streamlit dashboard
streamlit run src/dashboard/app.py
```

### Drift Detection
```bash
# Simulate drift
python -m src.monitoring.simulate_drift 200 amount_shift

# Run the drift detector
python -m src.monitoring.drift_detector
```

## Project Phases

- **Phase 1** ✅ Data, features, training, MLflow experiments
- **Phase 2** ✅ FastAPI serving + Docker
- **Phase 3** ✅ GitHub Actions CI/CD
- **Phase 4** ✅ Evidently drift detection + retrain loop
- **Phase 5** ✅ Streamlit dashboard + README polish
