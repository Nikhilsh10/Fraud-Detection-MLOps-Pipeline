# Fraud Detection MLOps Pipeline

An end-to-end MLOps reference implementation demonstrating the full model lifecycle:
experiment tracking → serving → CI/CD → drift detection → automated retraining.

## Architecture

```
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
> Phase 1 goal is to verify the MLflow logging loop — scores are not meaningful until real data is used.


## Drift Threshold Justification

> *(To be written in Phase 4 — the centerpiece decision.)*

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Download dataset
python -m src.data.download

# Run experiments
python -m src.training.train

# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Project Phases

- **Phase 1** ✅ Data, features, training, MLflow experiments
- **Phase 2** 🔲 FastAPI serving + Docker
- **Phase 3** 🔲 GitHub Actions CI/CD
- **Phase 4** 🔲 Evidently drift detection + retrain loop
- **Phase 5** 🔲 Streamlit dashboard + README polish
