"""
src/training/train.py
----------------------
MLflow experiment wrapper for the Credit Card Fraud detector.

Logs: params, metrics (roc_auc, avg_precision, f1), SHAP summary artifact.
Registers the best model to the MLflow Model Registry as 'fraud_detector'.

Usage:
    # Single run with defaults
    python -m src.training.train

    # Custom params
    python -m src.training.train \
        --n-estimators 200 \
        --max-depth 10 \
        --class-weight balanced \
        --use-smote false \
        --experiment-name fraud_detector_experiments

    # Drift-retrain trigger (sets tags accordingly)
    python -m src.training.train --trigger drift_retrain --drift-score 0.62
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.features.engineer import apply_pipeline, load_and_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_MODEL_NAME = "fraud_detector"
DEFAULT_EXPERIMENT_NAME = "fraud_detector_experiments"
DATA_CSV = "data/raw/creditcard.csv"
REPORTS_DIR = Path("reports")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------

def _generate_shap_plot(
    model: RandomForestClassifier,
    X_sample: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> Path | None:
    """
    Generate a SHAP summary plot and save it as a PNG.
    Returns the path on success, None if shap is unavailable.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        logger.info("Computing SHAP values on %d samples …", len(X_sample))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values for RF is a list [class_0, class_1]; take class_1
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved to %s", output_path)
        return output_path

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP plot generation failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def train(
    n_estimators: int = 100,
    max_depth: int | None = None,
    class_weight: str | dict = "balanced",
    use_smote: bool = True,
    random_state: int = 42,
    trigger: str = "manual",
    drift_score: float | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    register_model: bool = True,
    data_csv: str | Path = DATA_CSV,
) -> str:
    """
    Run a single MLflow training experiment.

    Parameters
    ----------
    n_estimators : int
    max_depth : int or None
    class_weight : str or dict
        e.g. 'balanced', 'balanced_subsample', or {0: 1, 1: 10}
    use_smote : bool
    random_state : int
    trigger : str
        'manual' | 'drift_retrain'  — stored as an MLflow tag.
    drift_score : float or None
        Drift share score that triggered retraining, if applicable.
    experiment_name : str
    register_model : bool
        Whether to push to the MLflow Model Registry.
    data_csv : str | Path

    Returns
    -------
    str
        MLflow run_id of the completed run.
    """
    # ------------------------------------------------------------------
    # Setup MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Load + split data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s …", data_csv)
    X_train, y_train, X_ref, y_ref, X_test, y_test = load_and_split(data_csv)

    window_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    logger.info(
        "Applying feature pipeline (use_smote=%s) …", use_smote
    )
    X_train_proc, y_train_proc, preprocessor, X_test_proc = apply_pipeline(
        X_train, y_train, X_val=X_test, use_smote=use_smote
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(
        "Training RandomForest(n_estimators=%d, max_depth=%s, "
        "class_weight=%s) …",
        n_estimators, max_depth, class_weight,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_proc, y_train_proc)

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_precision = float(average_precision_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    logger.info(
        "Metrics — ROC-AUC: %.4f | Avg Precision: %.4f | F1: %.4f",
        roc_auc, avg_precision, f1,
    )

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth is not None else "None",
        "class_weight": str(class_weight),
        "use_smote": use_smote,
        "random_state": random_state,
        "train_size": len(X_train_proc),
        "test_size": len(X_test),
    }
    metrics = {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "f1": f1,
    }
    tags = {
        "trigger": trigger,
        "training_data_window": f"{window_start}:{window_start}",
    }
    if drift_score is not None:
        tags["drift_score"] = str(round(drift_score, 4))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)

        # Log SHAP plot
        shap_path = REPORTS_DIR / "shap_summary.png"
        # Use a subsample for speed (SHAP on 200 samples is representative)
        shap_sample_size = min(200, len(X_test_proc))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_test_proc), size=shap_sample_size, replace=False)
        from src.features.engineer import get_feature_names
        feature_names = get_feature_names(preprocessor)
        shap_file = _generate_shap_plot(
            model,
            X_test_proc[idx],
            feature_names,
            shap_path,
        )
        if shap_file:
            mlflow.log_artifact(str(shap_file), artifact_path="plots")

        # Log preprocessor artifact
        import pickle
        prep_path = REPORTS_DIR / "preprocessor.pkl"
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact(str(prep_path), artifact_path="preprocessor")

        # Register model
        if register_model:
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=REGISTRY_MODEL_NAME,
            )
            logger.info(
                "Model registered: %s | URI: %s",
                REGISTRY_MODEL_NAME,
                model_info.model_uri,
            )
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("MLflow run completed: %s", run_id)

    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detector with MLflow tracking")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--max-depth",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="'balanced', 'balanced_subsample', or 'none'",
    )
    parser.add_argument(
        "--use-smote",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        choices=["manual", "drift_retrain"],
    )
    parser.add_argument("--drift-score", type=float, default=None)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
    )
    parser.add_argument("--data-csv", type=str, default=DATA_CSV)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip model registry (useful for quick test runs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    class_weight: str | None = (
        None if args.class_weight == "none" else args.class_weight
    )
    run_id = train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=class_weight,
        use_smote=args.use_smote,
        random_state=args.random_state,
        trigger=args.trigger,
        drift_score=args.drift_score,
        experiment_name=args.experiment_name,
        register_model=not args.no_register,
        data_csv=args.data_csv,
    )
    print(f"Run ID: {run_id}")
