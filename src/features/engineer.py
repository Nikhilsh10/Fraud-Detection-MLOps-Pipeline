"""
src/features/engineer.py
-------------------------
Feature engineering pipeline for the Credit Card Fraud dataset.

Key design decisions:
- StandardScaler applied only to 'Time' and 'Amount'; V1-V28 are already
  PCA-transformed by Kaggle and do not need re-scaling.
- SMOTE is applied OUTSIDE the sklearn Pipeline to avoid target leakage.
  It is only ever fit on training data.
- The fitted ColumnTransformer is returned separately so it can be saved
  as an MLflow artifact and applied identically at inference time.

Usage:
    from src.features.engineer import build_pipeline, apply_pipeline

    X_train_processed, y_train_resampled, preprocessor = apply_pipeline(
        X_train, y_train, use_smote=True
    )
    X_test_processed = preprocessor.transform(X_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_FEATURES = ["Time", "Amount"]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]   # pass-through
TARGET_COLUMN = "Class"


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline() -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer that:
      - StandardScales 'Time' and 'Amount'
      - Passes V1-V28 through unchanged (already PCA-transformed)

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready for .fit_transform() / .transform().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                SCALE_FEATURES,
            ),
            (
                "passthrough",
                "passthrough",
                PCA_FEATURES,
            ),
        ],
        remainder="drop",           # drop any unexpected columns
        verbose_feature_names_out=False,  # keep column names clean
    )
    return preprocessor


def apply_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    use_smote: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer, np.ndarray | None]:
    """
    Fit the preprocessor on training data, apply SMOTE (optional),
    and return processed arrays.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame or None
        Optional validation / test features to transform (not fitted on).
    use_smote : bool
        Whether to apply SMOTE oversampling to the training set.
    random_state : int
        Reproducibility seed for SMOTE.

    Returns
    -------
    X_train_proc : np.ndarray
        Processed (and optionally resampled) training features.
    y_train_proc : np.ndarray
        Training labels (possibly augmented by SMOTE).
    preprocessor : ColumnTransformer
        Fitted preprocessor — save this as an MLflow artifact.
    X_val_proc : np.ndarray or None
        Processed validation features, or None if X_val was not supplied.
    """
    preprocessor = build_pipeline()

    # Fit and transform training data
    X_train_proc = preprocessor.fit_transform(X_train)
    y_train_proc = y_train.to_numpy()

    logger.info(
        "Preprocessor fitted. Train shape: %s | Fraud ratio: %.4f%%",
        X_train_proc.shape,
        y_train_proc.mean() * 100,
    )

    # SMOTE — only on training data, after scaling
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE  # lazy import

            smote = SMOTE(random_state=random_state)
            X_train_proc, y_train_proc = smote.fit_resample(X_train_proc, y_train_proc)
            logger.info(
                "SMOTE applied. New train shape: %s | Fraud ratio: %.4f%%",
                X_train_proc.shape,
                y_train_proc.mean() * 100,
            )
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed — SMOTE skipped. "
                "Install with: pip install imbalanced-learn"
            )
    else:
        logger.info("SMOTE skipped (use_smote=False).")

    # Transform validation / test data (never fit on it)
    X_val_proc = None
    if X_val is not None:
        X_val_proc = preprocessor.transform(X_val)
        logger.info("Validation set transformed. Shape: %s", X_val_proc.shape)

    return X_train_proc, y_train_proc, preprocessor, X_val_proc


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return ordered feature names after transformation."""
    return SCALE_FEATURES + PCA_FEATURES


def load_and_split(
    csv_path: str | Path,
    train_frac: float = 0.70,
    ref_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame, pd.Series,   # train X, y
    pd.DataFrame, pd.Series,   # reference X, y  (for drift detection)
    pd.DataFrame, pd.Series,   # test X, y
]:
    """
    Load the raw CSV and produce a reproducible 70/15/15 split.

    The reference split is saved to data/reference/train_distribution.csv
    (feature-level statistics, not raw rows) for use by the drift detector.

    Parameters
    ----------
    csv_path : str | Path
        Path to data/raw/creditcard.csv.
    train_frac : float
        Fraction for training (default 0.70).
    ref_frac : float
        Fraction for reference/drift baseline (default 0.15).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Six DataFrames/Series: X_train, y_train, X_ref, y_ref, X_test, y_test
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    test_frac = 1.0 - train_frac - ref_frac
    assert test_frac > 0, "train_frac + ref_frac must be < 1.0"

    # First split: train vs (ref + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(ref_frac + test_frac),
        random_state=random_state,
        stratify=y,
    )

    # Second split: ref vs test
    ref_relative = ref_frac / (ref_frac + test_frac)
    X_ref, X_test, y_ref, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - ref_relative),
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Split sizes — train: %d | ref: %d | test: %d",
        len(X_train), len(X_ref), len(X_test),
    )

    # Save reference distribution for drift detection
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_stats = X_ref.describe().T  # feature-level stats, not raw rows
    ref_stats.to_csv(ref_dir / "train_distribution.csv")
    # Also save raw reference data for Evidently (needed for full drift report)
    ref_data = X_ref.copy()
    ref_data[TARGET_COLUMN] = y_ref.values
    ref_data.to_csv(ref_dir / "reference_data.csv", index=False)
    logger.info("Reference distribution saved to data/reference/")

    return X_train, y_train, X_ref, y_ref, X_test, y_test
