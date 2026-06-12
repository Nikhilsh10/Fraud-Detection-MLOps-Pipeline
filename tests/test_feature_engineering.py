"""
tests/test_feature_engineering.py
-----------------------------------
Unit tests for src/features/engineer.py.

Three cases:
1. Pipeline output shape is correct (30 features in → 30 out).
2. No data leakage: preprocessor fitted on train; applying to test gives
   different column statistics than if fitted on test directly.
3. Preprocessor is sklearn-compatible (has fit/transform/get_params).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from src.features.engineer import (
    PCA_FEATURES,
    SCALE_FEATURES,
    apply_pipeline,
    build_pipeline,
    get_feature_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_rows: int = 500,
    fraud_frac: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset matching the Kaggle CSV schema."""
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_rows * fraud_frac))
    n_legit = n_rows - n_fraud

    data = {
        "Time": rng.uniform(0, 172_792, n_rows),
        "Amount": rng.uniform(0, 5_000, n_rows),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n_rows)

    X = pd.DataFrame(data)
    y = pd.Series(
        [1] * n_fraud + [0] * n_legit,
        dtype=int,
        name="Class",
    )
    # Shuffle
    idx = rng.permutation(n_rows)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test 1 — Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_feature_count_preserved(self):
        """
        Preprocessor should output exactly len(SCALE_FEATURES + PCA_FEATURES)
        columns — no features dropped, no extras added.
        """
        X, y = _make_dataset(n_rows=200)
        X_train, y_train = X.iloc[:150], y.iloc[:150]
        X_test = X.iloc[150:]

        X_proc, y_proc, preprocessor, X_test_proc = apply_pipeline(
            X_train, y_train, X_val=X_test, use_smote=False
        )

        expected_features = len(SCALE_FEATURES) + len(PCA_FEATURES)  # 30
        assert X_proc.shape[1] == expected_features, (
            f"Expected {expected_features} features, got {X_proc.shape[1]}"
        )
        assert X_test_proc.shape[1] == expected_features

    def test_smote_increases_sample_count(self):
        """SMOTE should upsample the minority class, increasing row count."""
        X, y = _make_dataset(n_rows=500, fraud_frac=0.02)
        X_train, y_train = X.iloc[:400], y.iloc[:400]

        X_proc_no_smote, y_no_smote, _, _ = apply_pipeline(
            X_train, y_train, use_smote=False
        )
        X_proc_smote, y_smote, _, _ = apply_pipeline(
            X_train, y_train, use_smote=True
        )

        assert len(X_proc_smote) > len(X_proc_no_smote), (
            "SMOTE should produce more rows than the original training set"
        )
        # After SMOTE, fraud and non-fraud should be balanced
        assert (y_smote == 1).sum() == (y_smote == 0).sum(), \
            "SMOTE should produce a balanced dataset"

    def test_output_is_numpy_array(self):
        """apply_pipeline should return numpy arrays, not DataFrames."""
        X, y = _make_dataset(n_rows=100)
        X_proc, y_proc, _, _ = apply_pipeline(X, y, use_smote=False)

        assert isinstance(X_proc, np.ndarray), "X_proc should be np.ndarray"
        assert isinstance(y_proc, np.ndarray), "y_proc should be np.ndarray"

    def test_x_val_none_returns_none(self):
        """When X_val is not provided, the fourth return value should be None."""
        X, y = _make_dataset(n_rows=100)
        _, _, _, X_val_proc = apply_pipeline(X, y, use_smote=False, X_val=None)
        assert X_val_proc is None


# ---------------------------------------------------------------------------
# Test 2 — No data leakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_scaler_fitted_on_train_not_test(self):
        """
        Fitting on train and transforming test should NOT be equivalent
        to fitting on test data directly.
        The scaler's mean (fitted on train) should differ from the test mean.
        This confirms the preprocessor was not re-fitted on test data.
        """
        rng = np.random.default_rng(99)

        # Train: Amount drawn from N(100, 10)
        n_train = 300
        X_train = pd.DataFrame({
            "Time": rng.uniform(0, 100_000, n_train),
            "Amount": rng.normal(100, 10, n_train),   # mean ~100
            **{f"V{i}": rng.standard_normal(n_train) for i in range(1, 29)},
        })
        y_train = pd.Series(np.zeros(n_train, dtype=int), name="Class")

        # Test: Amount drawn from N(5000, 10) — very different distribution
        n_test = 100
        X_test = pd.DataFrame({
            "Time": rng.uniform(0, 100_000, n_test),
            "Amount": rng.normal(5000, 10, n_test),   # mean ~5000
            **{f"V{i}": rng.standard_normal(n_test) for i in range(1, 29)},
        })

        _, _, preprocessor, X_test_proc = apply_pipeline(
            X_train, y_train, X_val=X_test, use_smote=False
        )

        # The Amount column in X_test_proc should be (5000 - train_mean) / train_std
        # If there were leakage (fitted on test), the mean would be near 0.
        # With correct fit-on-train, the mean should be >> 0.
        amount_col_idx = 1  # 'Amount' is second in SCALE_FEATURES
        test_amount_scaled_mean = X_test_proc[:, amount_col_idx].mean()

        assert abs(test_amount_scaled_mean) > 10, (
            f"Scaled Amount mean is {test_amount_scaled_mean:.2f}; "
            "expected >> 0 (train scaler applied to shifted test data). "
            "Possible data leakage: scaler may have been refitted on test data."
        )

    def test_preprocessor_mean_matches_train(self):
        """
        The scaler's learned mean should match the training data mean,
        not the test data mean.
        """
        rng = np.random.default_rng(7)
        n = 400
        train_amount_mean = 200.0

        X_train = pd.DataFrame({
            "Time": rng.uniform(0, 50_000, n),
            "Amount": rng.normal(train_amount_mean, 5, n),
            **{f"V{i}": rng.standard_normal(n) for i in range(1, 29)},
        })
        y_train = pd.Series(np.zeros(n, dtype=int))

        _, _, preprocessor, _ = apply_pipeline(X_train, y_train, use_smote=False)

        # Extract the StandardScaler from the ColumnTransformer
        scaler = preprocessor.named_transformers_["scale"]
        learned_amount_mean = scaler.mean_[1]  # Amount is index 1 in SCALE_FEATURES

        assert abs(learned_amount_mean - train_amount_mean) < 5, (
            f"Scaler learned mean {learned_amount_mean:.1f} differs too much "
            f"from training mean {train_amount_mean:.1f}."
        )


# ---------------------------------------------------------------------------
# Test 3 — Sklearn compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompatibility:
    def test_preprocessor_has_fit_method(self):
        """build_pipeline() result must have a fit method."""
        preprocessor = build_pipeline()
        assert hasattr(preprocessor, "fit"), "Preprocessor must have .fit()"

    def test_preprocessor_has_transform_method(self):
        """build_pipeline() result must have a transform method."""
        preprocessor = build_pipeline()
        assert hasattr(preprocessor, "transform"), "Preprocessor must have .transform()"

    def test_preprocessor_has_get_params(self):
        """build_pipeline() result must have get_params (sklearn convention)."""
        preprocessor = build_pipeline()
        params = preprocessor.get_params()
        assert isinstance(params, dict), "get_params() must return a dict"
        assert len(params) > 0

    def test_fit_transform_idempotent(self):
        """
        fit_transform twice on the same data should give the same result
        (preprocessor is stateless between calls when re-built).
        """
        X, y = _make_dataset(n_rows=150)

        pp1 = build_pipeline()
        result1 = pp1.fit_transform(X)

        pp2 = build_pipeline()
        result2 = pp2.fit_transform(X)

        np.testing.assert_allclose(
            result1, result2, rtol=1e-10,
            err_msg="Two fresh pipelines fit on the same data should give identical results."
        )

    def test_get_feature_names_returns_correct_count(self):
        """get_feature_names should return exactly 30 names."""
        X, y = _make_dataset(n_rows=50)
        _, _, preprocessor, _ = apply_pipeline(X, y, use_smote=False)
        names = get_feature_names(preprocessor)
        assert len(names) == 30, f"Expected 30 feature names, got {len(names)}"
        assert names[0] == "Time"
        assert names[1] == "Amount"
        assert names[2] == "V1"
