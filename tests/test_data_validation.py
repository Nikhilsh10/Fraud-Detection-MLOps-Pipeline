"""
tests/test_data_validation.py
------------------------------
Unit tests for src/data/validate.py.

Four cases:
1. Happy path — clean DataFrame passes all checks.
2. Missing required column → errors list non-empty, passed=False.
3. Null percentage above threshold → passed=False.
4. Wrong dtype (string column) → passed=False.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validate import (
    MAX_NULL_FRACTION,
    ValidationResult,
    _check_columns,
    _check_dtypes,
    _check_nulls,
    _check_ranges,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clean_df(n_rows: int = 100) -> pd.DataFrame:
    """Build a minimal but valid Credit Card Fraud DataFrame."""
    rng = np.random.default_rng(0)
    data = {
        "Time": rng.uniform(0, 100_000, n_rows),
        "Amount": rng.uniform(0, 1_000, n_rows),
        "Class": rng.integers(0, 2, n_rows),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n_rows)
    return pd.DataFrame(data)


def _df_to_csv(df: pd.DataFrame) -> Path:
    """Write df to a temp CSV file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="w", newline=""
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Test 1 — Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_clean_dataframe_passes(self):
        """A well-formed DataFrame should pass all validation checks."""
        df = _make_clean_df()
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert result.passed is True, f"Expected pass but got errors: {result.errors}"
        assert result.errors == []
        assert "row_count" in result.stats
        assert result.stats["row_count"] == 100

    def test_class_distribution_in_stats(self):
        """Stats dict should contain class distribution info."""
        df = _make_clean_df(n_rows=500)
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert "class_distribution" in result.stats
        dist = result.stats["class_distribution"]
        assert "fraud" in dist
        assert "non_fraud" in dist
        assert dist["fraud"] + dist["non_fraud"] == 500


# ---------------------------------------------------------------------------
# Test 2 — Missing required column
# ---------------------------------------------------------------------------

class TestMissingColumn:
    def test_missing_v1_fails(self):
        """Dropping V1 should cause the validation to fail."""
        df = _make_clean_df()
        df = df.drop(columns=["V1"])
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert result.passed is False
        assert any("V1" in err for err in result.errors), \
            f"Expected 'V1' in errors, got: {result.errors}"

    def test_missing_class_column_fails(self):
        """Dropping the 'Class' target column should fail."""
        df = _make_clean_df()
        df = df.drop(columns=["Class"])
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert result.passed is False
        assert any("Class" in err for err in result.errors)

    def test_missing_time_and_amount_fails(self):
        """Dropping both Time and Amount should fail."""
        df = _make_clean_df()
        df = df.drop(columns=["Time", "Amount"])
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert result.passed is False
        assert len(result.errors) >= 1  # at least one error for missing cols


# ---------------------------------------------------------------------------
# Test 3 — Null percentage above threshold
# ---------------------------------------------------------------------------

class TestNullThreshold:
    def test_high_null_fraction_fails(self):
        """A column with 20% nulls should trigger a validation error."""
        df = _make_clean_df(n_rows=1_000)
        # Inject 20% nulls into V1 — well above 0.1% threshold
        null_idx = df.sample(frac=0.20, random_state=1).index
        df.loc[null_idx, "V1"] = np.nan
        csv_path = _df_to_csv(df)

        try:
            result = validate(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        assert result.passed is False
        assert any("V1" in err for err in result.errors), \
            f"Expected null error for V1, got: {result.errors}"

    def test_zero_nulls_passes(self):
        """A DataFrame with no nulls should pass the null check in isolation."""
        df = _make_clean_df(n_rows=200)
        result = ValidationResult()
        _check_nulls(df, result)
        assert result.passed is True
        assert result.errors == []

    def test_null_just_below_threshold_passes(self):
        """Nulls at exactly MAX_NULL_FRACTION - epsilon should pass."""
        n = 10_000
        df = _make_clean_df(n_rows=n)
        # Inject nulls just below threshold
        null_count = max(0, int(n * MAX_NULL_FRACTION) - 1)
        if null_count > 0:
            null_idx = df.sample(n=null_count, random_state=2).index
            df.loc[null_idx, "V2"] = np.nan
        result = ValidationResult()
        _check_nulls(df, result)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4 — Wrong dtype
# ---------------------------------------------------------------------------

class TestWrongDtype:
    def test_string_column_fails(self):
        """Replacing a float column with strings should fail dtype check."""
        df = _make_clean_df(n_rows=100)
        df["V1"] = df["V1"].astype(str).replace(to_replace=".*", value="bad_val", regex=True)
        result = ValidationResult()
        _check_dtypes(df, result)

        assert result.passed is False
        assert any("V1" in err for err in result.errors)

    def test_all_numeric_passes(self):
        """A clean DataFrame should pass dtype check."""
        df = _make_clean_df(n_rows=100)
        result = ValidationResult()
        _check_dtypes(df, result)
        assert result.passed is True

    def test_nonexistent_file_fails(self):
        """Validate on a missing path should return passed=False immediately."""
        result = validate(Path("nonexistent_file_that_does_not_exist.csv"))
        assert result.passed is False
        assert any("not found" in err.lower() for err in result.errors)
