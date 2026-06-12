"""
src/data/validate.py
---------------------
Schema, dtype, null, and range validation for the Credit Card Fraud dataset.

Returns a structured result dict — compatible with use inside a GitHub Actions
step (exits non-zero on failure) and importable for unit tests.

Usage:
    python -m src.data.validate                         # uses default path
    python -m src.data.validate --csv data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: dict[str, type] = {
    "Time": float,
    "Amount": float,
    "Class": int,
    **{f"V{i}": float for i in range(1, 29)},
}

# Acceptable null percentage per column (fraction, not percent)
MAX_NULL_FRACTION: float = 0.001  # 0.1 %

# Plausible value ranges (guard against corrupt downloads)
RANGE_GUARDS: dict[str, tuple[float, float]] = {
    "Time": (0.0, 200_000.0),       # seconds elapsed; dataset max ≈ 172792
    "Amount": (0.0, 30_000.0),      # highest transaction in dataset ≈ 25691
    "Class": (0, 1),                 # binary label
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Verify that all expected columns are present."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing:
        result.fail(f"Missing required columns: {sorted(missing)}")
    if extra:
        result.warn(f"Unexpected extra columns (ignored): {sorted(extra)}")

    result.stats["columns_found"] = len(df.columns)
    result.stats["columns_expected"] = len(EXPECTED_COLUMNS)


def _check_dtypes(df: pd.DataFrame, result: ValidationResult) -> None:
    """
    Verify dtypes are numeric (float or int) for all expected columns.
    We check numeric compatibility rather than exact dtype to handle
    int64 / float32 variants from different pandas versions.
    """
    non_numeric = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            continue  # already caught by _check_columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        result.fail(f"Non-numeric columns (expected numeric): {non_numeric}")


def _check_nulls(df: pd.DataFrame, result: ValidationResult) -> None:
    """Fail if any column exceeds the null fraction threshold."""
    null_fractions = df.isnull().mean()
    violations = null_fractions[null_fractions > MAX_NULL_FRACTION]

    result.stats["null_fractions"] = null_fractions[null_fractions > 0].to_dict()

    if not violations.empty:
        for col, frac in violations.items():
            result.fail(
                f"Column '{col}' has {frac:.4%} nulls "
                f"(threshold: {MAX_NULL_FRACTION:.4%})"
            )


def _check_ranges(df: pd.DataFrame, result: ValidationResult) -> None:
    """Guard against obviously corrupt / wrong data via value range checks."""
    for col, (lo, hi) in RANGE_GUARDS.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        result.stats[f"{col}_range"] = {"min": float(col_min), "max": float(col_max)}

        if col_min < lo:
            result.fail(
                f"Column '{col}' has min {col_min} below expected floor {lo}"
            )
        if col_max > hi:
            result.fail(
                f"Column '{col}' has max {col_max} above expected ceiling {hi}"
            )


def _check_row_count(df: pd.DataFrame, result: ValidationResult) -> None:
    """Warn if the dataset looks unexpectedly small."""
    n = len(df)
    result.stats["row_count"] = n
    if n < 1_000:
        result.warn(f"Only {n} rows found — expected ~284,807 for full dataset.")


def _check_class_balance(df: pd.DataFrame, result: ValidationResult) -> None:
    """Log class distribution info (not a failure, just informational)."""
    if "Class" not in df.columns:
        return
    counts = df["Class"].value_counts().to_dict()
    fraud_pct = counts.get(1, 0) / len(df) * 100
    result.stats["class_distribution"] = {
        "non_fraud": int(counts.get(0, 0)),
        "fraud": int(counts.get(1, 0)),
        "fraud_pct": round(fraud_pct, 4),
    }
    if fraud_pct < 0.05:
        result.warn(
            f"Fraud class is only {fraud_pct:.4f}% of data — "
            "very imbalanced, ensure class_weight='balanced' is used in training."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(csv_path: str | Path) -> ValidationResult:
    """
    Run all checks against the CSV at csv_path.
    Returns a ValidationResult (never raises — caller decides how to handle).
    """
    csv_path = Path(csv_path)
    result = ValidationResult()

    if not csv_path.exists():
        result.fail(f"File not found: {csv_path}")
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        result.fail(f"Failed to read CSV: {exc}")
        return result

    _check_columns(df, result)
    _check_dtypes(df, result)
    _check_nulls(df, result)
    _check_ranges(df, result)
    _check_row_count(df, result)
    _check_class_balance(df, result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the Credit Card Fraud CSV")
    parser.add_argument(
        "--csv",
        default="data/raw/creditcard.csv",
        help="Path to the CSV file (default: data/raw/creditcard.csv)",
    )
    args = parser.parse_args()

    res = validate(args.csv)

    if res.warnings:
        for w in res.warnings:
            logger.warning(w)

    if res.passed:
        logger.info("Validation PASSED. Stats: %s", res.stats)
        sys.exit(0)
    else:
        for err in res.errors:
            logger.error(err)
        logger.error("Validation FAILED with %d error(s).", len(res.errors))
        sys.exit(1)
