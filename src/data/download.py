"""
src/data/download.py
--------------------
Downloads the Kaggle Credit Card Fraud dataset via the kaggle CLI,
verifies the file landed correctly, and saves it to data/raw/creditcard.csv.

Usage:
    python -m src.data.download
    python -m src.data.download --output-dir data/raw

Environment variables required (in .env or shell):
    KAGGLE_USERNAME
    KAGGLE_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
EXPECTED_FILENAME = "creditcard.csv"
EXPECTED_MIN_ROWS = 280_000  # dataset has 284,807 rows; leave a little slack
EXPECTED_COLUMNS = {
    "Time", "Amount", "Class",
    *[f"V{i}" for i in range(1, 29)],  # V1 … V28
}


def _check_kaggle_credentials() -> None:
    """Fail fast if Kaggle credentials are missing.

    Accepts credentials from either:
      1. A kaggle.json file at the standard path (~/.config/kaggle/kaggle.json)
         Written by CI from the KAGGLE_JSON secret.
      2. KAGGLE_USERNAME + KAGGLE_KEY environment variables (local dev / .env).
    """
    kaggle_json = Path.home() / ".config" / "kaggle" / "kaggle.json"
    if kaggle_json.exists():
        logger.info("Kaggle credentials found via kaggle.json at %s", kaggle_json)
        return

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        logger.error(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables. "
            "Export them or add them to a .env file."
        )
        sys.exit(1)
    logger.info("Kaggle credentials found for user: %s", username)


def _download(output_dir: Path) -> Path:
    """
    Run the kaggle CLI to download the dataset.
    Returns the path to the unzipped CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "creditcardfraud.zip"

    if zip_path.exists():
        logger.info("Zip already exists at %s — skipping download.", zip_path)
    else:
        logger.info("Downloading dataset '%s' from Kaggle …", KAGGLE_DATASET)
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("kaggle CLI failed:\n%s", result.stderr)
            sys.exit(1)
        logger.info("Download complete.")

    # Unzip
    csv_path = output_dir / EXPECTED_FILENAME
    if csv_path.exists():
        logger.info("CSV already extracted at %s — skipping unzip.", csv_path)
    else:
        logger.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        logger.info("Extraction complete.")

    return csv_path


def _verify(csv_path: Path) -> None:
    """
    Basic sanity checks on the downloaded CSV.
    Exits with a non-zero code on failure.
    """
    import pandas as pd  # local import — only needed here

    if not csv_path.exists():
        logger.error("Expected file not found: %s", csv_path)
        sys.exit(1)

    logger.info("Verifying %s …", csv_path)
    df = pd.read_csv(csv_path, nrows=5)  # read header + a few rows first

    # Column check
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        logger.error("Missing expected columns: %s", missing_cols)
        sys.exit(1)

    # Row count (read full file)
    total_rows = sum(1 for _ in open(csv_path)) - 1  # subtract header
    if total_rows < EXPECTED_MIN_ROWS:
        logger.error(
            "Row count %d is below expected minimum %d. "
            "The download may be incomplete.",
            total_rows,
            EXPECTED_MIN_ROWS,
        )
        sys.exit(1)

    logger.info(
        "Verification passed: %d rows, %d columns.", total_rows, len(df.columns)
    )


def download(output_dir: str | Path = "data/raw") -> Path:
    """
    Public entry point.
    Returns the path to the verified creditcard.csv.
    """
    output_dir = Path(output_dir)
    _check_kaggle_credentials()
    csv_path = _download(output_dir)
    _verify(csv_path)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle Credit Card Fraud dataset")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save the dataset (default: data/raw)",
    )
    args = parser.parse_args()
    result_path = download(args.output_dir)
    print(f"Dataset ready at: {result_path}")
