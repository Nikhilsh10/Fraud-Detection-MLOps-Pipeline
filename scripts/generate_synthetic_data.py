"""
scripts/generate_synthetic_data.py
------------------------------------
Generates a synthetic dataset matching the Credit Card Fraud CSV schema.
Used for local development and CI when the Kaggle dataset is not available.

The class imbalance (0.172% fraud) is preserved to match real conditions.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --rows 50000 --output data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

FRAUD_RATE = 0.00172   # matches real dataset: 492 / 284807
N_DEFAULT = 50_000     # ~18% of real dataset — enough to train, fast to generate
RANDOM_STATE = 42


def generate(n_rows: int = N_DEFAULT, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a synthetic Credit Card Fraud dataset.

    Distribution choices:
    - Time: uniform [0, 172792] — matches real dataset range
    - Amount: log-normal (mimics real skewed transaction amounts)
    - V1-V28: independent standard normals (PCA components in real data)
    - Class: Bernoulli(FRAUD_RATE) — preserves real imbalance
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(10, int(n_rows * FRAUD_RATE))
    n_legit = n_rows - n_fraud

    logger.info(
        "Generating %d rows (%d fraud / %d legitimate) …",
        n_rows, n_fraud, n_legit,
    )

    # Time: strictly increasing (per-transaction timestamp)
    time_vals = np.sort(rng.uniform(0, 172_792, n_rows))

    # Amount: log-normal, clipped to realistic range
    amount_legit = np.clip(rng.lognormal(mean=3.0, sigma=1.5, size=n_legit), 0.01, 25_000)
    # Fraud transactions: slightly higher amounts on average
    amount_fraud = np.clip(rng.lognormal(mean=4.0, sigma=2.0, size=n_fraud), 0.01, 25_000)
    amounts = np.concatenate([amount_legit, amount_fraud])

    # PCA features V1-V28: independent normals
    # Fraud class has slightly shifted means on a few features (V4, V11, V12)
    # to make drift detection realistic
    v_legit = rng.standard_normal((n_legit, 28))
    v_fraud = rng.standard_normal((n_fraud, 28))
    v_fraud[:, 3] += 2.0   # V4: shift up for fraud
    v_fraud[:, 10] -= 1.5  # V11: shift down for fraud
    v_fraud[:, 11] -= 2.0  # V12: shift down for fraud
    v_features = np.vstack([v_legit, v_fraud])

    # Build DataFrame
    data = {"Time": time_vals}
    for i in range(28):
        data[f"V{i+1}"] = v_features[:, i]
    data["Amount"] = amounts
    data["Class"] = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        "Done. Shape: %s | Fraud rate: %.4f%%",
        df.shape,
        df["Class"].mean() * 100,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Credit Card Fraud dataset")
    parser.add_argument("--rows", type=int, default=N_DEFAULT, help=f"Number of rows (default: {N_DEFAULT})")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/creditcard.csv",
        help="Output CSV path (default: data/raw/creditcard.csv)",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = generate(n_rows=args.rows, random_state=args.random_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved to %s", output_path)
    print(f"Synthetic dataset saved: {output_path} ({len(df):,} rows)")
