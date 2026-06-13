"""
src/monitoring/drift_detector.py
--------------------------------
Drift detector using Evidently. Compares the reference dataset
(data/reference/reference_data.csv) against recent predictions from predictions.db.
"""

import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", "predictions.db"))
REFERENCE_PATH = Path("data/reference/reference_data.csv")
REPORT_HTML = Path("reports/drift_report.html")
REPORT_JSON = Path("reports/drift_summary.json")

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.25"))
GH_PAT = os.getenv("GH_PAT")


def load_reference_data() -> pd.DataFrame:
    """Load reference data and keep only columns we have in production."""
    if not REFERENCE_PATH.exists():
        print(f"Error: Reference data not found at {REFERENCE_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(REFERENCE_PATH)
    # We only log Time and Amount in production DB
    cols_to_keep = ["Time", "Amount", "Class"]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    return df[available_cols].rename(columns={"Time": "time_feature", "Amount": "amount", "Class": "is_fraud"})


def load_current_data() -> pd.DataFrame:
    """Load recent predictions from SQLite."""
    if not DB_PATH.exists():
        print(f"Warning: DB not found at {DB_PATH}. No predictions yet.")
        return pd.DataFrame(columns=["time_feature", "amount", "is_fraud"])
        
    conn = sqlite3.connect(str(DB_PATH))
    # Load last 1000 predictions
    query = """
        SELECT time_feature, amount, is_fraud
        FROM predictions_log
        ORDER BY ts DESC
        LIMIT 1000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def trigger_github_retrain(drift_share: float):
    """Trigger GitHub Actions retrain workflow via repository_dispatch."""
    if not GH_PAT:
        print("GH_PAT not set. Skipping GitHub Actions retrain trigger.")
        return

    # In GitHub Actions, GITHUB_REPOSITORY is automatically set (e.g. "owner/repo")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY not set. Cannot trigger retrain.")
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GH_PAT}",
    }
    data = {
        "event_type": "retrain-model",
        "client_payload": {
            "drift_share": drift_share,
            "message": "Data drift detected. Triggering automated retrain."
        }
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        urllib.request.urlopen(req)
        print("Successfully triggered retrain-model repository_dispatch.")
    except Exception as e:
        print(f"Failed to trigger retrain: {e}")


def main():
    print("Starting Drift Detection...")
    
    ref_df = load_reference_data()
    cur_df = load_current_data()
    
    if len(cur_df) < 50:
        print(f"Not enough current data ({len(cur_df)} rows). Need at least 50. Skipping drift check.")
        sys.exit(0)
        
    print(f"Reference size: {len(ref_df)}")
    print(f"Current size:   {len(cur_df)}")
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    
    # Ensure reports dir exists
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    
    report.save_html(str(REPORT_HTML))
    print(f"Saved HTML report to {REPORT_HTML}")
    
    report_dict = report.as_dict()
    # Save the full JSON
    with open(REPORT_JSON, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved JSON report to {REPORT_JSON}")
    
    metrics = report_dict["metrics"][0]["result"]
    drift_share = metrics["drift_share"]
    dataset_drift = metrics["dataset_drift"]
    
    print(f"\nDrift Results:")
    print(f"  Dataset Drifted: {dataset_drift}")
    print(f"  Drift Share:     {drift_share:.2f} (Threshold: {DRIFT_THRESHOLD})")
    
    if drift_share > DRIFT_THRESHOLD:
        print(f"\n🚨 DRIFT DETECTED (share > {DRIFT_THRESHOLD})")
        trigger_github_retrain(drift_share)
        sys.exit(1)
    else:
        print("\n✅ No significant drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
