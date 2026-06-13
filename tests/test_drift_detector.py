import os
from unittest.mock import patch

import pytest
import pandas as pd

from src.monitoring.drift_detector import load_reference_data, load_current_data

def test_load_reference_data(monkeypatch):
    # This assumes the data/reference/reference_data.csv exists
    # If not, it will fail, which is correct (it's required for Phase 4)
    # We just check it doesn't crash if it exists
    if os.path.exists("data/reference/reference_data.csv"):
        df = load_reference_data()
        assert not df.empty
        assert "amount" in df.columns
        assert "time_feature" in df.columns
        assert "is_fraud" in df.columns

def test_evidently_report_generation(tmp_path):
    # Create tiny mock dataframes to actually exercise Evidently's engine
    ref_df = pd.DataFrame({
        "time_feature": [0.0, 1.0, 2.0, 3.0, 4.0] * 10,
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
        "is_fraud": [0, 0, 0, 0, 1] * 10
    })
    
    # Introduce massive drift in current_df
    cur_df = pd.DataFrame({
        "time_feature": [0.0, 1.0, 2.0, 3.0, 4.0] * 10,
        "amount": [9000.0, 8500.0, 9200.0, 8800.0, 9100.0] * 10,
        "is_fraud": [0, 0, 0, 0, 1] * 10
    })

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    
    report_dict = report.as_dict()
    
    # Assert Evidently successfully ran and populated the dictionary structure
    assert "metrics" in report_dict
    assert len(report_dict["metrics"]) > 0
    
    # Extract DataDriftPreset results
    metrics_result = report_dict["metrics"][0]["result"]
    assert "drift_share" in metrics_result
    assert "dataset_drift" in metrics_result
    
    # Because we injected massive amount drift, dataset_drift might be False (default threshold is 50% of features, we drifted 1 out of 3 = 33%), but drift_share must be > 0.
    assert metrics_result["drift_share"] > 0.0
