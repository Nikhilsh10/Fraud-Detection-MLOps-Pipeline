import json
import os
import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import mlflow

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🕵️", layout="wide")

DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", "predictions.db"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

st.title("🕵️ Fraud Detection MLOps Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Overview & Logs", "🧪 Live Prediction", "📈 Drift Monitoring"])

# Helper functions
@st.cache_data(ttl=10)
def get_predictions_log():
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("SELECT * FROM predictions_log ORDER BY ts DESC LIMIT 100", conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_mlflow_metrics():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias("fraud_detector", "Production")
        run = client.get_run(mv.run_id)
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "metrics": run.data.metrics
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    st.header("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model in Production")
        model_info = get_mlflow_metrics()
        if "error" not in model_info:
            st.metric("Model Version", f"v{model_info['version']}")
            st.caption(f"Run ID: {model_info['run_id']}")
            
            st.write("**Key Metrics (from training):**")
            metrics = model_info["metrics"]
            if metrics:
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Avg Precision", f"{metrics.get('test_avg_precision', 0):.4f}")
                m_col2.metric("F1 Score", f"{metrics.get('test_f1', 0):.4f}")
                m_col3.metric("ROC AUC", f"{metrics.get('test_roc_auc', 0):.4f}")
        else:
            st.warning("Could not load MLflow data: " + model_info["error"])

    with col2:
        st.subheader("Recent Activity")
        df_log = get_predictions_log()
        if not df_log.empty:
            total_preds = len(df_log)
            fraud_preds = df_log["is_fraud"].sum()
            st.metric("Recent Predictions (limit 100)", total_preds)
            st.metric("Fraud Flags", fraud_preds)
        else:
            st.info("No predictions logged yet.")
            
    st.divider()
    st.subheader("Recent Predictions Log")
    if not df_log.empty:
        st.dataframe(df_log, use_container_width=True)


# ---------------- TAB 2: LIVE PREDICTION ----------------
with tab2:
    st.header("Test Live Prediction")
    st.write("Send a transaction to the FastAPI serving layer.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time", value=0.0)
            amount_val = st.number_input("Amount", value=100.0)
        with col2:
            st.write("Feature Values (V1 - V28 defaults to 0.0 for quick testing)")
            v1_val = st.number_input("V1", value=0.0)
            v2_val = st.number_input("V2", value=0.0)
            
        submit = st.form_submit_button("Predict")
        
        if submit:
            payload = {
                "Time": time_val,
                "Amount": amount_val,
            }
            # Fill V1 to V28
            for i in range(1, 29):
                if i == 1: payload[f"V{i}"] = v1_val
                elif i == 2: payload[f"V{i}"] = v2_val
                else: payload[f"V{i}"] = 0.0
                
            try:
                with st.spinner("Calling API..."):
                    resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=5.0)
                if resp.status_code == 200:
                    result = resp.json()
                    if result["is_fraud"]:
                        st.error(f"🚨 FRAUD DETECTED (Probability: {result['fraud_probability']:.2%})")
                    else:
                        st.success(f"✅ Transaction Approved (Probability: {result['fraud_probability']:.2%})")
                    st.json(result)
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to API at {API_URL}: {e}")


# ---------------- TAB 3: DRIFT MONITORING ----------------
with tab3:
    st.header("Data Drift Monitoring")
    st.write("Results from the latest Evidently drift detection run.")
    
    report_path = Path("reports/drift_summary.json")
    if report_path.exists():
        with open(report_path, "r") as f:
            drift_data = json.load(f)
            
        try:
            metrics = drift_data["metrics"][0]["result"]
            dataset_drift = metrics["dataset_drift"]
            drift_share = metrics["drift_share"]
            
            st.metric("Drift Share", f"{drift_share:.2%}")
            
            if dataset_drift:
                st.error("🚨 DATASET DRIFT DETECTED!")
            else:
                st.success("✅ No significant dataset drift.")
                
            st.write("Feature Drift Details:")
            drift_by_columns = metrics["drift_by_columns"]
            drift_rows = []
            for col, stats in drift_by_columns.items():
                drift_rows.append({
                    "Feature": col,
                    "Drift Detected": stats["drift_detected"],
                    "Drift Score": stats["drift_score"]
                })
            st.table(pd.DataFrame(drift_rows))
            
            html_report = Path("reports/drift_report.html")
            if html_report.exists():
                st.download_button(
                    label="Download Full HTML Report",
                    data=open(html_report, "rb").read(),
                    file_name="drift_report.html",
                    mime="text/html"
                )
        except KeyError:
            st.warning("Could not parse drift summary JSON structure.")
            
    else:
        st.info("No drift report found. Run `python -m src.monitoring.drift_detector` to generate one.")
