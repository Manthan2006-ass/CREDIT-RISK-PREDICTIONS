import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "credit_risk_db")

MYSQL_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

st.set_page_config(
    page_title="AI Credit Risk Assessment Dashboard",
    layout="wide"
)

st.title("AI-Powered Credit Scoring Dashboard")
st.caption("Decoupled Architecture: Streamlit UI → FastAPI Backend → MySQL Audit Log")

st.divider()

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code != 200:
        st.error("Backend health check failed.")
except Exception:
    st.warning("FastAPI service unreachable at http://127.0.0.1:8000")

st.subheader("Applicant Information Input")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Annual Income (₹)", min_value=1.0, value=600000.0, step=10000.0)
    credit = st.number_input("Requested Loan Amount (₹)", min_value=1.0, value=1200000.0, step=20000.0)
    annuity = st.number_input("Monthly EMI / Annuity (₹)", min_value=1.0, value=30000.0, step=1000.0)

with col2:
    age = st.slider("Age (Years)", 18, 70, 35)
    emp_years = st.slider("Years Employed", 0, 45, 10)
    goods_price = st.number_input("Goods Price (₹)", min_value=0.0, value=1000000.0, step=10000.0)

with col3:
    ext_2 = st.slider("External Score 2 (Repayment Trust)", 0.0, 1.0, 0.5)
    ext_3 = st.slider("External Score 3 (Default Risk)", 0.0, 1.0, 0.5)
    education = st.selectbox(
        "Education Level",
        ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"]
    )
    income_type = st.selectbox(
        "Income Source",
        ["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"]
    )

st.divider()

if st.button(" Evaluate Credit Risk", use_container_width=True):
    payload = {
        "income": income,
        "credit": credit,
        "annuity": annuity,
        "age": age,
        "emp_years": emp_years,
        "goods_price": goods_price,
        "ext_2": ext_2,
        "ext_3": ext_3,
        "education": education,
        "income_type": income_type
    }

    with st.spinner("Processing evaluation..."):
        try:
            pred_response = requests.post(f"{API_URL}/predict", json=payload)

            if pred_response.status_code == 422:
                st.error("Validation Error: Invalid input parameters.")
                st.json(pred_response.json())
                st.stop()
            elif pred_response.status_code != 200:
                st.error(f"Server Error: {pred_response.text}")
                st.stop()

            res_data = pred_response.json()
            prob = res_data["probability"]
            threshold = res_data["threshold"]
            decision = res_data["decision"]
            risk_level = res_data["risk_level"]

            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Predicted Default Probability", f"{prob * 100:.1f}%")
            col_res2.metric("Approval Risk Threshold", f"{threshold * 100:.0f}%")
            col_res3.metric("Decision Output", "REJECTED " if decision == "REJECTED" else "APPROVED ")

            st.progress(prob)

            if decision == "REJECTED":
                st.error(f"Application Decision: REJECTED. Default risk ({prob*100:.1f}%) exceeds threshold ({threshold*100:.0f}%).")
                st.warning(f"Risk Level: {risk_level}")
            else:
                st.success("Application Decision: APPROVED. Borrower parameters meet criteria.")

            exp_response = requests.post(f"{API_URL}/explain", json=payload)
            if exp_response.status_code == 200:
                st.divider()
                st.subheader(" Local SHAP Feature Contributions")

                top_features = exp_response.json()["top_features"]
                shap_df = pd.DataFrame(top_features)

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in shap_df['impact'][::-1]]
                ax.barh(shap_df['feature'][::-1], shap_df['impact'][::-1], color=colors)
                ax.set_xlabel("Impact on Default Score")
                ax.set_title("Top 10 Feature Drivers")
                st.pyplot(fig)

        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to backend service.")

st.divider()
with st.expander("View Audit Log (MySQL Data)"):
    try:
        engine = create_engine(MYSQL_DATABASE_URL)
        history_df = pd.read_sql_query("SELECT * FROM loan_predictions ORDER BY id DESC LIMIT 10", engine)
        st.dataframe(history_df, use_container_width=True)
    except Exception:
        st.info("Unable to read history table from MySQL.")
