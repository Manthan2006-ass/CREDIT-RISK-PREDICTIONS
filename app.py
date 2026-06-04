import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# --- NEW: SUPPRESS VERSION & FEATURE WARNINGS ---
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Credit Risk Dashboard",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    
    return joblib.load("final_credit_scoring_model.pkl")

model = load_model()

# ---------------- HEADER ----------------
st.title(" AI-Powered Credit Scoring Dashboard")
st.caption("Instant AI-based Credit Risk Prediction")

st.divider()

# ---------------- APPLICANT DETAILS ----------------
st.subheader(" Applicant Assessment")

col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:
    income = st.number_input("Annual Income (₹)", min_value=1, value=600000)
    annuity = st.number_input("Monthly EMI (₹)", min_value=1, value=30000)
    age = st.slider("Age", 18, 70, 35)
    ext_2 = st.slider("Credit Score (Repayment Trust)", 0.0, 1.0, 0.5, help="Higher value means better repayment history")
    education = st.selectbox(
        "Education",
        ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"]
    )

# RIGHT COLUMN
with col2:
    credit = st.number_input("Loan Amount (₹)", min_value=1, value=1200000)
    emp_years = st.slider("Years Employed", 0, 45, 10)
    ext_3 = st.slider("Risk Score (Default Risk)", 0.0, 1.0, 0.5, help="Higher value means lower default risk")
    income_type = st.selectbox(
        "Income Source ",
        ["Working", "Commercial associate", "State servant", "Pensioner"]
    )

st.divider()

# ---------------- INPUT DATA ----------------
input_dict = {
    "EXT_SOURCE_2": ext_2,
    "EXT_SOURCE_3": ext_3,
    "AMT_GOODS_PRICE": credit * 0.92,
    "DAYS_BIRTH": -age * 365,
    "DAYS_EMPLOYED": -emp_years * 365,
    "AMT_CREDIT": credit,
    "AMT_INCOME_TOTAL": income,
    "AMT_ANNUITY": annuity,
    "REGION_RATING_CLIENT": 2,
    "FLAG_DOCUMENT_3": 1,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "NAME_INCOME_TYPE": income_type,
    "NAME_EDUCATION_TYPE": education,
    "CODE_GENDER": "M",
    "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_CAR": "N",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "OCCUPATION_TYPE": "Laborers"
}

# ---------------- FEATURE ENGINEERING ----------------
input_dict['EXT_SOURCES_AVG'] = (ext_2 + ext_3 + 0.5) / 3
input_dict['CREDIT_TERM'] = annuity / credit
input_dict['DAYS_EMPLOYED_PERCENT'] = input_dict['DAYS_EMPLOYED'] / input_dict['DAYS_BIRTH']
input_dict['CREDIT_INCOME_PERCENT'] = credit / income
input_dict['ANNUITY_INCOME_PERCENT'] = annuity / income

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if st.button("Run Risk Assessment", use_container_width=True):
    try:
        # Validate critical inputs before prediction
        if credit <= 0 or income <= 0 or annuity <= 0:
            st.error("Income, loan amount, and annuity must all be greater than zero.")
        else:
            # --- New: FEATURE ALIGNMENT ---
            # This fixes the "indices imply 109" error by ensuring 
            # input_df has every column the model saw during training.
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                # Fill missing columns with 0 and reorder them correctly
                input_df = input_df.reindex(columns=expected_features, fill_value=0)

            # Get prediction probability
            prob = model.predict_proba(input_df)[0][1]

            # 2. Display Results
            threshold = 0.3
            is_rejected = prob > threshold

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Probability", f"{prob*100:.1f}%")
            col2.metric("Decision Threshold", f"{threshold*100:.0f}%")
            col3.metric("Status", "REJECTED" if is_rejected else "APPROVED")

            # ---------------- PROGRESS BAR ----------------
            st.subheader("Risk Visualization")
            st.progress(float(prob))

            # ---------------- FINAL RESULT ----------------
            if is_rejected:
                st.error(f"Default risk exceeds the bank's safe threshold of {threshold*100:.0f}%.")
                st.warning(" Very High Risk Applicant" if prob > 0.6 else "Moderate Risk Applicant")
            else:
                st.success(" Applicant falls within acceptable credit risk limits.")
                st.success(" Low Risk Applicant")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the column names in 'input_dict' match the CSV headers exactly.")