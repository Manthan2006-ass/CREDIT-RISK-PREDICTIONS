import os
import warnings
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, PositiveFloat
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "credit_risk_db")

MYSQL_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI(
    title="Credit Risk Inference Engine",
    version="2.0.0"
)

engine = create_engine(
    MYSQL_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "loan_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    income = Column(Float, nullable=False)
    credit_amount = Column(Float, nullable=False)
    annuity = Column(Float, nullable=False)
    risk_probability = Column(Float, nullable=False)
    decision = Column(String(20), nullable=False)


MODEL_PATH = "final_credit_scoring_model.pkl"
model = None


@app.on_event("startup")
def startup():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        Base.metadata.create_all(bind=engine)
    except Exception as err:
        print(f"Initialization error: {err}")


class ApplicantPayload(BaseModel):
    income: PositiveFloat = Field(...)
    credit: PositiveFloat = Field(...)
    annuity: PositiveFloat = Field(...)
    age: int = Field(..., ge=18, le=70)
    emp_years: int = Field(..., ge=0, le=45)
    goods_price: Optional[float] = Field(0.0, ge=0.0)
    ext_2: float = Field(0.5, ge=0.0, le=1.0)
    ext_3: float = Field(0.5, ge=0.0, le=1.0)
    education: str
    income_type: str


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    decision: str
    risk_level: str
    processed_at: str


class SHAPFeatureImpact(BaseModel):
    feature: str
    impact: float


class SHAPResponse(BaseModel):
    top_features: List[SHAPFeatureImpact]


def preprocess_payload(payload: ApplicantPayload) -> pd.DataFrame:
    safe_income = payload.income if payload.income > 0 else np.nan
    safe_credit = payload.credit if payload.credit > 0 else np.nan

    days_birth = -int(payload.age * 365.25)
    days_employed = -int(payload.emp_years * 365.25)

    raw_dict = {
        'AMT_INCOME_TOTAL': payload.income,
        'AMT_CREDIT': payload.credit,
        'AMT_ANNUITY': payload.annuity,
        'AMT_GOODS_PRICE': payload.goods_price if payload.goods_price > 0 else np.nan,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'EXT_SOURCE_1': np.nan,
        'EXT_SOURCE_2': payload.ext_2,
        'EXT_SOURCE_3': payload.ext_3,
        'CREDIT_INCOME_PERCENT': safe_credit / safe_income if safe_income else np.nan,
        'ANNUITY_INCOME_PERCENT': payload.annuity / safe_income if safe_income else np.nan,
        'CREDIT_TERM': payload.annuity / safe_credit if safe_credit else np.nan,
        'DAYS_EMPLOYED_PERCENT': days_employed / days_birth if days_birth != 0 else np.nan,
        'EXT_SOURCES_AVG': float(np.nanmean([payload.ext_2, payload.ext_3])),
        'NAME_EDUCATION_TYPE': payload.education,
        'NAME_INCOME_TYPE': payload.income_type,
        'REGION_RATING_CLIENT': 2,
        'FLAG_DOCUMENT_3': 1,
        'DEF_30_CNT_SOCIAL_CIRCLE': 0,
        'DEF_60_CNT_SOCIAL_CIRCLE': 0,
        'REG_CITY_NOT_LIVE_CITY': 0
    }

    input_df = pd.DataFrame([raw_dict])

    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[expected_cols]

    return input_df


def log_to_mysql(income: float, credit: float, annuity: float, prob: float, decision: str):
    db = SessionLocal()
    try:
        log_entry = PredictionLog(
            income=income,
            credit_amount=credit,
            annuity=annuity,
            risk_probability=prob,
            decision=decision
        )
        db.add(log_entry)
        db.commit()
    except Exception as err:
        db.rollback()
        print(f"DB Log Error: {err}")
    finally:
        db.close()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "database": "MySQL", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_credit_risk(payload: ApplicantPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Model uninitialized")

    try:
        input_df = preprocess_payload(payload)
        prob = float(model.predict_proba(input_df)[0][1])
        threshold = 0.30
        is_rejected = prob > threshold
        decision = "REJECTED" if is_rejected else "APPROVED"

        risk_level = "Safe"
        if is_rejected:
            risk_level = "Very High" if prob > 0.60 else "Moderate"

        log_to_mysql(payload.income, payload.credit, payload.annuity, prob, decision)

        return PredictionResponse(
            probability=round(prob, 4),
            threshold=threshold,
            decision=decision,
            risk_level=risk_level,
            processed_at=datetime.now().isoformat()
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/explain", response_model=SHAPResponse, status_code=status.HTTP_200_OK)
def explain_decision(payload: ApplicantPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Model uninitialized")

    try:
        input_df = preprocess_payload(payload)
        preprocessor = model.named_steps['pre']
        voting_clf = model.named_steps['model']

        X_transformed = preprocessor.transform(input_df)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except AttributeError:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]

        shap_vectors = []
        for _, estimator in voting_clf.named_estimators_.items():
            explainer = shap.TreeExplainer(estimator)
            vals = explainer.shap_values(X_transformed)
            if isinstance(vals, list):
                vec = vals[1][0]
            elif len(vals.shape) == 3:
                vec = vals[0, :, 1]
            else:
                vec = vals[0]
            shap_vectors.append(vec)

        avg_shap_vector = np.mean(shap_vectors, axis=0)

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Impact': avg_shap_vector
        }).sort_values(by='SHAP Impact', key=abs, ascending=False).head(10)

        results = [
            SHAPFeatureImpact(feature=row['Feature'], impact=round(float(row['SHAP Impact']), 4))
            for _, row in shap_df.iterrows()
        ]

        return SHAPResponse(top_features=results)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
