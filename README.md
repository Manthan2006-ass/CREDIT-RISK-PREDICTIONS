AI-Powered Credit Risk Scoring System
An end-to-end, decoupled microservice system designed to assess loan default risk in real time. The platform features an Ensemble Soft-Voting ML Pipeline, a FastAPI Inference Microservice, a Streamlit Dashboard, SHAP Tree Explainability, and an automated MySQL Audit Store.

🏗️ System Architecture
The project follows a production-grade decoupled microservice architecture separating the user interface, inference engine, data validation layer, and persistent database store.
┌─────────────────────────┐          HTTP POST (JSON)          ┌──────────────────────────────────┐
│  Streamlit Dashboard    │ ─────────────────────────────────► │    FastAPI Inference Engine      │
│     (Frontend UI)       │ ◄───────────────────────────────── │    (Pydantic Schema Validation)  │
└─────────────────────────┘          Prediction & SHAP         └──────────────────────────────────┘
                                                                               │
                                                                   SQLAlchemy Connection Pool
                                                                               │
                                                                               ▼
                                                                  ┌─────────────────────────┐
                                                                  │      MySQL Server       │
                                                                  │   (loan_predictions DB) │
                                                                  └─────────────────────────┘


Key Components
Frontend (Streamlit): Interactive user interface for loan officers to enter applicant parameters, view real-time default probabilities, and analyze feature importance.

Backend (FastAPI): Asynchronous REST microservice that runs validation rules via Pydantic, executes model inference, generates local SHAP feature contributions, and logs records to MySQL.

Database (MySQL): Persistent storage tracking every loan evaluation payload, probability output, and final decision for compliance auditing.

Secrets Management (.env): Configuration using python-dotenv to keep database credentials out of version control.
Tech Stack & Dependencies
Language: Python 3.9+

Machine Learning: scikit-learn, lightgbm, xgboost, shap

API Framework: fastapi, uvicorn, pydantic

Database & ORM: MySQL, SQLAlchemy, pymysql

Frontend UI: streamlit, matplotlib

Environment & Utilities: python-dotenv, joblib, pandas, numpy
