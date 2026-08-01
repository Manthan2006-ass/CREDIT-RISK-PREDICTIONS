# AI-Powered Credit Risk Scoring System
An end-to-end, decoupled microservice system designed to assess loan default risk in real time. The platform features an Ensemble Soft-Voting ML Pipeline, a FastAPI Inference Microservice, a Streamlit Dashboard, SHAP Tree Explainability, and an automated MySQL Audit Store.

## System Architecture
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


### Component Breakdown

* **Frontend Client (`app.py`):** Interactive UI for financial officers to input borrower parameters, observe real-time approval decisions, and inspect local feature attribution graphs.
* **Inference Backend (`api.py`):** Asynchronous FastAPI engine enforcing type coercion via Pydantic schemas, executing model inference, generating SHAP explanations, and writing audit logs.
* **Audit Store (MySQL):** Database backing storing prediction timestamps, input financial payloads, output risk probabilities, and final approval decisions for compliance audits.
* **Secrets Handler (`.env`):** Configuration management via `python-dotenv` preventing credential exposure in version control.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.9+
* **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`, `shap`
* **API Framework:** `fastapi`, `uvicorn`, `pydantic`
* **Database & ORM:** `MySQL`, `SQLAlchemy`, `pymysql`
* **UI & Data Viz:** `streamlit`, `matplotlib`
* **Utilities:** `python-dotenv`, `joblib`, `pandas`, `numpy`

---

## 📂 Repository Layout

```text
├── api.py                            # FastAPI microservice & SQLAlchemy ORM schemas
├── app.py                            # Streamlit frontend client application
├── train_model.py                    # Pipeline training, feature selection, & model export
├── final_credit_scoring_model.pkl    # Serialized VotingClassifier pipeline artifact
├── application_train.csv             # Credit training dataset (Home Credit Benchmark)
├── requirements.txt                  # Environment dependencies
├── .env                              # Credentials & database configuration (Local)
├── .gitignore                        # Version control exclusion rules
└── README.md                         # Project documentation

Environment & Utilities: python-dotenv, joblib, pandas, numpy
