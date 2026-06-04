# AI-Powered Credit Risk Scoring

This project demonstrates a complete credit risk modeling workflow using `application_train.csv` and the notebook `creditscore chnage.ipynb`.

## Project Overview
- Dataset: `application_train.csv`
- Notebook: `creditscore chnage.ipynb`
- Application: `app.py`
- Exported model: `final_credit_scoring_model.pkl`
- Dependencies: `requirements.txt`

## Notebook Workflow

### 1. Data Loading and Inspection
- Loaded the raw dataset into a pandas DataFrame.
- Reviewed shape, columns, summary statistics, and missing values.
- Checked data types and target distribution for early data quality insight.

### 2. Exploratory Data Analysis
- Visualized the class balance of the `TARGET` label.
- Inspected missing-value patterns and the columns with the highest missing rates.
- Examined distributions of `AMT_INCOME_TOTAL` and `AMT_CREDIT` to identify skew and outliers.

### 3. Data Cleaning
- Dropped features with more than 50% missing values, while retaining key external scoring fields `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`.
- Removed the identifier column `SK_ID_CURR` to avoid leakage.
- Replaced placeholder invalid employment values (`365243`) with `NaN` for correct imputation.

### 4. Feature Engineering
- Created derived risk features:
  - `CREDIT_INCOME_PERCENT`
  - `ANNUITY_INCOME_PERCENT`
  - `CREDIT_TERM`
  - `DAYS_EMPLOYED_PERCENT`
  - `EXT_SOURCES_AVG`
- Checked numerical features for outliers to support robust preprocessing.

### 5. Preprocessing and Split
- Performed train/test split with stratification on the target.
- Defined preprocessing pipelines:
  - numeric features: `SimpleImputer` + `RobustScaler`
  - categorical features: `SimpleImputer` + `OneHotEncoder`

### 6. Model Training and Evaluation
- Trained candidate models using a pipeline:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluated model performance using ROC-AUC for imbalanced classification.
- Visualized XGBoost feature importance and selected top predictive features.

### 7. Model Selection and Ensemble
- Tuned XGBoost hyperparameters with `RandomizedSearchCV`.
- Built a soft voting ensemble combining:
  - XGBoost
  - LightGBM
  - Logistic Regression
- Trained the final production pipeline and exported it to `final_credit_scoring_model.pkl`.

### 8. Performance Metrics
- Computed key metrics:
  - ROC-AUC
  - Gini coefficient
  - KS statistic
- Evaluated model performance across several classification thresholds.
- Generated classification reports and confusion matrices for chosen thresholds.

### 9. Explainability
- Prepared the preprocessed test set for SHAP explainability analysis.
- Set up SHAP for model interpretation and feature impact analysis.

## Application
- `app.py` serves a Streamlit dashboard for live credit risk prediction.
- The app loads the trained pipeline and accepts user inputs for applicant financial attributes.
- It displays a risk probability, decision recommendation, and visual progress indicator.

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- Streamlit
- SHAP

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Quick Start

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Alternative Installation (using setup.py)
```bash
pip install -e .
streamlit run app.py
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook `creditscore_change.ipynb` to reproduce the analysis and model training
3. Launch the dashboard: `streamlit run app.py`
4. Navigate to `http://localhost:8501` in your browser

## Running the Dashboard
https://credit-prediction-u9ag77zwllykg7bo2sew6d.streamlit.app/

## Notes
- The notebook is structured for reproducibility and clarity across data cleaning, model building, and evaluation.
- The exported model supports re-use in production and deployment workflows.
