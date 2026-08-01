import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df = pd.read_csv("application_train.csv")

df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace(0, np.nan)
ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
df['EXT_SOURCES_AVG'] = df[ext_cols].mean(axis=1)

missing_pct = df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.50].index.tolist()
cols_to_drop = [c for c in cols_to_drop if c not in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
cols_to_drop.append('SK_ID_CURR')
df_clean = df.drop(columns=list(set(cols_to_drop)), errors='ignore')

X = df_clean.drop(columns=['TARGET'])
y = df_clean['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

initial_preprocessor = ColumnTransformer(transformers=[('num', Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', RobustScaler())]), num_cols),('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])

X_train_proc = initial_preprocessor.fit_transform(X_train)
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_proc, y_train)

importances = rf_selector.feature_importances_
top_feature_indices = np.argsort(importances)[::-1][:20]

cat_encoder = initial_preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features_out = list(cat_encoder.get_feature_names_out(cat_cols))
all_feature_names = np.array(num_cols + cat_features_out)

selected_features = all_feature_names[top_feature_indices]
top_raw_cols = list(set([
    col.split('_')[0] if '_' in col and col.split('_')[0] in X.columns else col 
    for col in selected_features
]))
top_raw_cols = [c for c in top_raw_cols if c in X.columns]

X_train_top = X_train[top_raw_cols]
X_test_top = X_test[top_raw_cols]

top_num_cols = X_train_top.select_dtypes(include=['float64', 'int64']).columns.tolist()
top_cat_cols = X_train_top.select_dtypes(include=['object', 'category']).columns.tolist()

top_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), top_num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), top_cat_cols)
])

clf_lgbm = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
clf_xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf_rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

ensemble_voting = VotingClassifier(
    estimators=[
        ('lgbm', clf_lgbm),
        ('xgb', clf_xgb),
        ('rf', clf_rf)
    ],
    voting='soft'
)

final_production_model = Pipeline(steps=[
    ('pre', top_preprocessor),
    ('model', ensemble_voting)
])

final_production_model.fit(X_train_top, y_train)
y_probs = final_production_model.predict_proba(X_test_top)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

thresholds = np.linspace(0.1, 0.9, 81)
costs = []

for th in thresholds:
    preds = (y_probs >= th).astype(int)
    cm = confusion_matrix(y_test, preds)
    fn, fp = cm[1, 0], cm[0, 1]
    costs.append(fn * 5.0 + fp * 1.0)

optimal_threshold = thresholds[np.argmin(costs)]
print(f"Optimal Threshold: {optimal_threshold:.2f}")

opt_preds = (y_probs >= optimal_threshold).astype(int)
print(classification_report(y_test, opt_preds))

joblib.dump(final_production_model, "final_credit_scoring_model.pkl")
