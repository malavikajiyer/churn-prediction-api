import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
# read_excel loads our .xlsx file into a pandas DataFrame
# a DataFrame is like a table — rows are customers, columns are their attributes
df = pd.read_excel("data/Telco_customer_churn.xlsx", engine="openpyxl")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ── 2. Drop columns we don't need ─────────────────────────────────────────────
# CustomerID is just a label — not useful for prediction
# Churn Score/Reason would leak the answer to the model — that's cheating
# Location data is too granular to be useful
drop_cols = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Score", "Churn Reason", "Churn Value"
]
df = df.drop(columns=drop_cols)

# ── 3. Clean data ─────────────────────────────────────────────────────────────
# Total Charges has some blank strings — pd.to_numeric converts them to NaN
# then fillna(0) replaces NaN with 0
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(0)

# ── 4. Feature engineering ────────────────────────────────────────────────────
# We create NEW columns from existing ones to give the model more signal

# Average monthly spend — total divided by how long they've been a customer
# This shows value per month, not just raw total
df["AvgMonthlySpend"] = df["Total Charges"] / (df["Tenure Months"] + 1)

# Tenure group — bucket customers into time ranges
# Short term customers behave very differently from long term ones
df["TenureGroup"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
)

# High value flag — 1 if they pay above median, 0 if below
# These customers are most important to retain
median_charges = df["Monthly Charges"].median()
df["HighValue"] = (df["Monthly Charges"] > median_charges).astype(int)

# ── 5. Encode categorical columns ─────────────────────────────────────────────
# ML models only understand numbers — not text like "Yes", "No", "Male"
# LabelEncoder converts each unique text value to a number
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
categorical_cols = [c for c in categorical_cols if c != "Churn Label"]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target column: Yes = 1 (churned), No = 0 (stayed)
df["Churn Label"] = (df["Churn Label"] == "Yes").astype(int)
print(f"Churn rate: {df['Churn Label'].mean():.1%} of customers churned")

# ── 6. Split into features and target ─────────────────────────────────────────
# X = all the columns the model learns FROM
# y = the column we want the model to PREDICT
X = df.drop(columns=["Churn Label"])
y = df["Churn Label"]

# train_test_split: 80% for training, 20% for testing
# stratify=y ensures both splits have the same churn ratio
# random_state=42 means we get the same split every time we run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

# ── 7. Train XGBoost model ────────────────────────────────────────────────────
# XGBoost is a gradient boosting model — one of the best for tabular data
# scale_pos_weight corrects for imbalance: more non-churners than churners
# Without this the model would just predict "no churn" for everyone
churn_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=churn_ratio,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
print("Model trained successfully")

# ── 8. Evaluate the model ─────────────────────────────────────────────────────
# predict() gives us 0 or 1 for each customer
# predict_proba() gives us the probability — we use column 1 (churn probability)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Model Performance ──")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ── 9. Save model and feature names ──────────────────────────────────────────
# joblib saves the trained model as a file
# FastAPI will load this file later to make predictions without retraining
os.makedirs("ml", exist_ok=True)
joblib.dump(model, "ml/churn_model.pkl")
joblib.dump(list(X.columns), "ml/feature_names.pkl")
print("\nModel saved to ml/churn_model.pkl")
print("Feature names saved to ml/feature_names.pkl")