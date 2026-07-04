from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Optional

# ── Load model and feature names once at startup ──────────────────────────────
model = joblib.load("ml/churn_model.pkl")
feature_names = joblib.load("ml/feature_names.pkl")

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost",
    version="1.0.0"
)

# ── Add CORS middleware ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Define input schema with Pydantic ─────────────────────────────────────────
class CustomerData(BaseModel):
    tenure_months: float
    monthly_charges: float
    total_charges: float
    contract: str
    internet_service: str
    payment_method: str
    senior_citizen: int
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    paperless_billing: str
    gender: str
    cltv: Optional[float] = 0.0

# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "version": "1.0.0"}

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        data = {
            "Gender": customer.gender,
            "Senior Citizen": customer.senior_citizen,
            "Partner": customer.partner,
            "Dependents": customer.dependents,
            "Tenure Months": customer.tenure_months,
            "Phone Service": customer.phone_service,
            "Multiple Lines": customer.multiple_lines,
            "Internet Service": customer.internet_service,
            "Online Security": customer.online_security,
            "Online Backup": customer.online_backup,
            "Device Protection": customer.device_protection,
            "Tech Support": customer.tech_support,
            "Streaming TV": customer.streaming_tv,
            "Streaming Movies": customer.streaming_movies,
            "Contract": customer.contract,
            "Paperless Billing": customer.paperless_billing,
            "Payment Method": customer.payment_method,
            "Monthly Charges": customer.monthly_charges,
            "Total Charges": customer.total_charges,
            "CLTV": customer.cltv,
            "AvgMonthlySpend": customer.total_charges / (customer.tenure_months + 1),
            "HighValue": 1 if customer.monthly_charges > 64.76 else 0,
        }

        df = pd.DataFrame([data])
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = le.fit_transform(df[col].astype(str))

        tenure = customer.tenure_months
        if tenure <= 12:
            df["TenureGroup"] = 0
        elif tenure <= 24:
            df["TenureGroup"] = 1
        elif tenure <= 48:
            df["TenureGroup"] = 2
        else:
            df["TenureGroup"] = 3

        df = df.reindex(columns=feature_names, fill_value=0)

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        monthly_revenue = customer.monthly_charges
        retention_cost = 20.0
        roi = (monthly_revenue - retention_cost) if prediction == 1 else 0.0

        return {
            "churn_prediction": prediction,
            "churn_label": "Will Churn" if prediction == 1 else "Will Stay",
            "churn_probability": round(probability, 4),
            "risk_level": (
                "High" if probability > 0.7
                else "Medium" if probability > 0.4
                else "Low"
            ),
            "retention_roi": round(roi, 2),
            "message": (
                "⚠️ High churn risk — recommend immediate retention action"
                if prediction == 1
                else "✅ Low churn risk — customer likely to stay"
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))