import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load model directly — no API needed for deployment ────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("ml/churn_model.pkl")
    feature_names = joblib.load("ml/feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn risk and calculate retention ROI")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Customer Profile")

contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service",
    ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])

tenure_months = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges (£)", 0.0, 120.0, 65.0)
total_charges = monthly_charges * tenure_months

senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
    ["Yes", "No", "No phone service"])
online_security = st.sidebar.selectbox("Online Security",
    ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup",
    ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection",
    ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support",
    ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV",
    ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
    ["Yes", "No", "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# ── Predict button ────────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Predict Churn Risk", type="primary"):

    data = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_months,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": total_charges * 1.5,
        "AvgMonthlySpend": total_charges / (tenure_months + 1),
        "HighValue": 1 if monthly_charges > 64.76 else 0,
    }

    df = pd.DataFrame([data])
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    if tenure_months <= 12:
        df["TenureGroup"] = 0
    elif tenure_months <= 24:
        df["TenureGroup"] = 1
    elif tenure_months <= 48:
        df["TenureGroup"] = 2
    else:
        df["TenureGroup"] = 3

    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    roi = (monthly_charges - 20.0) if prediction == 1 else 0.0

    # ── Results ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("⚠️ Will Churn")
        else:
            st.success("✅ Will Stay")
        st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")

    with col2:
        st.metric("Churn Probability", f"{probability:.1%}")
        st.progress(probability)

    with col3:
        risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        st.metric("Risk Level", risk)
        st.metric("Retention ROI", f"£{roi:.2f}")

    st.divider()
    st.subheader("Business Insight")
    if prediction == 1:
        st.warning("⚠️ High churn risk — recommend immediate retention action")
    else:
        st.info("✅ Low churn risk — customer likely to stay")

    st.subheader("Customer Profile Summary")
    summary = {
        "Contract": contract,
        "Tenure": f"{tenure_months} months",
        "Monthly Charges": f"£{monthly_charges:.2f}",
        "Internet Service": internet_service,
        "Payment Method": payment_method,
    }
    st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

else:
    st.info("👈 Fill in the customer details in the sidebar and click "
            "**Predict Churn Risk**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("ROC-AUC Score", "0.85")
    with col3:
        st.metric("Training Data", "7,043 customers")