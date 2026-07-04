import streamlit as st
import requests
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
# This must be the first Streamlit command in the file
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── API URL ───────────────────────────────────────────────────────────────────
# This is where our FastAPI backend is running
# The frontend talks to the backend via HTTP requests — separated architecture
API_URL = "http://127.0.0.1:8000"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn risk and calculate retention ROI")
st.divider()

# ── Sidebar — customer input form ─────────────────────────────────────────────
# All user inputs go in the sidebar — keeps the main area clean for results
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Enter customer details below")

# Contract and service details
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)",
     "Credit card (automatic)"]
)

# Numeric inputs
tenure_months = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges (£)", 0.0, 120.0, 65.0)
total_charges = monthly_charges * tenure_months

# Customer demographics
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Services
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No",
                                                          "No phone service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No",
                                                            "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No",
                                                        "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No",
                                                                "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No",
                                                      "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No",
                                                      "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No",
                                                              "No internet service"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# ── Predict button ────────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Predict Churn Risk", type="primary"):

    # Build the payload to send to our FastAPI backend
    payload = {
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "paperless_billing": paperless_billing,
        "gender": gender,
        "cltv": total_charges * 1.5
    }

    # Send POST request to FastAPI backend
    # This is the separated architecture — frontend calls backend via API
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()

        # ── Results display ───────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        # Churn prediction
        with col1:
            if result["churn_prediction"] == 1:
                st.error("⚠️ Will Churn")
            else:
                st.success("✅ Will Stay")
            st.metric("Prediction", result["churn_label"])

        # Churn probability
        with col2:
            prob = result["churn_probability"]
            st.metric("Churn Probability", f"{prob:.1%}")
            st.progress(prob)

        # Risk level and ROI
        with col3:
            st.metric("Risk Level", result["risk_level"])
            st.metric("Retention ROI", f"£{result['retention_roi']:.2f}")

        st.divider()

        # ── Business insight ──────────────────────────────────────────────────
        st.subheader("Business Insight")
        st.info(result["message"])

        # ── Customer summary table ────────────────────────────────────────────
        st.subheader("Customer Profile Summary")
        summary = {
            "Contract": contract,
            "Tenure": f"{tenure_months} months",
            "Monthly Charges": f"£{monthly_charges:.2f}",
            "Internet Service": internet_service,
            "Payment Method": payment_method,
        }
        st.table(pd.DataFrame(summary.items(),
                              columns=["Feature", "Value"]))

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
        st.info("Make sure the FastAPI backend is running on port 8000")

else:
    # Default state — show instructions
    st.info("👈 Fill in the customer details in the sidebar and click "
            "**Predict Churn Risk**")

    # Show some context about the model
    st.subheader("About This Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("ROC-AUC Score", "0.85")
    with col3:
        st.metric("Training Data", "7,043 customers")

    st.markdown("""
    **How it works:**
    1. Enter customer details in the sidebar
    2. Click Predict Churn Risk
    3. Get instant churn probability and retention ROI
    4. Use insights to prioritise retention interventions
    """)