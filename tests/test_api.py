import pytest
from fastapi.testclient import TestClient
import sys
import os

# This adds the project root to Python's path so it can find app/main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# TestClient lets us test the API without running a real server
# This is how professional teams test APIs — fast, automated, no manual clicking
client = TestClient(app)

# ── Sample customer data ───────────────────────────────────────────────────────
# We define this once and reuse it across tests
HIGH_RISK_CUSTOMER = {
    "tenure_months": 2,
    "monthly_charges": 85.0,
    "total_charges": 170.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "paperless_billing": "Yes",
    "gender": "Female",
    "cltv": 2000.0
}

LOW_RISK_CUSTOMER = {
    "tenure_months": 60,
    "monthly_charges": 45.0,
    "total_charges": 2700.0,
    "contract": "Two year",
    "internet_service": "DSL",
    "payment_method": "Bank transfer (automatic)",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "Yes",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "paperless_billing": "No",
    "gender": "Male",
    "cltv": 5000.0
}

# ── Test 1: Root endpoint returns correct message ─────────────────────────────
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "running" in response.json()["message"]

# ── Test 2: Health endpoint returns healthy status ────────────────────────────
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# ── Test 3: Prediction returns correct structure ──────────────────────────────
def test_predict_returns_correct_fields():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    # Check all expected fields are present
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "retention_roi" in data
    assert "churn_label" in data
    assert "message" in data

# ── Test 4: Churn probability is between 0 and 1 ─────────────────────────────
def test_churn_probability_range():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0

# ── Test 5: Invalid data returns 422 validation error ────────────────────────
def test_invalid_input_returns_422():
    # Send completely wrong data — missing required fields
    response = client.post("/predict", json={"wrong_field": "wrong_value"})
    assert response.status_code == 422

# ── Test 6: Low risk customer scores lower than high risk customer ────────────
def test_low_risk_scores_lower_than_high_risk():
    high_response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    low_response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    high_prob = high_response.json()["churn_probability"]
    low_prob = low_response.json()["churn_probability"]
    # A loyal 5-year customer on a 2-year contract should score lower
    # than a new month-to-month customer — this validates model logic
    assert low_prob < high_prob