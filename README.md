# Customer Churn Prediction API

![CI Pipeline](https://github.com/malavikajiyer/churn-prediction-api/actions/workflows/ci.yml/badge.svg)

A production-grade machine learning system that predicts customer churn probability and calculates retention ROI : built with FastAPI, XGBoost, Docker, and GitHub Actions CI/CD.

**[Live App](https://churn-prediction-api-wenbjjbsuwcp4q2wplszzz.streamlit.app)**

---

## What it does

Input a customer profile → get a churn probability, risk level, and retention ROI calculation. Built to mirror real-world ML deployment: separated backend/frontend architecture, automated testing, and containerised deployment.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.85 |
| Churn Recall | 79% |
| Training Data | 7,043 customers |
| Tests | 6 passing |

---

## Architecture

\```
Streamlit Frontend (port 8501)
        ↓ HTTP POST /predict
FastAPI Backend (port 8000)
        ↓ loads
XGBoost Model (ml/churn_model.pkl)
        ↑ trained by
ml/train.py (runs in CI before every test)
\```

---

## Tech Stack

- **ML:** XGBoost, Scikit-learn, SHAP, Pandas, NumPy
- **API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **Testing:** pytest, httpx
- **DevOps:** Docker, GitHub Actions CI/CD

---

## Run locally

```bash
# Clone
git clone https://github.com/malavikajiyer/churn-prediction-api.git
cd churn-prediction-api

# Install
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Train model
python ml/train.py

# Start API (terminal 1)
uvicorn app.main:app --reload

# Start dashboard (terminal 2)
streamlit run streamlit_app.py
```

## API Docs
Visit `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

---

## Key Features

- **ROI Calculator** — quantifies business value of each retention intervention
- **Risk levels** — High/Medium/Low classification with probability score
- **Separated architecture** — FastAPI backend consumable by any client
- **CI/CD** — GitHub Actions runs 6 automated tests on every push
- **Docker** — runs anywhere with `docker build -t churn-api . && docker run -p 8000:8000 churn-api`
