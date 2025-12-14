# End-to-End Financial Time Series Prediction System (MLOps)

## Domain
**Economics & Finance**

---

## Project Overview
This project implements a **production-grade end-to-end Machine Learning system** for predicting **next-day stock returns** using historical market data and technical indicators.  

The system demonstrates **real-world MLOps practices**, including data ingestion, feature engineering, model training, CI/CD automation, deployment, and monitoring.

The primary objective is not only predictive modeling, but also to showcase **robust ML engineering workflows** used in industry.

---

## Machine Learning Tasks Covered
- **Time Series Analysis** – Stock price & return modeling  
- **Regression** – Predict next-day stock returns  
- **Classification (Derived)** – BUY / SELL signal generation  
- **Baseline Comparison** – Mean predictor vs ML model  

---

## Tech Stack
- **Python**
- **XGBoost**
- **Pandas / NumPy**
- **FastAPI** (Model Serving)
- **GitHub Actions** (CI/CD)
- **Docker** (Containerization)
- **Evidently AI** (Drift Monitoring – optional)
- **PyTest** (Automated Testing)

---

## Data Source
- **Yahoo Finance API**
- Multi-year historical stock data (OHLCV)

---

## Feature Engineering
- Moving Averages (MA-10, MA-50)
- Volatility (Rolling Std of Returns)
- RSI (Relative Strength Index)
- Return-based targets

---

## Model Training
- **Model:** XGBoost Regressor
- **Evaluation Metric:** RMSE
- **Baseline:** Mean-return predictor
- **Time-series aware split** (no data leakage)

> Financial returns are noisy by nature; baseline comparison is used to validate learning effectiveness.

---

## API Endpoints
### Health Check