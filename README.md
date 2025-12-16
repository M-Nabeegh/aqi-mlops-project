# End-to-End MLOps Pipeline for Financial Analytics

## Domain
**Economics & Finance**

---

## Project Overview
This project implements a **production-grade end-to-end Machine Learning system** for analyzing **Apple Inc. (AAPL)** stock behavior. It moves beyond simple static analysis by integrating real-time data ingestion, automated orchestration, and containerized deployment.

The system provides **Dual-Layer Intelligence**: predicting exact future returns and classifying market risk levels to aid in decision-making. The primary objective is to showcase **robust ML engineering workflows** (MLOps) used in professional industry environments.

---

## System Architecture
The system follows a microservices approach, decoupling data fetching, training, and serving into distinct, orchestrated layers.

![System Architecture Diagram](1.jpg)

---

## Machine Learning Tasks Covered
- **Time Series Analysis** – Historical OHLCV market data processing
- **Regression** – Predict next-day stock returns (Numeric)
- **Classification** – Categorize market risk state (Low / Medium / High)
- **Baseline Comparison** – Machine Learning performance vs. Mean predictor

---

## Tech Stack
- **Python 3.10**
- **XGBoost** (Regression Engine)
- **Random Forest** (Classification Engine)
- **FastAPI** (Real-time Model Serving)
- **Prefect** (Workflow Orchestration & DAGs)
- **Docker & Docker Compose** (Containerization)
- **GitHub Actions** (CI/CD Automation)
- **PyTest** (Automated Unit Testing)

---

## Data Source
- **Yahoo Finance API (`yfinance`)**
- Real-time and historical stock data (Open, High, Low, Close, Volume) for AAPL.
- **Resilience:** Includes automated retries and timeout handling for network stability.

---

## Feature Engineering
- **Moving Averages:** MA-10 (Short-term trend), MA-50 (Medium-term trend)
- **Momentum:** RSI (Relative Strength Index) to detect overbought/oversold conditions
- **Volatility:** Rolling Standard Deviation of returns
- **Target Engineering:** Time-shifted returns ($t+1$) to prevent look-ahead bias

---

## Model Details
### 1. Regression Model (Price Forecasting)
- **Algorithm:** XGBoost Regressor
- **Objective:** Minimize Root Mean Squared Error (RMSE)
- **Hyperparameters:** Tuned for tree depth and learning rate to prevent overfitting.
- **Output:** Continuous numeric value (Expected Return)

### 2. Classification Model (Risk Profiling)
- **Algorithm:** Random Forest Classifier
- **Objective:** Maximize Classification Accuracy (~85%)
- **Classes:** - `0`: Low Risk (Stable)
  - `1`: Medium Risk (Normal Volatility)
  - `2`: High Risk (High Volatility)
- **Strategy:** Uses representative sampling to handle class imbalance during high-volatility periods.

---

## Orchestration & Automation
### Prefect Workflow
- **Dependency Graph:** Enforces strict execution order: `Fetch` → `Feature Engineering` → `Train`.
- **Fault Tolerance:** If data ingestion fails, the pipeline halts immediately to prevent training on corrupt data.

### CI/CD Pipeline (GitHub Actions)
- **Trigger:** Automated run on every `push` to the `main` branch.
- **Steps:**
  1. Environment Setup (Python 3.10)
  2. Dependency Installation
  3. Data Generation (Fresh fetch)
  4. Model Retraining
  5. **Unit Testing (`pytest`):** Validates API endpoints and model artifact integrity before deployment.

---

## API Endpoints
### `/health`
- **Method:** GET
- **Purpose:** Checks system health and verifies model artifacts are loaded.

### `/predict`
- **Method:** POST
- **Purpose:** Real-time inference. Accepts a single JSON object with technical indicators and returns the predicted return and risk level.

### `/predict_batch`
- **Method:** POST
- **Purpose:** Batch processing. Accepts a `.csv` file upload containing historical data and returns predictions for all rows.
