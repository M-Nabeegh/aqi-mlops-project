from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import pandas as pd
import io

app = FastAPI(title="Apple Stock Prediction API (Regression + Classification)")

# ==========================================
# 1. LOAD MODELS (UPDATED)
# ==========================================
# The model.pkl is now a dictionary containing:
# - "regressor": The XGBoost model (predicts return)
# - "classifier": The Random Forest model (predicts risk)
# - "features": The list of column names expected
artifacts = joblib.load("model.pkl")

reg_model = artifacts["regressor"]
class_model = artifacts["classifier"]
model_features = artifacts["features"]

# Helper mapping for risk
risk_map = {0: "Low", 1: "Medium", 2: "High"}

@app.get("/health")
def health():
    return {"status": "running", "models_loaded": ["Regressor", "Classifier"]}

# ==========================================
# 2. JSON ENDPOINT (Single Prediction)
# ==========================================
@app.post("/predict")
def predict(data: dict):
    """
    Input: JSON with features (ma_10, ma_50, volatility, rsi)
    Output: Predicted Return (Numeric) AND Risk Profile (Category)
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([data])

    # specific check to ensure columns match
    missing_cols = [col for col in model_features if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols}")

    # 1. Predict Return (Regression)
    pred_return = reg_model.predict(df[model_features])[0]

    # 2. Predict Risk (Classification)
    pred_risk_idx = class_model.predict(df[model_features])[0]
    pred_risk_label = risk_map.get(pred_risk_idx, "Unknown")

    return {
        "predicted_next_day_return": round(float(pred_return), 6),
        "predicted_risk_profile": pred_risk_label,
        "risk_index": int(pred_risk_idx)
    }

# ==========================================
# 3. FILE UPLOAD ENDPOINT (Required for Project)
# ==========================================
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Requirement: Handle file uploads.
    Upload a CSV file. Returns predictions for all rows.
    """
    # Read file content
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    # Validate columns
    if not all(col in df.columns for col in model_features):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {model_features}")

    # Run predictions on the whole file
    returns = reg_model.predict(df[model_features])
    risks = class_model.predict(df[model_features])
    
    # Convert risk indices (0,1,2) to labels (Low, Med, High)
    risk_labels = [risk_map.get(r, "Unknown") for r in risks]

    return {
        "batch_predictions": [
            {
                "return": round(float(ret), 6), 
                "risk": lbl
            } 
            for ret, lbl in zip(returns, risk_labels)
        ]
    }