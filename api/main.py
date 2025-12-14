from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Stock Return Prediction API")

# Load trained finance model
model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    """
    Expected input JSON:
    {
        "ma_10": float,
        "ma_50": float,
        "volatility": float,
        "rsi": float
    }
    """
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    signal = "BUY" if prediction > 0 else "SELL"

    return {
        "predicted_return": round(float(prediction), 6),
        "signal": signal
    }