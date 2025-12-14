from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="AQI Prediction API")

model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    status = "Hazardous" if prediction >= 4 else "Safe"

    return {
        "predicted_aqi": round(float(prediction), 2),
        "status": status
    }