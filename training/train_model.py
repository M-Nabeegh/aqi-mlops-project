import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("features.csv")

if df.shape[0] < 5:
    raise ValueError(f"Not enough data to train model. Found {df.shape[0]} rows.")

X = df.drop(columns=["aqi", "timestamp"])
y = df["pm2_5"]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)

mlflow.start_run()

model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

mlflow.log_metric("rmse_pm2_5", rmse)
mlflow.log_metric("r2_pm2_5", r2)

joblib.dump(model, "model.pkl")

input_example = X_train.iloc[:5]
mlflow.sklearn.log_model(
    model,
    name="aqi_model",
    input_example=input_example
)

mlflow.end_run()

if r2 > 0.95:
    print("⚠️ Warning: Very high R2 — check for possible leakage or overfitting")

print("RMSE:", rmse)
print("R2:", r2)