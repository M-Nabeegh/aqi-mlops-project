from xgboost import XGBRegressor
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# =====================
# Load data
# =====================
df = pd.read_csv("features.csv")

features = ["ma_10", "ma_50", "volatility", "rsi"]
target = "return"

# Force numeric + clean
df[features + [target]] = df[features + [target]].apply(
    pd.to_numeric, errors="coerce"
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# =====================
# Time-series split
# =====================
split = int(len(df) * 0.8)

X_train = df.iloc[:split][features]
X_test  = df.iloc[split:][features]
y_train = df.iloc[:split][target]
y_test  = df.iloc[split:][target]

# =====================
# Model
# =====================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# Predictions
# =====================
preds = model.predict(X_test)

# =====================
# Evaluation
# =====================
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Model RMSE:", rmse)

# =====================
# Save model
# =====================
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")