import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading data...")
df = pd.read_csv("features.csv")

# Define features (Input)
features = ["ma_10", "ma_50", "volatility", "rsi"]

# Define targets (Output)
# We need 'return' for regression and 'risk' for classification
# Ensure risk is calculated in your build_features.py before running this!

# Clean numeric columns
df[features + ["return"]] = df[features + ["return"]].apply(pd.to_numeric, errors="coerce")

# ==========================================
# 2. LOGIC FIX: PREDICT THE FUTURE (SHIFT)
# ==========================================
# We want to use Today's features to predict TOMORROW'S return/risk.
# shift(-1) moves the target column UP by one day.
df["target_return"] = df["return"].shift(-1)
df["target_risk"] = df["risk"].shift(-1)

# The last row will now be NaN (because there is no tomorrow for it), so we drop it.
df.dropna(inplace=True)

# Encode 'risk' labels to numbers for the Classification model
# Low -> 0, Medium -> 1, High -> 2
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["target_risk_encoded"] = df["target_risk"].map(risk_mapping)

# Remove any rows where mapping failed (just in case)
df = df.dropna(subset=["target_risk_encoded"])

# ==========================================
# 3. SPLIT DATA (TIME SERIES SPLIT)
# ==========================================
# We cannot shuffle time-series data. We must cut it at a specific point.
split = int(len(df) * 0.8)

X_train = df.iloc[:split][features]
X_test  = df.iloc[split:][features]

# Regression Targets
y_train_reg = df.iloc[:split]["target_return"]
y_test_reg  = df.iloc[split:]["target_return"]

# Classification Targets
y_train_class = df.iloc[:split]["target_risk_encoded"]
y_test_class  = df.iloc[split:]["target_risk_encoded"]

print(f"Training data size: {len(X_train)} rows")
print(f"Testing data size: {len(X_test)} rows")

# ==========================================
# 4. TRAIN REGRESSION MODEL (Predict Number)
# ==========================================
print("\nTraining Regression Model (XGBoost)...")
reg_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
reg_model.fit(X_train, y_train_reg)

# Evaluate Regression
reg_preds = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
print(f"Regression Performance (RMSE): {rmse:.5f}")

# ==========================================
# 5. TRAIN CLASSIFICATION MODEL (Predict Label)
# ==========================================
print("\nTraining Classification Model (Random Forest)...")
class_model = RandomForestClassifier(n_estimators=100, random_state=42)
class_model.fit(X_train, y_train_class)

# Evaluate Classification
class_preds = class_model.predict(X_test)
accuracy = accuracy_score(y_test_class, class_preds)
print(f"Classification Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(classification_report(y_test_class, class_preds, target_names=["Low", "Medium", "High"]))
# ==========================================
# 7. PREDICT FOR TOMORROW (INFERENCE)
# ==========================================
print("\n🔮 PREDICTING FOR TOMORROW:")

# Get the very last row of data (Today's market data)
# We select the features: ['ma_10', 'ma_50', 'volatility', 'rsi']
latest_data = df.iloc[[-1]][features]

# 1. Predict Next Day Return (Regression)
next_day_return = reg_model.predict(latest_data)[0]

# 2. Predict Risk Profile (Classification)
next_day_risk_index = class_model.predict(latest_data)[0]

# Convert the risk number (0, 1, 2) back to a word
risk_map_reverse = {0: "Low", 1: "Medium", 2: "High"}
risk_label = risk_map_reverse[next_day_risk_index]

print(f"📈 Predicted Next Day Return: {next_day_return:.5f}")
print(f"⚠️ Predicted Risk Profile:    {risk_label}")

# ==========================================
# 6. SAVE EVERYTHING
# ==========================================
# We save a dictionary containing both models so we can load them easily later
artifacts = {
    "regressor": reg_model,
    "classifier": class_model,
    "features": features
}

joblib.dump(artifacts, "model.pkl")
print("\n✅ Success! Models saved to 'model.pkl'")