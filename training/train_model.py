import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

print("Loading data...")
df = pd.read_csv("features.csv")

features = ["ma_10", "ma_50", "volatility", "rsi"]


df[features + ["return"]] = df[features + ["return"]].apply(pd.to_numeric, errors="coerce")
df["target_return"] = df["return"].shift(-1)
df["target_risk"] = df["risk"].shift(-1)


df.dropna(inplace=True)

risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["target_risk_encoded"] = df["target_risk"].map(risk_mapping)

df = df.dropna(subset=["target_risk_encoded"])
split = int(len(df) * 0.8)

X_train = df.iloc[:split][features]
X_test  = df.iloc[split:][features]
y_train_reg = df.iloc[:split]["target_return"]
y_test_reg  = df.iloc[split:]["target_return"]

y_train_class = df.iloc[:split]["target_risk_encoded"]
y_test_class  = df.iloc[split:]["target_risk_encoded"]

print(f"Training data size: {len(X_train)} rows")
print(f"Testing data size: {len(X_test)} rows")
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

reg_preds = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
print(f"Regression Performance (RMSE): {rmse:.5f}")

print("\nTraining Classification Model (Random Forest)...")
class_model = RandomForestClassifier(n_estimators=100, random_state=42)
class_model.fit(X_train, y_train_class)

class_preds = class_model.predict(X_test)
accuracy = accuracy_score(y_test_class, class_preds)
print(f"Classification Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(classification_report(y_test_class, class_preds, target_names=["Low", "Medium", "High"]))
print("\n🔮 PREDICTING FOR TOMORROW:")

latest_data = df.iloc[[-1]][features]

next_day_return = reg_model.predict(latest_data)[0]

next_day_risk_index = class_model.predict(latest_data)[0]

risk_map_reverse = {0: "Low", 1: "Medium", 2: "High"}
risk_label = risk_map_reverse[next_day_risk_index]

print(f"📈 Predicted Next Day Return: {next_day_return:.5f}")
print(f"⚠️ Predicted Risk Profile:    {risk_label}")
artifacts = {
    "regressor": reg_model,
    "classifier": class_model,
    "features": features
}

joblib.dump(artifacts, "model.pkl")
print("\n✅ Success! Models saved to 'model.pkl'")