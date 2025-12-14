import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(BASE_DIR, "raw_aqi.csv")
FEATURE_FILE = os.path.join(BASE_DIR, "features.csv")

def build_features():
    df = pd.read_csv(RAW_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)
    df["aqi_roll_24"] = df["aqi"].rolling(24).mean()

    df.dropna(inplace=True)
    df.to_csv(FEATURE_FILE, index=False)

    return df

if __name__ == "__main__":
    print(build_features().head())