import pandas as pd
import numpy as np

df = pd.read_csv("raw_market.csv")

df["return"] = df["Close"].pct_change()
df["volatility"] = df["return"].rolling(20).std()
df["ma_10"] = df["Close"].rolling(10).mean()
df["ma_50"] = df["Close"].rolling(50).mean()
df["rsi"] = 100 - (100 / (1 + df["return"].rolling(14).mean()))


df["risk"] = pd.cut(
    df["volatility"],
    bins=[-1, 0.01, 0.02, 1],
    labels=["Low", "Medium", "High"]
)

df.dropna(inplace=True)
df.to_csv("features.csv", index=False)