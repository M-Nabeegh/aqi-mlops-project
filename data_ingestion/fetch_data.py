import yfinance as yf
import pandas as pd

def fetch_data(symbol="AAPL", period="5y"):
    # Download data
    df = yf.download(symbol, period=period, auto_adjust=True)

    # ✅ FIX 1: Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index (Date → column)
    df.reset_index(inplace=True)

    # ✅ FIX 2: Force numeric conversion safely
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save clean CSV
    df.to_csv("raw_market.csv", index=False)

    return df

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
    print(df.dtypes)