import requests
import pandas as pd
import os
from datetime import datetime, timedelta

API_KEY = "6cf977bc3a8f16c3ccf6091eff955338"
LAT, LON = 33.6844, 73.0479

FILE = "raw_aqi.csv"

def fetch_historical(days=7):
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    records = []

    while start <= end:
        ts = int(start.timestamp())

        url = (
            "http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={LAT}&lon={LON}&start={ts}&end={ts+3600}&appid={API_KEY}"
        )

        res = requests.get(url).json()

        if "list" in res and len(res["list"]) > 0:
            data = res["list"][0]
            records.append({
                "timestamp": datetime.utcfromtimestamp(data["dt"]),
                "pm2_5": data["components"]["pm2_5"],
                "pm10": data["components"]["pm10"],
                "no2": data["components"]["no2"],
                "o3": data["components"]["o3"],
                "aqi": data["main"]["aqi"]
            })

        start += timedelta(hours=1)

    df = pd.DataFrame(records)

    if not os.path.exists(FILE):
        df.to_csv(FILE, index=False)
    else:
        df.to_csv(FILE, mode="a", header=False, index=False)

    return df

if __name__ == "__main__":
    df = fetch_historical(days=10)
    print(f"Collected rows: {len(df)}")