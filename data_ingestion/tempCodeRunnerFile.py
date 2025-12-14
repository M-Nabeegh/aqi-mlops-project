import requests
import pandas as pd
from datetime import datetime

API_KEY = "6cf977bc3a8f16c3ccf6091eff955338"
LAT, LON = 33.6844, 73.0479  # Islamabad

def fetch_aqi():
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}"
    )
    response = requests.get(url).json()
    data = response["list"][0]

    record = {
        "timestamp": datetime.utcfromtimestamp(data["dt"]),
        "pm2_5": data["components"]["pm2_5"],
        "pm10": data["components"]["pm10"],
        "no2": data["components"]["no2"],
        "o3": data["components"]["o3"],
        "aqi": data["main"]["aqi"]
    }

    df = pd.DataFrame([record])
    df.to_csv("raw_aqi.csv", mode="a", header=False, index=False)

    return df

if __name__ == "__main__":
    print(fetch_aqi())