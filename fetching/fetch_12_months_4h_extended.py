# fetch_12_months_4h_extended.py

import os
import time
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_PREMIUM_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_last_n_months(n=12):
    today = datetime.today().replace(day=1)
    return [(today - timedelta(days=30 * i)).strftime("%Y-%m") for i in range(n)]

def label_session(dt: pd.Timestamp) -> str:
    hour = dt.hour + dt.minute / 60
    if 9.5 <= hour < 16:
        return "regular"
    return "extended"

def fetch_12_months_4h_extended(symbol: str, save: bool = True, delay_seconds: int = 12) -> pd.DataFrame:
    months = get_last_n_months(12)
    all_data = []

    print(f"🔄 Fetching 1-hour data with extended hours for last 12 months for: {symbol}")
    for month in months:
        print(f"\n📦 Requesting month: {month}")
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "60min",
            "month": month,
            "outputsize": "full",
            "datatype": "csv",
            "extended_hours": "true",
            "apikey": API_KEY
        }

        try:
            response = requests.get(BASE_URL, params=params)
            text = response.text.strip()

            print(f"🔍 Response preview for {month}:\n{text[:200]}")

            if "thank you" in text.lower() or "invalid" in text.lower() or "error" in text.lower() or len(text.splitlines()) < 2:
                print(f"⚠️ Skipping {month}: No data or throttled.")
                time.sleep(delay_seconds)
                continue

            df = pd.read_csv(StringIO(text))

            if df.shape[0] == 0:
                print(f"⚠️ Skipping {month}: Empty CSV (no rows).")
                time.sleep(delay_seconds)
                continue

            df.columns = [col.lower() for col in df.columns]
            time_col = "time" if "time" in df.columns else "timestamp" if "timestamp" in df.columns else None
            if not time_col:
                print(f"⚠️ Columns returned: {list(df.columns)}")
                print(f"⚠️ Skipping {month}: Missing timestamp column.")
                time.sleep(delay_seconds)
                continue

            df["time"] = pd.to_datetime(df[time_col])
            df = df.sort_values("time")
            all_data.append(df)

        except Exception as e:
            print(f"❌ Error on {month}: {e}")
        finally:
            time.sleep(delay_seconds)

    if not all_data:
        raise RuntimeError("❌ No valid data retrieved from Alpha Vantage.")

    df_full = pd.concat(all_data).drop_duplicates().reset_index(drop=True)
    df_full.set_index("time", inplace=True)

    # Resample to 4-hour candles
    df_4h = df_full.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    # Tag regular vs extended hours
    df_4h["session"] = df_4h["time"].apply(label_session)

    print(f"\n✅ Total 4-hour candles: {len(df_4h)}")
    print(f"🔖 Session breakdown: {df_4h['session'].value_counts().to_dict()}")

    if save:
        filename = f"{symbol}_4h_12mo_extended.csv"
        df_4h.to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

    return df_4h

if __name__ == "__main__":
    symbol = "AAPL"
    df = fetch_12_months_4h_extended(symbol, save=True)
    print(df.tail())
