# src/fetch_data.py

import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
import os

# === Step 1: Load API Key ===
with open("../alpha_key.txt", "r") as file:
    api_key = file.read().strip()

# === Step 2: Initialize Alpha Vantage client ===
fx = ForeignExchange(key=api_key)

# === Step 3: Fetch historical daily EUR/USD data ===
print("Fetching historical daily EUR/USD data...")

data, meta_data = fx.get_currency_exchange_daily(
    from_symbol='EUR',
    to_symbol='USD',
    outputsize='full'  # 'compact' = last 100 days, 'full' = full history
)

# === Step 4: Convert to pandas DataFrame ===
df = pd.DataFrame.from_dict(data, orient='index')

# Rename columns
df = df.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close'
})

# Convert index to datetime and sort
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# === Step 5: Save as CSV ===
os.makedirs("../data/raw", exist_ok=True)
output_path = "../data/raw/EURUSD_daily.csv"
df.to_csv(output_path)

print(f"Data successfully saved to {output_path}")
print("Here are the first 5 rows:")
print(df.head())
