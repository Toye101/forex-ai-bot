# 📄 src/build_indicators.py
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

# Load raw data
print("Loading raw data...")
df = pd.read_csv("../data/raw/EURUSD_daily.csv", parse_dates=True, index_col=0)

print("Building indicators...")

# --- Trend Indicators ---
df["SMA10"] = talib.SMA(df["Close"], timeperiod=10)
df["SMA30"] = talib.SMA(df["Close"], timeperiod=30)
df["EMA10"] = talib.EMA(df["Close"], timeperiod=10)
df["EMA30"] = talib.EMA(df["Close"], timeperiod=30)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)

# --- Momentum Indicators ---
df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
df["STOCH_K"], df["STOCH_D"] = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
df["CCI"] = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=14)

# --- Volatility Indicators ---
df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df["BB_upper"] = upper
df["BB_middle"] = middle
df["BB_lower"] = lower
df["BB_width"] = (upper - lower) / middle  # relative width

# Save dataset
df.to_csv("../data/processed/EURUSD_with_indicators.csv")

print("Indicators saved to ../data/processed/EURUSD_with_indicators.csv")
print("\nFirst 5 rows with indicators:")
print(df.head())

# Plot to visualize
print("Saving plot...")
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"], label="Close", alpha=0.7)
plt.plot(df.index, df["SMA10"], label="SMA10", alpha=0.7)
plt.plot(df.index, df["SMA30"], label="SMA30", alpha=0.7)
plt.plot(df.index, df["EMA10"], label="EMA10", alpha=0.7)
plt.plot(df.index, df["EMA30"], label="EMA30", alpha=0.7)
plt.legend()
plt.title("EURUSD with Moving Averages")
plt.savefig("../data/plots/indicators.png")
plt.close()
print("Chart saved to ../data/plots/indicators.png")
