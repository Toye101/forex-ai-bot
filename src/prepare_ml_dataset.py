# 📄 src/prepare_ml_dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Loading processed data: ../data/processed/EURUSD_with_indicators.csv")
df = pd.read_csv("../data/processed/EURUSD_with_indicators.csv", parse_dates=True, index_col=0)

# --- Drop NaN rows (because indicators like SMA/RSI need warm-up periods) ---
df = df.dropna()
print(f"Rows after dropping warm-up NaNs: {len(df)}")

# --- Define target ---
# Binary target: 1 if next day's close > today's close else 0
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()
print(f"Rows after dropping last row (no next-day target): {len(df)}")

# --- Features ---
features = [
    "SMA10", "SMA30", "EMA10", "EMA30",
    "MACD", "MACD_signal", "MACD_hist",
    "ADX", "RSI", "STOCH_K", "STOCH_D", "CCI",
    "ATR", "BB_upper", "BB_middle", "BB_lower", "BB_width"
]

X = df[features]
y = df["Target"]

# --- Train/Validation/Test Split ---
train_size = 0.70
val_size = 0.15
test_size = 0.15

train_end = int(len(df) * train_size)
val_end = train_end + int(len(df) * val_size)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print(f"Total rows: {len(df)}")
print(f"Train: {len(X_train)} rows")
print(f"Validation: {len(X_val)} rows")
print(f"Test: {len(X_test)} rows")

# --- Save datasets ---
ml_ready = pd.concat([X, y], axis=1)
ml_ready.to_csv("../data/ml_ready/EURUSD_ML_dataset.csv")
pd.concat([X_train, y_train], axis=1).to_csv("../data/ml_ready/EURUSD_train.csv")
pd.concat([X_val, y_val], axis=1).to_csv("../data/ml_ready/EURUSD_val.csv")
pd.concat([X_test, y_test], axis=1).to_csv("../data/ml_ready/EURUSD_test.csv")

print("Datasets saved to ../data/ml_ready/")

# --- Plot target distribution ---
target_counts = y.value_counts(normalize=True)
print("\nTarget distribution (full dataset):")
print(target_counts)

plt.figure(figsize=(5,4))
target_counts.plot(kind="bar", title="Target Distribution (0 = Down, 1 = Up)")
plt.savefig("../data/ml_ready/target_distribution.png")
plt.close()
print("Target distribution plot saved to: ../data/ml_ready/target_distribution.png")
