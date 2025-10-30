"""
build_features_regime.py
Enhanced feature engineering with regime detection for the forex AI bot.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def engineer_features(df):
    print("🔧 Building hybrid features (clean + regime)...")

    # Ensure columns are lowercase
    df.columns = df.columns.str.lower()

    # === Basic Features ===
    df["return"] = df["close"].pct_change()
    df["sma_diff"] = df["close"].rolling(5).mean() - df["close"].rolling(20).mean()

    # === Momentum Indicators ===
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # === Trend Indicators ===
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # === Volatility Indicators ===
    bb = BollingerBands(df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["high_low_ratio"] = df["high"] / df["low"]

    # === Lagged Features ===
    for col in ["return", "rsi_14", "macd", "macd_hist"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    # === Statistical Features ===
    df["return_mean_5"] = df["return"].rolling(5).mean()
    df["return_zscore_20"] = (df["return"] - df["return"].rolling(20).mean()) / df["return"].rolling(20).std()

    # === Linear Regression Trend ===
    df["slope_short"] = df["close"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    df["slope_long"] = df["close"].rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # === Regime Detection ===
    df["volatility_20"] = df["return"].rolling(window=20).std()
    vol_q1 = df["volatility_20"].quantile(0.33)
    vol_q2 = df["volatility_20"].quantile(0.66)

    cond_low_vol = df["volatility_20"] <= vol_q1
    cond_medium_vol = (df["volatility_20"] > vol_q1) & (df["volatility_20"] <= vol_q2)
    cond_high_vol = df["volatility_20"] > vol_q2

    df["regime"] = np.select(
        [cond_low_vol, cond_medium_vol, cond_high_vol],
        ["low_vol", "medium_vol", "high_vol"],
        default="unknown"
    )

    # Remove unknown regime rows (usually first 20)
    df = df[df["regime"] != "unknown"].copy()
    print("✅ Regime feature successfully added.")

    # === Regime-Based Feature Interaction ===
    # Create a composite feature combining volatility regime with trend slope
    df["trend_vol_combo"] = df["slope_long"] * df["volatility_20"]

    # === Numeric Encoding for Regime ===
    regime_map = {"low_vol": 0, "medium_vol": 1, "high_vol": 2}
    df["regime_encoded"] = df["regime"].map(regime_map)

    # === Drop rows with NaN values ===
    df.dropna(inplace=True)

    print("✅ Feature engineering completed successfully!")
    return df


def main():
    print("🚀 Starting regime-aware feature builder...")

    raw_path = "../data/raw/EURUSD_daily.csv"
    save_path = "../data/processed/EURUSD_features_regime.csv"

    # Load raw dataset
    df = pd.read_csv(raw_path)
    print(f"✅ Loaded raw data: {df.shape}")

    df = engineer_features(df)

    df.to_csv(save_path, index=False)
    print(f"💾 Saved processed features to {save_path}")
    print(f"✅ Final columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()