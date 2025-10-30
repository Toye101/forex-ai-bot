import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings("ignore")

def build_refined_features(input_path="../data/processed/EURUSD_features_regime.csv", output_path="../data/processed/EURUSD_features_refined.csv"):
    print("🔧 Building hybrid + regime-aware refined features...")

    # === Load data ===
    df = pd.read_csv(input_path)
    df.columns = [c.lower() for c in df.columns]

    # Ensure essential columns exist
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("❌ Missing one of required columns: open, high, low, close")

    # === Basic Features ===
    df["return"] = df["close"].pct_change() * 100
    df["sma_diff"] = SMAIndicator(df["close"], 10).sma_indicator() - SMAIndicator(df["close"], 30).sma_indicator()

    # === Momentum Indicators ===
    df["rsi_14"] = RSIIndicator(df["close"], 14).rsi()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], 14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # === Trend Indicators ===
    df["ema_20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["close"], 50).ema_indicator()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # === Volatility Indicators ===
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["atr_14"] = df["high"] - df["low"]

    # === Regime Feature (volatility-based) ===
    df["volatility_20"] = df["return"].rolling(20).std()
    df["regime"] = np.where(df["volatility_20"] > df["volatility_20"].median(), 1, 0)

    # === Hybrid Features (momentum-trend-volatility combos) ===
    df["trend_vol_combo"] = df["sma_diff"] * df["volatility_20"]
    df["rsi_macd_combo"] = df["rsi_14"] * df["macd_hist"]
    df["regime_momentum"] = df["regime"] * df["rsi_14"]

    # === Lagged Features ===
    for col in ["return", "rsi_14", "macd_hist"]:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # === Drop non-useful hybrid columns ===
    drop_cols = ["trend_vol_combo", "rsi_macd_combo", "regime_momentum"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # === Drop NaN and Save ===
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Refined features built and saved to {output_path}")
    print(f"✅ Final shape: {df.shape}")

if __name__ == "__main__":
    build_refined_features()