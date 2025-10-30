import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # === Basic price returns ===
    df["return"] = df["close"].pct_change()
    df["O-C"] = df["open"] - df["close"]

    # === Trend Indicators ===
    df["sma_5"] = SMAIndicator(df["close"], 5).sma_indicator().shift(1)
    df["sma_20"] = SMAIndicator(df["close"], 20).sma_indicator().shift(1)
    df["sma_50"] = SMAIndicator(df["close"], 50).sma_indicator().shift(1)
    df["ema_20"] = EMAIndicator(df["close"], 20).ema_indicator().shift(1)
    df["ema_50"] = EMAIndicator(df["close"], 50).ema_indicator().shift(1)
    df["sma_diff"] = df["sma_5"] - df["sma_20"]

    # === Momentum Indicators ===
    df["rsi_14"] = RSIIndicator(df["close"], 14).rsi().shift(1)
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_k"] = stoch.stoch().shift(1)
    df["stoch_d"] = stoch.stoch_signal().shift(1)
    macd = MACD(df["close"], 26, 12, 9)
    df["macd"] = macd.macd().shift(1)
    df["macd_signal"] = macd.macd_signal().shift(1)
    df["macd_hist"] = macd.macd_diff().shift(1)

    # === Volatility Indicators ===
    bb = BollingerBands(df["close"], 20, 2)
    df["bb_high"] = bb.bollinger_hband().shift(1)
    df["bb_low"] = bb.bollinger_lband().shift(1)
    df["bb_width"] = df["bb_high"] - df["bb_low"]
    df["volatility_20"] = df["return"].rolling(20).std().shift(1)
    atr = AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["atr_14"] = atr.average_true_range().shift(1)

    # === Price Strength Ratios ===
    df["close_open_ratio"] = (df["close"] / df["open"]) - 1
    df["high_low_ratio"] = (df["high"] / df["low"]) - 1

    # === Momentum Smoothers ===
    df["return_mean_5"] = df["return"].rolling(5).mean().shift(1)
    df["return_zscore_20"] = ((df["return"] - df["return"].rolling(20).mean()) /
                              df["return"].rolling(20).std()).shift(1)

    # === Volume Features ===
    if "volume" in df.columns:
        df["volume_change"] = df["volume"].pct_change().shift(1)
        df["vol_norm"] = (df["volume"] / df["volume"].rolling(20).mean()).shift(1)

    # === Lag Features ===
    for col in ["return", "rsi_14", "macd", "macd_hist"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    # === Market Regime ===
    df["slope_short"] = df["sma_5"] - df["sma_20"]
    df["slope_long"] = df["sma_20"] - df["sma_50"]
    conds = [
        (df["slope_short"] > 0) & (df["slope_long"] > 0),
        (df["slope_short"] < 0) & (df["slope_long"] < 0)
    ]
    df["regime"] = pd.Series(
        np.select(conds, [1, -1], default=0),
        index=df.index
    ).astype(int).shift(1)

    # === Drop NaNs ===
    df = df.dropna().reset_index(drop=True)

    # === Drop highly correlated features (keep 'close') ===
    print("🔗 Checking and dropping highly correlated features...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95) and col != "close"]
    df = df.drop(columns=to_drop, errors="ignore")
    print(f"⚙ Dropped correlated columns (except 'close'): {to_drop}")

    print("\n✅ Feature engineering completed successfully!\n")
    return df


if __name__ == "__main__":
    print("🔧 Building hybrid features (clean + robust)...")
    df = pd.read_csv("../data/raw/EURUSD_daily.csv")
    df = engineer_features(df)
    df.to_csv("../data/processed/EURUSD_features.csv", index=False)
    print("✅ Saved processed features to ../data/processed/EURUSD_features.csv")
    print(f"✅ Final columns: {list(df.columns)}")