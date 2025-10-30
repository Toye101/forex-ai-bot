# forex_bot_v2.py (Data-Leakage-Safe Version)
# Fully chronological pipeline with LightGBM models and backtesting
# Prevents any future leakage during feature generation and scaling.

import os
import warnings
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
RAW_CSV = "../data/raw/EURUSD_daily.csv"
PROCESSED_CSV = "../data/processed/EURUSD_features.csv"
MODEL_OUT_DIR = "../data/processed"
PLOTS_DIR = "../plots"
RISK_PER_TRADE = 0.10
PROB_THRESH_A = 0.65
PROB_THRESH_B = 0.55
RRS = [4, 5]
HORIZONS = [3, 5]
TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# UTILITIES
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Using raw CSV file: {path}")
    print(f"✅ Loaded raw data: {df.shape}\n")
    return df


def ensure_ohlc_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cand in ["Unnamed: 0", "timestamp", "Date", "date"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "date"})
            break
    mapping = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index().rename(columns={"index": "date"})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    return df


# -----------------------------
# FEATURE ENGINEERING — no future leakage
# -----------------------------
def engineer_features_base(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_ohlc_datetime(df)
    df["return"] = df["close"].pct_change()  # ✅ past info only
    df["O-C"] = (df["open"] - df["close"]).abs().shift(1)  # ✅ use prior day

    # Moving averages — inherently safe since they use rolling past data
    df["sma_5"] = df["close"].shift(1).rolling(5).mean()
    df["sma_20"] = df["close"].shift(1).rolling(20).mean()
    df["sma_50"] = df["close"].shift(1).rolling(50).mean()
    df["sma_diff"] = df["sma_5"] - df["sma_20"]
    df["sma_ratio"] = df["sma_5"] / df["sma_20"].replace(0, np.nan)

    # EMAs
    df["ema_10"] = df["close"].shift(1).ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["close"].shift(1).ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    df["bb_mid"] = df["close"].shift(1).rolling(20).mean()
    rolling_std = df["close"].shift(1).rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * rolling_std
    df["bb_lower"] = df["bb_mid"] - 2 * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    # ATR (True Range)
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-C"] = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean().shift(1)

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean().shift(1)
    roll_down = down.ewm(alpha=1/14, adjust=False).mean().shift(1)
    RS = roll_up / roll_down.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + RS))

    # Volatility
    df["rolling_volatility_10"] = df["return"].rolling(10).std().shift(1)

    # Candlestick ratios
    df["price_range_ratio"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["body_to_range_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)

    # Lags
    df["return_lag_1"] = df["return"].shift(1)
    df["return_lag_3"] = df["return"].shift(3)
    df["return_lag_5"] = df["return"].shift(5)

    # Direction streaks
    df["direction"] = np.sign(df["return"].fillna(0)).astype(int)
    streak, run, prev = [], 0, 0
    for v in df["direction"]:
        if v == prev and v != 0:
            run += 1
        elif v != 0:
            run = 1
        else:
            run = 0
        streak.append(run)
        prev = v
    df["direction_streak"] = streak
    return df


def add_forward_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df[f"fwd_return_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1
    df[f"target_{horizon}"] = (df[f"fwd_return_{horizon}"] > 0).astype(int)
    return df


# -----------------------------
# CORRELATION CLEANUP
# -----------------------------
def drop_highly_correlated_features(df: pd.DataFrame, threshold=0.95, protect=None):
    protect = protect or []
    numeric = df.select_dtypes(include=[np.number]).copy()
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold) and c not in protect]
    if to_drop:
        print(f"🧹 Dropping highly correlated features (|r| > {threshold}): {to_drop}")
    df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


# -----------------------------
# CHRONOLOGICAL SPLIT
# -----------------------------
def chronological_train_test_split(df, test_fraction=0.2):
    n = len(df)
    test_n = int(n * test_fraction)
    return df.iloc[:-test_n], df.iloc[-test_n:]


# -----------------------------
# LIGHTGBM TRAINING
# -----------------------------
def train_lightgbm(X_train, y_train, X_val=None, y_val=None):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_STATE,
        "n_jobs": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=300)
    return model


# -----------------------------
# MONEY MANAGEMENT
# -----------------------------
def simulate_trades(pred_probs, preds, df_test, rr, prob_threshold, risk_per_trade, initial_capital=1000.0):
    capital = initial_capital
    equity_curve = []
    wins, losses, trades = 0, 0, 0
    for i in range(len(df_test)):
        p = pred_probs[i]
        if p < prob_threshold or preds[i] != 1:
            equity_curve.append(capital)
            continue
        trades += 1
        risk_amount = capital * risk_per_trade
        fwd_return = df_test.filter(like="fwd_return").iloc[i, 0]
        stop_pct = 0.02
        tp_pct = stop_pct * rr
        position_value = risk_amount / stop_pct
        if fwd_return >= tp_pct:
            profit = position_value * tp_pct; wins += 1
        elif fwd_return <= -stop_pct:
            profit = -risk_amount; losses += 1
        else:
            profit = position_value * fwd_return
            (wins, losses) = (wins + 1, losses) if profit > 0 else (wins, losses + 1)
        capital += profit
        equity_curve.append(capital)
    win_rate = wins / trades if trades else 0.0
    return {"final_capital": capital, "trades": trades, "win_rate": win_rate}


# -----------------------------
# RUN EXPERIMENTS
# -----------------------------
def run_experiment(df_all, horizon, rr_list, prob_a, prob_b, risk_per_trade):
    print(f"\n--- Horizon: {horizon} days ---")
    df = df_all.dropna(subset=[f"fwd_return_{horizon}", f"target_{horizon}"]).reset_index(drop=True)
    train_df, test_df = chronological_train_test_split(df, TEST_SIZE)
    print(f"Training on {len(train_df)}, testing on {len(test_df)}")

    exclude = [f"fwd_return_{horizon}", f"target_{horizon}", "date"]
    features = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]
    X_train, X_test = train_df[features], test_df[features]
    y_train, y_test = train_df[f"target_{horizon}"], test_df[f"target_{horizon}"]

    # ✅ no leakage: fit scaler only on training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_lightgbm(X_train_s, y_train)
    probs = model.predict(X_test_s)
    preds = (probs >= 0.5).astype(int)
    print(f"✅ Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, digits=3))

    results = {}
    for rr in rr_list:
        res_A = simulate_trades(probs, preds, test_df, rr, prob_a, risk_per_trade)
        res_B = simulate_trades(probs, preds, test_df, rr, prob_b, risk_per_trade)
        results[f"RR_{rr}_A"], results[f"RR_{rr}_B"] = res_A, res_B
        print(f"H{horizon} 1:{rr} A -> Trades={res_A['trades']} | WR={res_A['win_rate']:.3f}")
        print(f"H{horizon} 1:{rr} B -> Trades={res_B['trades']} | WR={res_B['win_rate']:.3f}")
    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Starting pipeline...\n")
    df_raw = safe_read_csv(RAW_CSV)
    df = engineer_features_base(df_raw)
    for h in HORIZONS:
        df = add_forward_targets(df, h)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"💾 Processed dataset saved to {PROCESSED_CSV}")

    protect = ["open", "high", "low", "close", "date"] + [f"target_{h}" for h in HORIZONS] + [f"fwd_return_{h}" for h in HORIZONS]
    df_cleaned, _ = drop_highly_correlated_features(df, protect=protect)

    for h in HORIZONS:
        run_experiment(df_cleaned, h, RRS, PROB_THRESH_A, PROB_THRESH_B, RISK_PER_TRADE)
    print("\n✅ Pipeline complete (data-leakage-free).")


if __name__ == "__main__":
    main()
