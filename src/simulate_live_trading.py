# simulate_live_trading.py
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import warnings, random
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---------------- FILE PATHS ----------------
MODEL_PATH = "../data/processed/lgb_h3_model.txt"
SCALER_PATH = "../data/processed/scaler_h3.save"
FEATURE_NAMES_PATH = "../data/processed/feature_names_h3.pkl"
DATA_PATH = "../data/processed/EURUSD_features_with_targets.csv"
OUTPUT_PATH = "../data/results/simulation_results.csv"
EQUITY_CURVE_PATH = "../plots/equity_curve.png"

# ---------------- SIMULATION SETTINGS ------------------
INITIAL_CAPITAL = 100.0
FIXED_RISK_DOLLARS = 10.0     # $10 risk per trade
LOT_SIZE = 0.1                # assume 0.1 lot = $10,000 exposure
PIP_VALUE = 0.0001
REWARD_TO_RISK = 3.0
HORIZON = 3

# --- broker realism ---
SPREAD = 0.0002
COMMISSION = 0.00005
SLIPPAGE_MIN = 0.0001
SLIPPAGE_MAX = 0.0003

print("🚀 Starting unseen backtest (chronological 80/20 split + broker realism)...")

# --- load model and data ---
if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
    raise FileNotFoundError("Model or scaler not found. Run forex_bot_v2.py first.")
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Feature dataset not found at {DATA_PATH}.")

model = lgb.Booster(model_file=MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
if Path(FEATURE_NAMES_PATH).exists():
    model_feature_names = [c.lower().strip() for c in joblib.load(FEATURE_NAMES_PATH)]
else:
    model_feature_names = [c.lower().strip() for c in model.feature_name()]

df = pd.read_csv(DATA_PATH)
df.columns = [c.lower().strip() for c in df.columns]
print(f"✅ Loaded dataset: {df.shape}")

# --- chronological split ---
split_index = int(len(df) * 0.80)
df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()
print(f"🧭 Chronological split: Train={df_train.shape}, Test={df_test.shape}")

# --- align features ---
numeric_df = df_test.select_dtypes(include=[np.number]).copy()
ordered_features = [f for f in model_feature_names if f in numeric_df.columns]
for mf in [f for f in model_feature_names if f not in numeric_df.columns]:
    numeric_df[mf] = 0.0
X = numeric_df[ordered_features]
print(f"🧩 Final aligned feature shape: {X.shape}")

# --- scale & predict ---
X_scaled = scaler.transform(X)
probs = model.predict(X_scaled)
preds = (probs > 0.5).astype(int)
df_sim = df_test.reset_index(drop=True).copy()
df_sim["pred_prob"] = probs
df_sim["pred"] = preds
df_sim["signal"] = np.where(df_sim["pred"] == 1, "BUY", "SELL")
print("✅ Predictions generated.")

# --- simulate ---
balance = INITIAL_CAPITAL
trade_records, wins, losses = [], 0, 0
atr_col = next((c for c in ["atr_14", "atr14", "atr"] if c in df_sim.columns), None)

for i in range(len(df_sim) - 1):
    signal = df_sim.loc[i, "signal"]
    entry_index = i + 1
    if entry_index >= len(df_sim):
        break

    entry_price = df_sim.loc[entry_index, "open"]
    if pd.isna(entry_price) or entry_price == 0:
        continue

    # --- determine stop distance using ATR or 10 pips default ---
    stop_distance = df_sim.loc[i, atr_col] if atr_col else 0.0010
    stop_distance = max(stop_distance, 0.0010)

    # --- lot risk model ---
    # calculate pip distance
    pip_distance = stop_distance / PIP_VALUE
    pip_value_total = LOT_SIZE * 10  # e.g. 0.1 lot → $1 per pip
    dollar_risk = pip_distance * pip_value_total

    # scale exposure so that real risk = FIXED_RISK_DOLLARS
    exposure_factor = FIXED_RISK_DOLLARS / dollar_risk
    actual_lot = LOT_SIZE * exposure_factor

    slippage = random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
    if signal == "BUY":
        entry_price += slippage + SPREAD / 2
        sl_price = entry_price - stop_distance
        tp_price = entry_price + REWARD_TO_RISK * stop_distance
    else:
        entry_price -= slippage + SPREAD / 2
        sl_price = entry_price + stop_distance
        tp_price = entry_price - REWARD_TO_RISK * stop_distance

    exit_price = None
    hit_type = None
    for j in range(entry_index, min(entry_index + HORIZON, len(df_sim))):
        high, low = df_sim.loc[j, ["high", "low"]]
        if signal == "BUY":
            if high >= tp_price: exit_price, hit_type = tp_price, "tp"; break
            if low <= sl_price: exit_price, hit_type = sl_price, "sl"; break
        else:
            if low <= tp_price: exit_price, hit_type = tp_price, "tp"; break
            if high >= sl_price: exit_price, hit_type = sl_price, "sl"; break

    if exit_price is None:
        exit_price = df_sim.loc[min(entry_index + HORIZON - 1, len(df_sim)-1), "close"]

    exit_slip = random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
    if signal == "BUY": exit_price -= exit_slip + SPREAD / 2
    else: exit_price += exit_slip + SPREAD / 2

    # --- pip gain/loss ---
    pip_change = (exit_price - entry_price) / PIP_VALUE if signal == "BUY" else (entry_price - exit_price) / PIP_VALUE
    profit = pip_change * actual_lot * 10 - COMMISSION * 10
    balance += profit

    if profit > 0: wins += 1
    else: losses += 1

    trade_records.append({
        "signal": signal,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hit_type": hit_type,
        "profit": profit,
        "balance_after": balance
    })

# --- summary ---
df_trades = pd.DataFrame(trade_records)
win_rate = (wins / len(df_trades)) * 100 if len(df_trades) else 0
avg_profit = df_trades["profit"].mean() if len(df_trades) else 0
total_pnl = df_trades["profit"].sum() if len(df_trades) else 0

print("\n✅ Simulation finished (with broker realism).")
print("------------------------------------------------------------")
print(f"Initial capital : ${INITIAL_CAPITAL:,.2f}")
print(f"Final capital   : ${balance:,.2f}")
print(f"Total PnL       : ${total_pnl:,.2f}")
print(f"Total trades    : {len(df_trades)}")
print(f"Wins / Losses   : {wins} / {losses}")
print(f"Win rate        : {win_rate:.2f}%")
print(f"Avg trade profit: ${avg_profit:,.2f}")
print("------------------------------------------------------------")

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df_trades.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved results to: {OUTPUT_PATH}")

# --- equity curve ---
plt.figure(figsize=(10, 5))
plt.plot(df_trades["balance_after"], linewidth=2)
plt.title("Equity Curve (0.1 Lot Risk per Trade)")
plt.xlabel("Trade #")
plt.ylabel("Account Balance ($)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(EQUITY_CURVE_PATH)
plt.close()
print(f"✅ Equity curve saved to: {EQUITY_CURVE_PATH}")