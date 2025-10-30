# walk_forward_validation.py
"""
Walk-forward validation (10 chronological segments) with retraining per-window (Option B).
- For each of 10 chronological test segments:
    - Train on ALL data before that segment (no leakage).
    - Fit scaler on train features, train LightGBM on train.
    - Predict on test segment, run the same broker-realistic simulator.
    - Save segment metrics and equity curves.
- Outputs:
    - ../data/results/walk_forward_summary.csv
    - ../plots/walk_forward_equity_per_segment.png
    - ../plots/walk_forward_combined_equity.png
"""

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ---------------- USER / DATA PATHS ----------------
DATA_PATH = "../data/processed/EURUSD_features_with_targets.csv"
OUT_SUMMARY = "../data/results/walk_forward_summary.csv"
OUT_PLOTS_DIR = "../plots"
os.makedirs(Path(OUT_PLOTS_DIR), exist_ok=True)
os.makedirs(Path(OUT_SUMMARY).parent, exist_ok=True)

# ---------------- SIMULATION HYPERPARAMS ------------------
N_SEGMENTS = 10
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.20       # fraction of current capital (we will also allow fixed-dollar option)
FIXED_DOLLAR_RISK = None    # set to a number (e.g. 10.0) to use fixed dollar per trade instead of fraction
REWARD_TO_RISK = 3.0
HORIZON = 3
ENTRY_PRICE_PREF = "open"   # 'open' or 'close'
# Broker realism
SPREAD = 0.0002
COMMISSION = 0.00005
SLIPPAGE_MIN = 0.0001
SLIPPAGE_MAX = 0.0003

# LightGBM training defaults
LGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

# === Helper: simulate trades on a test dataframe ===
def simulate_trades(df_test, probs, preds, initial_capital=1000.0,
                    risk_per_trade=0.2, fixed_dollar_risk=None,
                    reward_to_risk=3.0, horizon=3,
                    spread=0.0002, commission=0.00005,
                    slip_min=0.0001, slip_max=0.0003,
                    entry_price_pref="open"):
    """
    df_test: test slice (index preserved) with OHLC and indicators
    probs, preds: model outputs aligned to df_test order
    returns: trades_df (records), summary dict, equity_series (list of balances)
    """
    df = df_test.reset_index(drop=True).copy()
    df["pred_prob"] = probs
    df["pred"] = preds
    df["signal"] = np.where(df["pred"] == 1, "BUY", "SELL")

    balance = float(initial_capital)
    trades = []
    wins = losses = 0

    # find possible ATR column
    atr_col = next((c for c in ["atr_14", "atr14", "atr"] if c in df.columns), None)
    n = len(df)
    i = 0
    equity = []

    while i < n - 1:
        row = df.loc[i]
        signal = row["signal"]

        # Determine entry index (next bar) to avoid lookahead
        entry_index = i + 1
        if entry_index >= n:
            break

        # entry price (prefer next open)
        if entry_price_pref == "open" and "open" in df.columns:
            entry_price = df.loc[entry_index, "open"]
        else:
            entry_price = df.loc[i, "close"] if "close" in df.columns else np.nan

        if pd.isna(entry_price) or entry_price == 0:
            i += 1
            continue

        # position sizing
        if fixed_dollar_risk is not None:
            position_value = float(fixed_dollar_risk)  # fixed $ risk target - we'll treat as max risk; actual position will be scaled by stop-distance later
        else:
            position_value = balance * float(risk_per_trade)

        # stop distance
        if atr_col and not pd.isna(df.loc[i, atr_col]) and df.loc[i, atr_col] > 0:
            stop_distance = df.loc[i, atr_col]
        else:
            stop_distance = 0.001 * entry_price

        # apply random slippage at entry
        slippage = random.uniform(slip_min, slip_max)

        # adjust entry for spread/slippage depending on side
        if signal == "BUY":
            entry_price_exec = entry_price + slippage + spread / 2
            sl_price = entry_price_exec - stop_distance
            tp_price = entry_price_exec + reward_to_risk * stop_distance
        else:
            entry_price_exec = entry_price - slippage - spread / 2
            sl_price = entry_price_exec + stop_distance
            tp_price = entry_price_exec - reward_to_risk * stop_distance

        # For fixed-dollar-risk mode: compute notional from risk and stop_distance:
        # risk_dollars = position_value (user provided) -> position_notional = risk_dollars / stop_pct
        # But stop_pct = stop_distance / entry_price_exec
        if fixed_dollar_risk is not None:
            stop_pct = stop_distance / entry_price_exec if entry_price_exec != 0 else 0.0
            if stop_pct <= 0:
                # fallback to small notional
                notional = position_value
            else:
                notional = position_value / stop_pct
            # position_value here is notional; we'll compute profit = notional * ret_pct (consistent with earlier code)
            position_value = notional

        # search forward for TP/SL/horizon close
        exit_price = None
        exit_idx = None
        hit_type = None
        for j in range(entry_index, min(entry_index + horizon, n)):
            future_high = df.loc[j, "high"] if "high" in df.columns else df.loc[j, "close"]
            future_low = df.loc[j, "low"] if "low" in df.columns else df.loc[j, "close"]

            if signal == "BUY":
                if future_high >= tp_price:
                    exit_price, exit_idx, hit_type = tp_price, j, "tp"
                    break
                if future_low <= sl_price:
                    exit_price, exit_idx, hit_type = sl_price, j, "sl"
                    break
            else:
                if future_low <= tp_price:
                    exit_price, exit_idx, hit_type = tp_price, j, "tp"
                    break
                if future_high >= sl_price:
                    exit_price, exit_idx, hit_type = sl_price, j, "sl"
                    break

        if exit_price is None:
            exit_idx = min(entry_index + horizon - 1, n - 1)
            exit_price = df.loc[exit_idx, "close"]

        # exit slippage + spread
        exit_slip = random.uniform(slip_min, slip_max)
        if signal == "BUY":
            exit_price_exec = exit_price - exit_slip - spread / 2
        else:
            exit_price_exec = exit_price + exit_slip + spread / 2

        # pct return directional
        if signal == "BUY":
            ret_pct = (exit_price_exec - entry_price_exec) / entry_price_exec
        else:
            ret_pct = (entry_price_exec - exit_price_exec) / entry_price_exec

        profit = position_value * ret_pct
        # apply commission (flat fraction of position value)
        profit -= COMMISSION * position_value

        balance += profit

        if profit > 0:
            wins += 1
        else:
            losses += 1

        trades.append({
            "entry_index": int(entry_index),
            "exit_index": int(exit_idx),
            "signal": signal,
            "entry_price": float(entry_price_exec),
            "exit_price": float(exit_price_exec),
            "hit_type": hit_type,
            "ret_pct": float(ret_pct),
            "position_value": float(position_value),
            "profit": float(profit),
            "balance_after": float(balance),
        })

        equity.append(balance)

        # advance to after exit
        i = exit_idx + 1 if exit_idx is not None else i + 1

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate = (wins / total_trades) if total_trades else 0.0
    avg_ret = trades_df["ret_pct"].mean() if total_trades else 0.0
    total_pnl = trades_df["profit"].sum() if total_trades else 0.0

    summary = dict(
        initial_capital=initial_capital,
        final_capital=float(balance),
        total_pnl=float(total_pnl),
        total_trades=int(total_trades),
        wins=int(wins),
        losses=int(losses),
        win_rate_pct=float(win_rate * 100),
        avg_return_pct=float(avg_ret * 100) if not pd.isna(avg_ret) else 0.0,
    )
    return trades_df, summary, equity


# === Main walk-forward routine ===
def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]

    if "target_3" not in df.columns:
        raise ValueError("target_3 not found in dataset. Run make_targets first.")

    n = len(df)
    seg_len = n // N_SEGMENTS
    summaries = []
    all_equities = []

    for k in range(N_SEGMENTS):
        test_start = k * seg_len
        test_end = (k + 1) * seg_len if k < N_SEGMENTS - 1 else n

        # skip if there's no train data (can't train on nothing)
        if test_start == 0:
            print(f"Skipping segment {k+1} (no training data before this test slice).")
            continue

        df_train = df.iloc[:test_start].copy()
        df_test = df.iloc[test_start:test_end].copy()
        print(f"\n=== Segment {k+1}/{N_SEGMENTS} | Train={df_train.shape} Test={df_test.shape} ===")

        # prepare numeric feature matrix (drop target/fwd_return columns)
        drop_cols = [c for c in df_train.columns if c.startswith("target_") or c.startswith("fwd_return_")]
        if "regime" in df_train.columns:
            # if textual 'regime' exists, prefer numeric encoded column if present; otherwise keep regime as numeric only if encoded
            # to be safe, drop raw 'regime' textual variable (we expect 'regime_encoded' or similar numeric)
            drop_cols.append("regime")
        X_train = df_train.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
        y_train = df_train[f"target_3"].copy()

        X_test = df_test.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
        y_test = df_test[f"target_3"].copy()

        # Align columns: ensure same feature columns in test as train (create missing with 0)
        train_cols = X_train.columns.tolist()
        for c in train_cols:
            if c not in X_test.columns:
                X_test[c] = 0.0
        # drop any extra test columns not in train
        X_test = X_test[train_cols]

        # Fit scaler on train, transform both
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train LightGBM on train only
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train_scaled, y_train)

        # Predict on test slice
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)

        # Optional quick accuracy print
        acc = accuracy_score(y_test, preds)
        print(f"Segment {k+1} classification accuracy on test slice: {acc:.4f}")

        # simulate trades on df_test using the same index ordering as X_test
        # call simulate_trades with either fraction or fixed-dollar risk
        fixed_risk = FIXED_DOLLAR_RISK  # None or number
        trades_df, summary, equity = simulate_trades(
            df_test,
            probs,
            preds,
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            fixed_dollar_risk=fixed_risk,
            reward_to_risk=REWARD_TO_RISK,
            horizon=HORIZON,
            spread=SPREAD,
            commission=COMMISSION,
            slip_min=SLIPPAGE_MIN,
            slip_max=SLIPPAGE_MAX,
            entry_price_pref=ENTRY_PRICE_PREF,
        )

        # add metadata
        summary.update({"segment": int(k + 1), "train_end_index": int(test_start), "test_start_index": int(test_start), "test_end_index": int(test_end)})
        summaries.append(summary)
        all_equities.append({"segment": k + 1, "equity": equity, "trades_df": trades_df})

        # Save per-segment trades if you want
        seg_out = f"../data/results/walk_segment_{k+1}_trades.csv"
        trades_df.to_csv(seg_out, index=False)

    # Save overall summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"\n✅ Walk-forward summary saved to: {OUT_SUMMARY}")

    # Plot equity curves: per segment and combined
    plt.figure(figsize=(12, 8))
    for seg in all_equities:
        eq = seg["equity"]
        if len(eq) == 0:
            continue
        plt.plot(np.arange(len(eq)), eq, label=f"Seg {seg['segment']}")
    plt.title("Walk-forward equity curves (per test segment)")
    plt.xlabel("Trade number (within segment)")
    plt.ylabel("Balance ($)")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    per_seg_plot = f"{OUT_PLOTS_DIR}/walk_forward_equity_per_segment.png"
    plt.savefig(per_seg_plot)
    plt.close()
    print(f"✅ Per-segment equity plot saved to: {per_seg_plot}")

    # Combined cumulative equity across time (concatenate segments sequentially)
    combined_balances = []
    for seg in all_equities:
        eq = seg["equity"]
        if len(eq) == 0:
            continue
        # start each segment equity from previous final (chained simulation) OR normalize to initial capital
        # we'll plot normalized starting at initial capital for clarity: shift so first value = INITIAL_CAPITAL
        if len(eq) > 0:
            # normalize to start at INITIAL_CAPITAL
            adj = np.array(eq)
            scale = INITIAL_CAPITAL / adj[0] if adj[0] != 0 else 1.0
            adj = adj * scale
            combined_balances.extend(adj.tolist())

    if combined_balances:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(combined_balances)), combined_balances, linewidth=2)
        plt.title("Walk-forward combined equity (segments concatenated, normalized per segment)")
        plt.xlabel("Sequential trade index across test segments")
        plt.ylabel("Balance ($) (per-segment normalized)")
        plt.grid(True, linestyle="--", alpha=0.5)
        combined_plot = f"{OUT_PLOTS_DIR}/walk_forward_combined_equity.png"
        plt.tight_layout()
        plt.savefig(combined_plot)
        plt.close()
        print(f"✅ Combined equity plot saved to: {combined_plot}")

    print("\nAll done.")


if __name__ == "__main__":
    main()