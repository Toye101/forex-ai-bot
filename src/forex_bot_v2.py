# forex_bot_v2.py  (UPDATED for Option 1: train with regime features)
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os

warnings.filterwarnings("ignore")

OUT_DIR = "../data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    # drop the usual index column if present
    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])
    print(f"✅ Loaded dataset: {df.shape}")
    return df


def get_numeric_feature_frame(df, drop_target_prefix="target_", drop_fwd_prefix="fwd_return_"):
    """
    Return dataframe with only numeric features suitable for training / scaling.
    This will keep numeric 'regime' if present (e.g. regime_encoded or regime numeric).
    It will drop textual regime (object dtype) automatically.
    """
    drop_cols = [c for c in df.columns if c.startswith(drop_target_prefix) or c.startswith(drop_fwd_prefix)]
    # don't blindly drop a numeric regime; only drop textual 'regime' columns (object dtype)
    if "regime" in df.columns and df["regime"].dtype == object:
        drop_cols.append("regime")

    X = df.drop(columns=drop_cols, errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    return X_numeric


def train_with_cv_and_save(df, horizon_label=3, n_splits=5):
    target_col = f"target_{horizon_label}"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Build numeric feature matrix (this will keep numeric regime features like 'regime_encoded' or numeric 'regime')
    X = get_numeric_feature_frame(df)
    y = df[target_col]

    # Ensure alignment: drop rows with NaN in X or y
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[X.columns]
    y = combined[target_col]

    feature_names = X.columns.tolist()
    print(f"\n🎯 Target column: {target_col}")
    print(f"✅ Using {len(feature_names)} numeric features (first 12): {feature_names[:12]}")

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    print("\n--- Running TimeSeries CV ---")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        cv_scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")
        print(classification_report(y_val, preds, digits=3))

    print("------------------------------------------------------------")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    print("------------------------------------------------------------")

    # Retrain final model on full dataset (X_scaled, y)
    final_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    final_model.fit(X_scaled, y)

    # Save scaler, model (LightGBM booster), and feature list
    scaler_path = f"{OUT_DIR}/scaler_h{horizon_label}.save"
    model_path = f"{OUT_DIR}/lgb_h{horizon_label}_model.txt"
    feat_path = f"{OUT_DIR}/feature_names_h{horizon_label}.pkl"

    joblib.dump(scaler, scaler_path)
    # Save the LightGBM booster binary (keeps feature names internally if training used DataFrame with column names)
    final_model.booster_.save_model(model_path)
    joblib.dump(feature_names, feat_path)

    # attach feature names for backtest usage (helpful later)
    final_model.feature_names_ = feature_names

    print(f"✅ Saved scaler -> {scaler_path}")
    print(f"✅ Saved LightGBM model -> {model_path}")
    print(f"✅ Saved feature list -> {feat_path}")

    return final_model, scaler


def backtest_walkforward(model, scaler, df, horizon_label=3, reward_risk=(1, 3)):
    # This is a reasonable walk-forward backtest that uses only numeric features aligned with model
    target_col = f"target_{horizon_label}"
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Prepare numeric X and y (consistent with training)
    X = get_numeric_feature_frame(df)
    y = df[target_col].values

    # Align with model feature names (if present)
    model_features = getattr(model, "feature_names_", X.columns.tolist())
    available_features = [f for f in model_features if f in X.columns]
    if not available_features:
        raise ValueError("No model features found in dataset for backtest.")
    X = X[available_features]

    print(f"🧩 Backtest using {len(available_features)} features.")

    window = 200
    step = 200
    preds_list = []
    idx_list = []

    for start in range(window, len(X) - step, step):
        end = start + step
        X_train, y_train = X.iloc[:start], y[:start]
        X_valid, y_valid = X.iloc[start:end], y[start:end]

        # Fit scaler on-train (walk-forward retrain), transform validation
        X_train_s = scaler.fit_transform(X_train)
        X_valid_s = scaler.transform(X_valid)

        model.fit(X_train_s, y_train)
        preds = model.predict(X_valid_s)

        acc = accuracy_score(y_valid, preds)
        print(f"Segment {start:5d}-{end:5d}: accuracy = {acc:.4f}")

        preds_list.append(preds)
        idx_list.append(np.arange(start, end))

    preds_all = np.concatenate(preds_list)
    valid_idx = np.concatenate(idx_list)

    backtest_df = df.iloc[valid_idx].copy()
    backtest_df["pred"] = preds_all
    backtest_df["signal"] = np.where(backtest_df["pred"] == 1, "BUY", "SELL")

    acc_overall = accuracy_score(backtest_df[target_col], preds_all)
    print("\nOverall backtest accuracy:", f"{acc_overall:.4f}")
    print(classification_report(backtest_df[target_col], preds_all))

    # Simulate trades (simple reward-risk)
    rr = reward_risk[1]
    if "close" not in backtest_df.columns:
        backtest_df["close"] = df["close"].iloc[-len(backtest_df):].values

    entry_prices = backtest_df["close"].values
    signals = backtest_df["signal"].values

    wins = losses = 0
    pnl = 0.0
    trade_returns = []
    for i in range(len(signals) - horizon_label):
        entry = entry_prices[i]
        exit_p = entry_prices[i + horizon_label]
        signal = signals[i]
        if np.isnan(entry) or np.isnan(exit_p):
            change = 0.0
        elif signal == "BUY":
            change = (exit_p - entry) / entry
        else:
            change = (entry - exit_p) / entry

        if change > 0:
            profit = rr * abs(change)
            wins += 1
        else:
            profit = -abs(change)
            losses += 1
        trade_returns.append(profit)
        pnl += profit

    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    print("------------------------------------------------------------")
    print(f"Total trades: {total}, Win rate: {win_rate*100:.2f}% , PnL: {pnl:.3f}")
    print("------------------------------------------------------------")

    save_path = f"{OUT_DIR}/backtest_results_h{horizon_label}.csv"
    backtest_df["trade_return"] = trade_returns + [np.nan] * (len(backtest_df) - len(trade_returns))
    backtest_df.to_csv(save_path, index=False)
    print("Saved backtest results to:", save_path)


def main():
    print("🚀 forex_bot_v2.py pipeline started...")
    path = "../data/processed/EURUSD_features_with_targets.csv"
    df = load_data(path)

    horizon = 3
    model, scaler = train_with_cv_and_save(df, horizon_label=horizon, n_splits=5)
    backtest_walkforward(model, scaler, df, horizon_label=horizon, reward_risk=(1, 3))


if __name__ == "__main__":
    main()