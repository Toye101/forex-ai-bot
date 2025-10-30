"""
make_target.py
──────────────────────────────────────────────
Generates forward return and classification targets for the forex AI bot.
Targets represent BUY / SELL signals for specified horizons.
"""

import pandas as pd
import numpy as np


def make_targets(df: pd.DataFrame, horizons=[1, 3, 5]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError("❌ 'close' column missing — cannot compute forward returns.")

    print("\n🎯 Generating forward returns and target signals...")

    # Fixed threshold for small movement filter (0.1%)
    threshold = 0.001

    for h in horizons:
        # forward return after h days
        df[f"fwd_return_{h}"] = df["close"].shift(-h) / df["close"] - 1

        # convert to target: BUY = 1 if > threshold, SELL = 0 otherwise
        df[f"target_{h}"] = np.where(df[f"fwd_return_{h}"] > threshold, 1, 0)

        print(f"✅ Target_{h} created (horizon={h} days, threshold={threshold*100:.2f}%)")

    df = df.dropna().reset_index(drop=True)
    print(f"✅ Final dataset with targets: {df.shape}")

    return df


if __name__ == "__main__":
    print("🔧 Creating model training targets...")
    df = pd.read_csv("../data/processed/EURUSD_features_refined.csv")
    df = make_targets(df, horizons=[1, 3, 5])
    df.to_csv("../data/processed/EURUSD_features_with_targets.csv", index=False)
    print("💾 Saved dataset with targets to ../data/processed/EURUSD_features_with_targets.csv")