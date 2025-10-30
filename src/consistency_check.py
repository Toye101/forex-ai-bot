# consistency_check.py
# Purpose: Verify time-series consistency and model performance across rolling windows.

import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "../data/processed"
HORIZONS = [3, 5]  # Check both horizons

def run_consistency_check(df, target_col, horizon):
    """Run rolling (non-overlapping) consistency check."""
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in features:
        features.remove(target_col)

    print(f"\nTotal features: {len(features)}")

    # Drop missing values
    df = df.dropna(subset=features + [target_col])
    df = df.reset_index(drop=True)

    # Split into 5 roughly equal time periods
    total_len = len(df)
    n_periods = 5
    fold_size = total_len // n_periods
    acc_scores = []

    print(f"\n🧪 Running rolling consistency check ({n_periods} periods)...")

    for i in range(n_periods - 1):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(test_df) == 0:
            break

        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        acc_scores.append(acc)

        print(f"📅 Period {i+1}: {acc:.3f}")

    avg_acc = np.mean(acc_scores)
    print(f"\n📊 Average accuracy across periods: {avg_acc:.3f}")
    print("✅ Consistency check completed.")


if __name__ == "__main__":
    for horizon in HORIZONS:
        print("\n" + "="*60)
        print(f"📘 Consistency Check — Horizon: {horizon}-day model")
        print("="*60)

        file_path = os.path.join(DATA_PATH, f"EURUSD_features_h{horizon}.csv")
        if not os.path.exists(file_path):
            print(f"❌ Missing dataset for horizon {horizon}")
            continue

        df = pd.read_csv(file_path)
        target_col = f"target_{horizon}" if f"target_{horizon}" in df.columns else "target"
        run_consistency_check(df, target_col, horizon)
