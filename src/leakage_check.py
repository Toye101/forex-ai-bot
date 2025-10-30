import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def run_leakage_test(path, horizon):
    print(f"\n{'='*60}")
    print(f"📘 Testing HORIZON: {horizon}-day model")
    print("="*60)

    df = pd.read_csv(path)
    df = df.dropna()
    print(f"✅ Loaded dataset for horizon {horizon}: {df.shape}")

    # Filter dataset by horizon column name if both horizons are in one file
    # (Optional: comment out if separate datasets)
    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]

    # Features and target
    X = df.drop(columns=["target", "date"], errors="ignore")
    # Flexible target detection
    target_col = None
    for c in df.columns:
       if "target" in c.lower():
          target_col = c
          break

    if target_col is None:
       raise ValueError("❌ No target column found in dataset! Please check column names.")
    else:
       print(f"🎯 Using target column: {target_col}")
       y = df[target_col]


    # --------- A. Chronological split (realistic) ---------
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("\n=== Chronological Split (Realistic) ===")
    model_time = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    model_time.fit(X_train, y_train)
    pred_time = model_time.predict(X_test)
    acc_time = accuracy_score(y_test, pred_time)
    print(f"Accuracy (chronological): {acc_time:.3f}")
    print(classification_report(y_test, pred_time, digits=3))

    # --------- B. Random shuffled split (leakage test) ---------
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("\n=== Shuffled Split (Leakage Check) ===")
    model_rand = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    model_rand.fit(X_train_s, y_train_s)
    pred_rand = model_rand.predict(X_test_s)
    acc_rand = accuracy_score(y_test_s, pred_rand)
    print(f"Accuracy (shuffled): {acc_rand:.3f}")
    print(classification_report(y_test_s, pred_rand, digits=3))

    # --------- C. Comparison ---------
    print("\n📊 Accuracy Comparison")
    print("----------------------")
    print(f"Chronological Accuracy: {acc_time:.3f}")
    print(f"Shuffled Accuracy:      {acc_rand:.3f}")

    if acc_rand > acc_time + 0.05:
        print("⚠️ Potential leakage detected — model might be seeing the future!")
    else:
        print("✅ Model seems clean — no obvious data leakage.")

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    # You can test both processed feature files here if separate
    path_3 = "../data/processed/EURUSD_features.csv"  # Horizon 3-day
    path_5 = "../data/processed/EURUSD_features.csv"  # Horizon 5-day (same if combined)

    run_leakage_test(path_3, horizon=3)
    run_leakage_test(path_5, horizon=5)

    print("\n✅ Leakage test completed for both horizons.")
