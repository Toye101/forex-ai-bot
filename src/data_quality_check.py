"""
data_quality_check.py
────────────────────────────────────
Performs full data quality diagnostics for the forex AI bot pipeline.
Checks: missing values, variance, correlations, stationarity, and feature importance.
Generates both console output and a text summary report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import datetime

plt.style.use("seaborn-v0_8-whitegrid")

REPORT_PATH = Path("../reports/data_quality_report.txt")


def log(msg, file):
    """Print and log message to report file."""
    print(msg)
    file.write(msg + "\n")


def check_missing_values(df, file):
    log("\n🔍 Missing Value Check", file)
    missing = df.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log("✅ No missing values detected.", file)
    else:
        log(str(missing), file)
    return missing


def check_variance(df, file, threshold=1e-5):
    log("\n📉 Low Variance Feature Check", file)
    variances = df.var()
    low_var = variances[variances < threshold]
    if low_var.empty:
        log("✅ No near-constant features found.", file)
    else:
        log("⚠ Low variance features detected:", file)
        log(str(low_var), file)
    return low_var


def check_correlation(df, file, threshold=0.95):
    log("\n🔗 Correlation Matrix Check", file)
    corr = df.corr()
    high_corr = []
    for col in corr.columns:
        high_corr += [
            (col, other)
            for other in corr.columns
            if col != other and abs(corr.loc[col, other]) > threshold
        ]
    if not high_corr:
        log("✅ No highly correlated features found.", file)
    else:
        log(f"⚠ {len(high_corr)} highly correlated feature pairs (|r| > {threshold}):", file)
        for a, b in high_corr:
            log(f"  {a} ↔ {b} ({corr.loc[a, b]:.2f})", file)

    # Save correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("../plots/feature_correlation_heatmap.png")
    plt.close()
    log("📊 Saved heatmap to ../plots/feature_correlation_heatmap.png", file)

    return high_corr


def check_stationarity(df, file, columns, max_tests=10):
    log("\n📈 ADF Stationarity Test (max 10 features)", file)
    tested = columns[:max_tests]
    for col in tested:
        try:
            result = adfuller(df[col].dropna())
            p_value = result[1]
            status = "✅ Stationary" if p_value < 0.05 else "⚠ Non-stationary"
            log(f"{col:20s} → p-value={p_value:.4f} → {status}", file)
        except Exception as e:
            log(f"{col:20s} → ⚠ ADF test failed ({e})", file)


def feature_importance_diagnostics(df, file, target_col="target_3"):
    log("\n🌲 Feature Importance Diagnostics (RandomForest)", file)

    if target_col not in df.columns:
        log(f"⚠ Target column '{target_col}' not found. Skipping importance check.", file)
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Keep numeric features only
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("\nTop 15 Important Features:", file)
    log(str(importances.head(15)), file)

    # Plot top features
    plt.figure(figsize=(10, 6))
    importances.head(15).plot(kind="barh")
    plt.title("Top 15 Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig("../plots/feature_importance.png")
    plt.close()
    log("📊 Saved importance plot to ../plots/feature_importance.png", file)


def run_diagnostics(filepath="../data/processed/EURUSD_features_regime.csv"):
    print("🔧 Running Data Quality Diagnostics...")
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    numeric_df = df.select_dtypes(include=[np.number])

    # Create reports directory if not exists
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        log("📅 Report generated: " + str(datetime.datetime.now()), file)
        log(f"✅ Loaded dataset: {df.shape}, numeric subset: {numeric_df.shape}", file)

        check_missing_values(df, file)
        check_variance(numeric_df, file)
        check_correlation(numeric_df, file)
        check_stationarity(numeric_df, file, numeric_df.columns)
        feature_importance_diagnostics(df, file)

        log("\n✅ Data quality check completed successfully!", file)

    print(f"\n📄 Full report saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    run_diagnostics()