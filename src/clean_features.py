import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os

def adf_test(series):
    """Return True if stationary, False if not."""
    try:
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 → stationary
    except:
        return False


def clean_features(df: pd.DataFrame, corr_thresh=0.95, var_thresh=1e-5):
    report = []
    df = df.copy()

    # === 0. Keep only numeric columns for analysis ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    df = df[numeric_cols]

    report.append(f"🧭 Ignored non-numeric columns: {non_numeric_cols}")

    # === 1. Handle Missing Values ===
    nan_pct = df.isna().mean()
    too_many_nans = nan_pct[nan_pct > 0.2].index.tolist()
    df = df.drop(columns=too_many_nans, errors="ignore")

    df = df.ffill().bfill()

    report.append(f"🧹 Dropped columns with >20% NaN: {too_many_nans}")
    report.append(f"✅ Filled remaining NaNs forward/backward")

    # === 2. Drop Low Variance Columns ===
    variances = df.var(numeric_only=True)
    low_var_cols = variances[variances < var_thresh].index.tolist()
    df = df.drop(columns=low_var_cols, errors="ignore")
    report.append(f"⚙ Dropped low-variance columns: {low_var_cols}")

    # === 3. Drop Highly Correlated Columns ===
    # === 3. Drop Highly Correlated Columns ===
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > corr_thresh)]

    # ⚠ Keep 'close' column for trading simulation
    if "close" in high_corr:
        high_corr.remove("close")

    df = df.drop(columns=high_corr, errors="ignore")
    report.append(f"🔗 Dropped highly correlated columns (kept 'close'): {high_corr}")
    
    # === 4. Stationarity Check (ADF Test) ===
    non_stationary = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if not adf_test(df[col]):
            df[col] = df[col].diff().fillna(0)
            non_stationary.append(col)
    report.append(f"📉 Differenced non-stationary features: {non_stationary}")

    # === 5. Final Check ===
    final_shape = df.shape
    report.append(f"✅ Final dataset shape: {final_shape}")

    return df, report


if __name__ == "__main__":
    print("🔧 Running data cleaning pipeline...")

    input_path = "../data/processed/EURUSD_features.csv"
    output_path = "../data/processed/EURUSD_features_clean.csv"
    report_path = "../reports/data_quality_clean_report.txt"

    os.makedirs("../reports", exist_ok=True)

    df = pd.read_csv(input_path)
    df_clean, report = clean_features(df)

    df_clean.to_csv(output_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
       f.write("\n".join(report))

    print("\n".join(report))
    print(f"\n✅ Cleaned data saved to {output_path}")
    print(f"📝 Report saved to {report_path}")