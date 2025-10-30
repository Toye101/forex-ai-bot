import pandas as pd

df = pd.read_csv("../data/processed/EURUSD_features_h5.csv")

# Count how many features are constant
numeric = df.select_dtypes(include='number')
constant_features = [c for c in numeric.columns if numeric[c].nunique() <= 1]

print(f"Total features: {len(numeric.columns)}")
print(f"Constant features ({len(constant_features)}): {constant_features}")

# Also check for NaN ratio
nan_ratio = df.isna().mean().sort_values(ascending=False)
print("\nTop 10 columns by NaN ratio:")
print(nan_ratio.head(10))
