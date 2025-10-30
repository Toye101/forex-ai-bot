import pandas as pd

# Load your dataset
df = pd.read_csv("../data/processed/EURUSD_features_h5.csv")

# Check the target column values
print(df["target_5"].value_counts())
print("Share of upward moves:", (df["target_5"] > 0).mean())
