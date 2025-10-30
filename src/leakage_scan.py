# leakage_scan.py
import pandas as pd
import numpy as np
import os

DATA_PATH = "../data/processed"
HORIZONS = [3, 5]

for horizon in HORIZONS:
    print("\n" + "="*60)
    print(f"🔍 Checking correlation leakage for horizon={horizon}")
    print("="*60)
    
    file_path = os.path.join(DATA_PATH, f"EURUSD_features_h{horizon}.csv")
    df = pd.read_csv(file_path)

    target_col = f"target_{horizon}"
    if target_col not in df.columns:
        continue

    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()[target_col].sort_values(key=np.abs, ascending=False)
    print(corr.head(10))
