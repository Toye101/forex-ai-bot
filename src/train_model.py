# src/train_model.py
"""
Train ML models to predict EURUSD direction.

Steps:
- Load train/val/test datasets
- Train RandomForest and LogisticRegression with GridSearch
- Evaluate on validation and test
- Save models, confusion matrices, feature importances, and correlation heatmap
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Paths
DATA_DIR = os.path.join("..", "data", "ml_ready")
TRAIN_PATH = os.path.join(DATA_DIR, "EURUSD_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "EURUSD_val.csv")
TEST_PATH = os.path.join(DATA_DIR, "EURUSD_test.csv")

MODELS_DIR = os.path.join("..", "models")
PLOTS_DIR = os.path.join("..", "plots")

# --- New: Correlation Plot Function ---
def plot_feature_correlation(df, feature_cols, save_path):
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature correlation heatmap saved to: {save_path}")

def main():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH, index_col=0, parse_dates=True)
    val_df = pd.read_csv(VAL_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_PATH, index_col=0, parse_dates=True)

    # Features and target
    feature_cols = ["SMA10", "SMA30", "RSI", "MACD"]
    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_val, y_val = val_df[feature_cols], val_df["Target"]
    X_test, y_test = test_df[feature_cols], test_df["Target"]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- New: Save correlation heatmap ---
    plot_feature_correlation(train_df, feature_cols, os.path.join(PLOTS_DIR, "feature_correlation.png"))

    # === RandomForest with GridSearch ===
    print("\n=== RandomForest with GridSearch ===")
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Best parameters:", rf.best_params_)
    print("Best CV accuracy:", rf.best_score_)

    y_val_pred = rf.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Save RandomForest
    joblib.dump(rf.best_estimator_, os.path.join(MODELS_DIR, "RandomForest.pkl"))
    print("RandomForest model saved ✅")

    # Feature importances
    importances = rf.best_estimator_.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_cols, importances)
    plt.title("Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importances_RandomForest.png"))
    plt.close()
    print("Feature importances saved ✅")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.best_estimator_.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - RandomForest (Validation)")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_RandomForest.png"))
    plt.close()

    # Test evaluation
    print("\n=== RandomForest Test Evaluation ===")
    print(classification_report(y_test, rf.predict(X_test)))

    # === LogisticRegression with GridSearch ===
    print("\n=== LogisticRegression with GridSearch ===")
    log_params = {"C": [0.01, 0.1, 1.0], "penalty": ["l2"], "solver": ["lbfgs"]}
    log = GridSearchCV(LogisticRegression(max_iter=500), log_params, cv=3, n_jobs=-1)
    log.fit(X_train, y_train)
    print("Best parameters:", log.best_params_)
    print("Best CV accuracy:", log.best_score_)

    y_val_pred = log.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Save LogisticRegression
    joblib.dump(log.best_estimator_, os.path.join(MODELS_DIR, "LogisticRegression.pkl"))
    print("LogisticRegression model saved ✅")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log.best_estimator_.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - LogisticRegression (Validation)")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_LogisticRegression.png"))
    plt.close()

    # Test evaluation
    print("\n=== LogisticRegression Test Evaluation ===")
    print(classification_report(y_test, log.predict(X_test)))


if __name__ == "__main__":
    main()
