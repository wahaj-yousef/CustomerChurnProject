import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path
import joblib

# -----------------------------
# Paths
# -----------------------------
current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
models_dir = current_dir.parent / "models"

new_data_path = data_dir / "new_customer_data.csv"
old_data_path = data_dir / "customer_churn_with_features.csv"
model_path = models_dir / "rf_model.pkl"
scaler_path = models_dir / "scaler.pkl"

# -----------------------------
# Hyperparameters (نفس الإعدادات من 03)
# -----------------------------
N_ESTIMATORS = 100
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 5
DRIFT_THRESHOLD = 0.05      # الـ p-value للـ drift
DRIFT_FEATURES_LIMIT = 3    # إذا drift في 3+ features → retrain

# -----------------------------
# Load data
# -----------------------------
df_old = pd.read_csv(old_data_path)

try:
    df_new = pd.read_csv(new_data_path)
except FileNotFoundError:
    print(f"⚠️ File not found: {new_data_path}")
    df_new = None

model = joblib.load(model_path)

features = df_old.drop(
    columns=["userId", "churn", "last_auth_status"], errors="ignore"
).columns

# -----------------------------
# Drift Detection
# -----------------------------
if df_new is not None:
    drift_results = {}
    for col in features:
        if col in df_new.columns:
            stat, p = ks_2samp(df_old[col], df_new[col])
            drift_results[col] = p
        else:
            print(f"⚠️ Column '{col}' does not exist in the new data.")

    print("📊 Drift test results for each feature (p-value):")
    for k, v in drift_results.items():
        print(f"{k}: {v:.4f}")

    drift_detected = {k: v for k, v in drift_results.items() if v < DRIFT_THRESHOLD}
    print("\n🚨 Features with potential drift (p < 0.05):")
    if drift_detected:
        for k, v in drift_detected.items():
            print(f"{k}: p={v:.4f}")
    else:
        print("No features with potential drift detected.")

    # -----------------------------
    # Retraining Decision
    # -----------------------------
    if len(drift_detected) >= DRIFT_FEATURES_LIMIT:
        print(f"\n🔄 Drift detected in {len(drift_detected)} features. Starting retraining...")

        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        print(f"📊 Combined dataset size: {df_combined.shape[0]} rows")

        X_new = df_combined.drop(columns=["userId", "churn", "last_auth_status"], errors="ignore")
        y_new = df_combined["churn"]

        new_scaler = StandardScaler()
        X_new_scaled = new_scaler.fit_transform(X_new)

        new_model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=42
        )
        new_model.fit(X_new_scaled, y_new)

        new_probs = new_model.predict_proba(X_new_scaled)[:, 1]
        new_auc = roc_auc_score(y_new, new_probs)
        print(f"📈 New model ROC-AUC: {new_auc:.4f}")

        old_scaler = joblib.load(scaler_path)
        X_old_scaled = old_scaler.transform(X_new)
        old_probs = model.predict_proba(X_old_scaled)[:, 1]
        old_auc = roc_auc_score(y_new, old_probs)
        print(f"📈 Old model ROC-AUC: {old_auc:.4f}")

        if new_auc >= old_auc:
            joblib.dump(new_model, model_path)
            joblib.dump(new_scaler, scaler_path)
            print("✅ New model is better. Model updated successfully!")
        else:
            print("⚠️ New model is not better. Keeping old model.")

    else:
        print(f"\n✅ Drift in {len(drift_detected)} features only. No retraining needed.")

else:
    print("⚠️ Drift not calculated because new data is not available.")
