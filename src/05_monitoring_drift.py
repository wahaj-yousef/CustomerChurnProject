# 05_data_drift_check.py
import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path
import joblib

# -----------------------------
# المسارات الديناميكية
# -----------------------------
current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
models_dir = current_dir.parent / "models"

new_data_path = data_dir / "new_customer_data.csv"
old_data_path = data_dir / "customer_churn_with_features.csv"
model_path = models_dir / "rf_model.pkl"

# -----------------------------
# تحميل البيانات
# -----------------------------
df_new = pd.read_csv(new_data_path)
df_old = pd.read_csv(old_data_path)

# تحميل النموذج (لو حبيت تستخدمه لاحقًا)
model = joblib.load(model_path)

# استخدام نفس المميزات
features = df_old.drop(columns=['userId','churn','last_auth_status'], errors='ignore').columns

# -----------------------------
# حساب Drift لكل feature
# -----------------------------
drift_results = {}
for col in features:
    if col in df_new.columns:
        stat, p = ks_2samp(df_old[col], df_new[col])
        drift_results[col] = p
    else:
        print(f"⚠️ العمود '{col}' غير موجود في البيانات الجديدة.")

# طباعة النتائج
print("📊 نتائج اختبار drift لكل ميزة (p-value):")
for k, v in drift_results.items():
    print(f"{k}: {v:.4f}")

# الأعمدة اللي فيها drift محتمل
drift_detected = {k:v for k,v in drift_results.items() if v < 0.05}
print("\n🚨 الميزات اللي فيها احتمال drift (p < 0.05):")
if drift_detected:
    for k,v in drift_detected.items():
        print(f"{k}: p={v:.4f}")
else:
    print("لا توجد ميزات بها drift محتمل.")
