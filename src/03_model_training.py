# src/03_model_training.py
"""
تدريب نموذج RandomForest للتنبؤ بانسحاب المستخدمين
مع MLflow للتتبع، وحفظ النموذج والـ scaler محليًا
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import mlflow
from pathlib import Path

# -----------------------------
# المسارات
# -----------------------------
current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
models_dir = current_dir.parent / "models"
models_dir.mkdir(exist_ok=True)

data_path = data_dir / "customer_churn_with_features.csv"

# -----------------------------
# تحميل البيانات
# -----------------------------
df = pd.read_csv(data_path)
print(f"✅ Loaded data: {df.shape}")

X = df.drop(columns=['userId','churn','last_auth_status'], errors='ignore')
y = df['churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MLflow Tracking محلي
# -----------------------------
mlflow.set_tracking_uri(f"file://{current_dir.parent / 'mlruns'}")

with mlflow.start_run():
    # تعريف النموذج
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # تسجيل المعلمات
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})

    # تدريب النموذج
    model.fit(X_train, y_train)

    # التنبؤ والتقييم
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # المقاييس الأساسية
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # تسجيل المقاييس
    mlflow.log_metrics({
        "accuracy": acc,
        "roc_auc": roc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    })

    # -----------------------------
    # حفظ النموذج والscaler محليًا فقط
    # -----------------------------
    joblib.dump(model, models_dir / "rf_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    # -----------------------------
    # طباعة النتائج بشكل مرتب
    # -----------------------------
    print("\n===== Model Evaluation =====")
    print(f"Accuracy:  {acc:.4f} | ROC-AUC: {roc:.4f}")
    print(f"F1-score:  {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("===============================\n")
    print("✅ Model and scaler saved locally")
