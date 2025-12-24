import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
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
# تتبع MLflow
# -----------------------------
with mlflow.start_run():
    # تعريف النموذج
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # تسجيل الـ hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # تدريب النموذج
    model.fit(X_train, y_train)

    # التنبؤ والتقييم
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)

    # تسجيل المقاييس
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)

    # تسجيل النموذج
    mlflow.sklearn.log_model(model, "rf_model")

    # حفظ نسخة محلية للنموذج والscaler
    joblib.dump(model, models_dir / "rf_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print("✅ Model and scaler saved locally")
