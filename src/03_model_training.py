import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import joblib
from pathlib import Path
import mlflow

# -----------------------------
# Paths
# -----------------------------
current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
models_dir = current_dir.parent / "models"
models_dir.mkdir(exist_ok=True)
data_path = data_dir / "customer_churn_with_features.csv"

# -----------------------------
# Load features dataset
# -----------------------------
df = pd.read_csv(data_path)
print(f"✅ Loaded data: {df.shape}")
print("Class distribution:\n", df['churn'].value_counts())

X = df.drop(columns=['userId', 'churn', 'churn_score'], errors='ignore')
y = df['churn']

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Stratified K-Fold CV
# -----------------------------
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
accuracy_list, roc_list, f1_list, precision_list, recall_list = [], [], [], [], []

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)
    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    else:
        y_proba = np.zeros_like(y_test, dtype=float)
        if model.classes_[0] == 1:
            y_proba[:] = 1.0

    accuracy_list.append(accuracy_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred, zero_division=0))
    precision_list.append(precision_score(y_test, y_pred, zero_division=0))
    recall_list.append(recall_score(y_test, y_pred, zero_division=0))

    if len(np.unique(y_test)) > 1:
        roc_list.append(roc_auc_score(y_test, y_proba))
    else:
        roc_list.append(np.nan)

# -----------------------------
# CV Metrics
# -----------------------------
print("\n===== Cross-Validated Model Evaluation =====")
print(f"Accuracy:  {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
print(f"ROC-AUC:   {np.nanmean(roc_list):.4f} ± {np.nanstd(roc_list):.4f}")
print(f"F1-score:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
print(f"Recall:    {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
print("===============================\n")

# -----------------------------
# Train final model
# -----------------------------
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
final_model.fit(X_scaled, y)

# -----------------------------
# Save model and scaler locally
# -----------------------------
joblib.dump(final_model, models_dir / "rf_model.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")
joblib.dump(list(X.columns), models_dir / "feature_names.pkl")

print("✅ Final model and scaler saved locally")

# -----------------------------
# MLflow Logging (Default Experiment)
# -----------------------------
with mlflow.start_run():

    # Hyperparameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("cv_folds", 3)

    # Metrics
    mlflow.log_metric("accuracy", np.mean(accuracy_list))
    mlflow.log_metric("roc_auc", np.nanmean(roc_list))
    mlflow.log_metric("f1_score", np.mean(f1_list))
    mlflow.log_metric("precision", np.mean(precision_list))
    mlflow.log_metric("recall", np.mean(recall_list))

    # Artifacts
    mlflow.log_artifact(models_dir / "rf_model.pkl")
    mlflow.log_artifact(models_dir / "scaler.pkl")
    mlflow.log_artifact(models_dir / "feature_names.pkl")
