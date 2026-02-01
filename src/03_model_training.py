import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
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
print("Class distribution:\n", df["churn"].value_counts())

X = df.drop(columns=["userId", "churn", "churn_score"], errors="ignore")
y = df["churn"]

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Define Hyperparameters
# -----------------------------
MODEL_TYPE = "RandomForest"
N_ESTIMATORS = 100
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 5
CV_FOLDS = 3
THRESHOLD = 0.3  

# -----------------------------
# Stratified K-Fold CV
# -----------------------------
kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
accuracy_list, roc_list, f1_list, precision_list, recall_list = [], [], [], [], []

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize model with hyperparameters
    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict probabilities and apply threshold
    y_probs = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else np.zeros_like(y_test)
    y_pred = (y_probs > THRESHOLD).astype(int)

    # Evaluate metrics
    accuracy_list.append(accuracy_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred, zero_division=0))
    precision_list.append(precision_score(y_test, y_pred, zero_division=0))
    recall_list.append(recall_score(y_test, y_pred, zero_division=0))
    roc_list.append(roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else np.nan)

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
    class_weight="balanced",
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42
)
final_model.fit(X_scaled, y)
# -----------------------------
# Feature Importance
# -----------------------------
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

print("\n===== Feature Importance =====")
print(feature_importance_df.to_string(index=False))
print("==============================\n")

# Save feature importance as CSV
feature_importance_df.to_csv(data_dir / "feature_importance.csv", index=False)
print("✅ Feature importance saved to data/feature_importance.csv")

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
mlflow.set_tracking_uri("http://127.0.0.1:1235")

with mlflow.start_run():

    # Hyperparameters
    mlflow.log_param("model_type", MODEL_TYPE)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("threshold", THRESHOLD)  # Log the threshold used

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
