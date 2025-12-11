"""Train anomaly detection models per UC-01 spec."""
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder
import joblib

# MLflow setup per spec
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("uc01_permission_anomaly_detection")

# Load features
features = pd.read_parquet('data/synthetic/user_features.parquet')

# Feature columns for modeling
FEATURE_COLS = [
    'total_permissions', 'permission_velocity_7d', 'permission_velocity_30d',
    'cross_department_ratio', 'sensitivity_weighted_score', 'permission_breadth',
    'admin_permission_flag', 'pct_full_control', 'pct_direct_grants',
    'highly_confidential_count', 'recent_grant_ratio', 'perm_zscore', 'sens_zscore'
]

X = features[FEATURE_COLS].fillna(0)
y_true = features['is_anomaly'] if 'is_anomaly' in features.columns else None

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter grid from spec
CONTAMINATION_VALUES = [0.01, 0.05, 0.1]
N_ESTIMATORS_VALUES = [100, 200, 500]

best_model = None
best_score = 0

for contamination in CONTAMINATION_VALUES:
    for n_estimators in N_ESTIMATORS_VALUES:
        with mlflow.start_run(run_name=f"IF_c{contamination}_n{n_estimators}"):
            
            # Log parameters
            mlflow.log_params({
                "model_type": "IsolationForest",
                "contamination": contamination,
                "n_estimators": n_estimators,
                "feature_count": len(FEATURE_COLS)
            })
            
            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            # Score (lower = more anomalous)
            scores = -model.score_samples(X_scaled)  # Negate so higher = more anomalous
            predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
            
            # Evaluation metrics
            anomaly_rate = (predictions == -1).mean()
            
            if y_true is not None:
                from sklearn.metrics import precision_score, recall_score, roc_auc_score
                y_pred = (predictions == -1).astype(int)
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, scores)
                
                mlflow.log_metrics({
                    "precision": precision,
                    "recall": recall,
                    "auc_roc": auc,
                    "anomaly_rate": anomaly_rate
                })
                
                if auc > best_score:
                    best_score = auc
                    best_model = model
            else:
                mlflow.log_metric("anomaly_rate", anomaly_rate)
            
            # Log model
            mlflow.sklearn.log_model(model, "isolation_forest")

# Save best model
joblib.dump(best_model, 'models/isolation_forest_best.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print(f"Best AUC-ROC: {best_score:.4f}")