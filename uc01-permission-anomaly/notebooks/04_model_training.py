"""
04_model_training.py - Anomaly Model Training

Trains ensemble anomaly detection model per UC-01 spec:
- Isolation Forest (0.4 weight) + Autoencoder (0.6 weight)
- Weekly retraining (Sunday 06:00 UTC per spec)

Per spec: fabric_workload_structure.notebook_workflow[3]
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime
import sys
import json

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import PermissionAnomalyDetector
from evaluation import (
    compute_all_metrics, evaluate_against_targets,
    format_metrics_report, calibrate_threshold, save_metrics_report
)

# ============================================================================
# Configuration
# ============================================================================

GOLD_DATA_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_DIR = PROJECT_ROOT / "models"

# Feature columns for modeling (must match models.py)
FEATURE_COLS = PermissionAnomalyDetector.FEATURE_COLS

# MLflow experiment (from spec)
EXPERIMENT_NAME = "uc01_permission_anomaly_detection"
EXPERIMENT_TAGS = {
    "domain": "access_permissions",
    "capability": "anomaly_detection",
    "outcome": "reduce_security_risk",
    "owner": "data_science_team"
}


# ============================================================================
# Data Loading
# ============================================================================

def load_features() -> pd.DataFrame:
    """Load user features from gold layer."""
    path = GOLD_DATA_DIR / "user_permission_features.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} users from {path}")
    return df


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth labels from synthetic permissions data."""
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    perms_path = synthetic_dir / "fact_permissions.parquet"
    
    if perms_path.exists():
        perms = pd.read_parquet(perms_path)
        # Aggregate to user level - user is anomaly if ANY of their permissions are anomalies
        user_anomaly = perms.groupby('user_id')['is_anomaly'].max().reset_index()
        print(f"Loaded ground truth: {user_anomaly['is_anomaly'].sum()} anomalous users")
        return user_anomaly
    
    return pd.DataFrame()


def prepare_training_data(features: pd.DataFrame) -> tuple:
    """Prepare feature matrix and labels.
    
    Returns:
        (X, y_true, user_ids) - feature matrix, labels if available, user IDs
    """
    # Select feature columns (with fallback for missing)
    available_cols = [c for c in FEATURE_COLS if c in features.columns]
    missing_cols = set(FEATURE_COLS) - set(available_cols)
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
    
    X = features[available_cols].fillna(0).values
    user_ids = features['user_id'].values
    
    # Get ground truth labels from synthetic data
    y_true = None
    ground_truth = load_ground_truth()
    if len(ground_truth) > 0:
        # Join with features on user_id
        features_with_labels = features.merge(ground_truth, on='user_id', how='left')
        y_true = features_with_labels['is_anomaly'].fillna(0).astype(int).values
        print(f"  Ground truth: {y_true.sum()} anomalies ({y_true.mean():.2%})")
    
    return X, y_true, user_ids


# ============================================================================
# Training Functions
# ============================================================================

def train_model(
    X: np.ndarray,
    y_true: np.ndarray = None,
    if_params: dict = None,
    ae_params: dict = None
) -> PermissionAnomalyDetector:
    """Train the ensemble anomaly detector."""
    
    detector = PermissionAnomalyDetector(
        if_weight=0.4,
        ae_weight=0.6,
        if_params=if_params,
        ae_params=ae_params
    )
    
    detector.fit(X, y_true)
    
    return detector


def evaluate_model(
    detector: PermissionAnomalyDetector,
    X: np.ndarray,
    y_true: np.ndarray = None
) -> dict:
    """Evaluate model and compute metrics."""
    
    # Get scores
    scores = detector.score(X)
    
    if y_true is not None:
        # Calibrate threshold for 95% recall
        threshold = calibrate_threshold(y_true, scores, target_recall=0.95)
        detector.set_threshold(threshold)
        
        # Compute all metrics
        metrics = compute_all_metrics(y_true, scores, threshold)
        targets_met = evaluate_against_targets(metrics)
        
        print(format_metrics_report(metrics, targets_met))
        
        return metrics
    else:
        # No labels - just return score statistics
        return {
            'mean_anomaly_score': np.mean(scores),
            'score_std': np.std(scores),
            'anomaly_rate': np.mean(scores >= detector.threshold)
        }


# ============================================================================
# MLflow Experiment
# ============================================================================

def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Set experiment tags
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        for key, value in EXPERIMENT_TAGS.items():
            mlflow.set_experiment_tag(key, value)


def run_experiment(
    X: np.ndarray,
    y_true: np.ndarray,
    run_name: str = None
) -> PermissionAnomalyDetector:
    """Run a training experiment with MLflow tracking and versioned model saving."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_name is None:
        run_name = f"ensemble_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        # Log parameters
        mlflow.log_params({
            "model_type": "Ensemble_IF_AE",
            "if_weight": 0.4,
            "ae_weight": 0.6,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_set_version": "v1.0"
        })
        
        # Train model
        print("\nTraining ensemble model...")
        detector = train_model(X, y_true)
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = evaluate_model(detector, X, y_true)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Generate versioned model directory name
        # Format: {model_type}_{timestamp}_run_{mlflow_run_id[:8]}
        key_metric = f"auc{metrics.get('auc_roc', 0):.2f}".replace(".", "")
        version_name = f"ensemble_{timestamp}_run_{run_id[:8]}"
        versioned_model_dir = MODEL_DIR / version_name
        
        # Save model artifacts to versioned directory
        print(f"\nSaving model to versioned directory: {version_name}")
        versioned_model_dir.mkdir(parents=True, exist_ok=True)
        detector.save(
            versioned_model_dir,
            mlflow_run_id=run_id,
            mlflow_experiment_id=experiment_id
        )
        
        # Save metrics report
        run_info = {
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": experiment_id,
            "model_path": str(versioned_model_dir),
            "timestamp": timestamp
        }
        save_metrics_report(metrics, versioned_model_dir, run_info)
        
        # Update 'latest' symlink
        latest_link = MODEL_DIR / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            import shutil
            shutil.rmtree(latest_link)
        latest_link.symlink_to(version_name)
        print(f"  Updated 'latest' symlink -> {version_name}")
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(str(versioned_model_dir), artifact_path="model")
        
        return detector


# ============================================================================
# Main Execution  
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("04_model_training.py - Anomaly Model Training")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load data
    print("Loading features from gold layer...")
    features = load_features()
    X, y_true, user_ids = prepare_training_data(features)
    
    print(f"\nTraining data shape: {X.shape}")
    if y_true is not None:
        print(f"Ground truth available: {y_true.sum()} anomalies ({y_true.mean():.2%})")
    
    # Setup MLflow
    setup_mlflow()
    
    # Run training experiment
    detector = run_experiment(X, y_true)
    
    print("\n" + "=" * 60)
    print("✅ Model training complete!")
    print(f"Model saved to: {MODEL_DIR / 'latest'}")
    print(f"Threshold: {detector.threshold:.2f}")
    print("=" * 60)