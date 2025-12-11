"""Model classes for UC-01 Permission Anomaly Detection.

Implements ensemble approach per spec:
- Primary: Isolation Forest (fast initial screening)
- Secondary: Autoencoder (accurate confirmation)
- Ensemble: 0.4 × IF_score + 0.6 × AE_reconstruction_error
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.auto_encoder import AutoEncoder
import joblib
from pathlib import Path
from typing import Optional, Tuple
import json


class IsolationForestWrapper:
    """Isolation Forest wrapper with UC-01 spec hyperparameters."""
    
    # Hyperparameter ranges from spec
    PARAM_RANGES = {
        'n_estimators': [100, 200, 500],
        'max_samples': ['auto', 256, 512],
        'contamination': [0.01, 0.05, 0.1],
        'max_features': [0.5, 0.75, 1.0]
    }
    
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        random_state: int = 42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'IsolationForestWrapper':
        """Fit the model."""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        # Negate so higher scores = more anomalous
        return -self.model.score_samples(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1=anomaly, 0=normal."""
        preds = self.model.predict(X)
        # Convert from -1/1 to 1/0
        return (preds == -1).astype(int)


class AutoencoderWrapper:
    """Autoencoder wrapper using PyOD with UC-01 spec hyperparameters."""
    
    # Hyperparameter ranges from spec
    PARAM_RANGES = {
        'encoding_dim': [32, 64, 128],
        'hidden_layers': [[256, 128], [512, 256, 128]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001]
    }
    
    def __init__(
        self,
        encoding_dim: int = 64,
        hidden_neurons: list = None,
        dropout_rate: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        contamination: float = 0.05,
        random_state: int = 42
    ):
        if hidden_neurons is None:
            hidden_neurons = [128, 64]
        
        # PyOD AutoEncoder uses hidden_neuron_list parameter
        self.model = AutoEncoder(
            hidden_neuron_list=hidden_neurons,
            hidden_activation_name='relu',
            epoch_num=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            contamination=contamination,
            random_state=random_state,
            verbose=0
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'AutoencoderWrapper':
        """Fit the model."""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error scores (higher = more anomalous)."""
        return self.model.decision_function(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1=anomaly, 0=normal."""
        return self.model.predict(X)


class PermissionAnomalyDetector:
    """Ensemble anomaly detector per UC-01 spec.
    
    Combines Isolation Forest and Autoencoder with weighted averaging:
    Final Score = 0.4 × IF_score + 0.6 × AE_reconstruction_error
    """
    
    # Feature columns used for modeling
    FEATURE_COLS = [
        'total_permissions', 'permission_velocity_7d', 'permission_velocity_30d',
        'cross_department_ratio', 'sensitivity_weighted_score', 'permission_breadth',
        'admin_permission_flag', 'external_share_ratio', 'off_hours_access_ratio',
        'pct_full_control', 'pct_direct_grants', 'highly_confidential_count',
        'recent_grant_ratio', 'perm_zscore', 'sens_zscore'
    ]
    
    def __init__(
        self,
        if_weight: float = 0.4,
        ae_weight: float = 0.6,
        if_params: dict = None,
        ae_params: dict = None
    ):
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        
        # Initialize models
        self.isolation_forest = IsolationForestWrapper(**(if_params or {}))
        self.autoencoder = AutoencoderWrapper(**(ae_params or {}))
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.score_scaler = MinMaxScaler(feature_range=(0, 100))
        
        # Threshold for anomaly classification
        self.threshold = 50.0  # Default: score >= 50 is anomaly
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'PermissionAnomalyDetector':
        """Fit both models on the data.
        
        Args:
            X: Feature matrix
            y: Optional ground truth labels for threshold calibration
        """
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Fit both models
        print("  Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        print("  Training Autoencoder...")
        self.autoencoder.fit(X_scaled)
        
        # Calibrate score scaler using training data
        raw_scores = self._compute_raw_ensemble_scores(X_scaled)
        self.score_scaler.fit(raw_scores.reshape(-1, 1))
        
        self.is_fitted = True
        return self
    
    def _compute_raw_ensemble_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute raw ensemble scores before scaling."""
        if_scores = self.isolation_forest.score_samples(X_scaled)
        ae_scores = self.autoencoder.score_samples(X_scaled)
        
        # Normalize each component to 0-1 range
        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
        
        # Weighted combination
        return self.if_weight * if_norm + self.ae_weight * ae_norm
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Score samples on 0-100 scale (per spec commercial outcome).
        
        Returns:
            Array of anomaly scores, higher = more anomalous
        """
        X_scaled = self.feature_scaler.transform(X)
        raw_scores = self._compute_raw_ensemble_scores(X_scaled)
        
        # Scale to 0-100
        scores = self.score_scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        return np.clip(scores, 0, 100)
    
    def score_with_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score with individual model components.
        
        Returns:
            (ensemble_score, if_score, ae_score) - all on 0-100 scale
        """
        X_scaled = self.feature_scaler.transform(X)
        
        if_scores = self.isolation_forest.score_samples(X_scaled)
        ae_scores = self.autoencoder.score_samples(X_scaled)
        
        # Normalize to 0-100
        if_norm = 100 * (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        ae_norm = 100 * (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
        
        ensemble = self.score(X)
        
        return ensemble, if_norm, ae_norm
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict anomalies based on threshold.
        
        Args:
            X: Feature matrix
            threshold: Score threshold (default: self.threshold)
        
        Returns:
            Array of predictions: 1=anomaly, 0=normal
        """
        if threshold is None:
            threshold = self.threshold
        
        scores = self.score(X)
        return (scores >= threshold).astype(int)
    
    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly threshold."""
        self.threshold = threshold
    
    def save(self, model_dir: Path, mlflow_run_id: str = None, mlflow_experiment_id: str = None) -> None:
        """Save model artifacts to directory.
        
        Args:
            model_dir: Directory to save model artifacts
            mlflow_run_id: Optional MLflow run ID for correlation
            mlflow_experiment_id: Optional MLflow experiment ID
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn model
        joblib.dump(self.isolation_forest.model, model_dir / 'isolation_forest.pkl')
        joblib.dump(self.feature_scaler, model_dir / 'feature_scaler.pkl')
        joblib.dump(self.score_scaler, model_dir / 'score_scaler.pkl')
        
        # Save PyOD autoencoder
        joblib.dump(self.autoencoder.model, model_dir / 'autoencoder.pkl')
        
        # Save config with MLflow correlation
        config = {
            'if_weight': self.if_weight,
            'ae_weight': self.ae_weight,
            'threshold': self.threshold,
            'feature_cols': self.FEATURE_COLS,
            'mlflow_run_id': mlflow_run_id,
            'mlflow_experiment_id': mlflow_experiment_id,
            'model_path': str(model_dir)
        }
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: Path) -> 'PermissionAnomalyDetector':
        """Load model from directory."""
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        detector = cls(
            if_weight=config['if_weight'],
            ae_weight=config['ae_weight']
        )
        detector.threshold = config['threshold']
        
        # Load models
        detector.isolation_forest.model = joblib.load(model_dir / 'isolation_forest.pkl')
        detector.isolation_forest.is_fitted = True
        
        detector.autoencoder.model = joblib.load(model_dir / 'autoencoder.pkl')
        detector.autoencoder.is_fitted = True
        
        detector.feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
        detector.score_scaler = joblib.load(model_dir / 'score_scaler.pkl')
        
        detector.is_fitted = True
        
        print(f"Model loaded from {model_dir}")
        return detector
