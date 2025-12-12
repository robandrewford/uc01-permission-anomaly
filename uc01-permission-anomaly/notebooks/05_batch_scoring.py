"""
05_batch_scoring.py - Daily Batch Scoring

Scores all users with trained anomaly detection model.
Per spec: "Score all users daily" at 05:00 UTC

Usage:
    uv run python notebooks/05_batch_scoring.py
    uv run python notebooks/05_batch_scoring.py --model-path models/ensemble_20251210_180133_run752f2510/

Per spec: fabric_workload_structure.notebook_workflow[4]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import PermissionAnomalyDetector

# ============================================================================
# Configuration
# ============================================================================

GOLD_DATA_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "latest"

# Feature columns must match training
FEATURE_COLS = PermissionAnomalyDetector.FEATURE_COLS


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score all users with trained anomaly detection model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model directory (default: models/latest/)"
    )
    return parser.parse_args()


# ============================================================================
# Data Loading
# ============================================================================

def load_features() -> pd.DataFrame:
    """Load user features from gold layer."""
    path = GOLD_DATA_DIR / "user_permission_features.parquet"
    return pd.read_parquet(path)


def load_model(model_path: str) -> PermissionAnomalyDetector:
    """Load trained model from specified path."""
    model_path = Path(model_path)
    
    # Resolve symlink if needed
    if model_path.is_symlink():
        resolved = model_path.resolve()
        print(f"  Resolved symlink: {model_path.name} -> {resolved.name}")
        model_path = resolved
    
    return PermissionAnomalyDetector.load(model_path)

def score_users(
    features: pd.DataFrame,
    detector: PermissionAnomalyDetector
) -> pd.DataFrame:
    """Score all users with the trained model.
    
    Returns:
        DataFrame with user_id, scores, and components
    """
    # Prepare feature matrix
    available_cols = [c for c in FEATURE_COLS if c in features.columns]
    X = features[available_cols].fillna(0).values
    
    # Get ensemble scores with components
    ensemble_scores, if_scores, ae_scores = detector.score_with_components(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'user_id': features['user_id'].values,
        'anomaly_score': ensemble_scores,
        'if_score': if_scores,
        'ae_score': ae_scores,
        'is_anomaly': (ensemble_scores >= detector.threshold).astype(int),
        'scored_at': datetime.now()
    })
    
    # Add rank (1 = most anomalous)
    results['anomaly_rank'] = results['anomaly_score'].rank(
        ascending=False, method='dense'
    ).astype(int)
    
    # Sort by score descending
    results = results.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    
    return results


def save_scores(scores: pd.DataFrame) -> None:
    """Save scores to gold layer."""
    GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save current scores
    output_path = GOLD_DATA_DIR / "anomaly_scores.parquet"
    scores.to_parquet(output_path, index=False)
    print(f"  Saved anomaly_scores: {len(scores):,} rows -> {output_path}")
    
    # Append to score history for trend analysis
    history_path = GOLD_DATA_DIR / "score_history.parquet"
    
    # Subset columns for history (to save space)
    history_cols = ['user_id', 'anomaly_score', 'is_anomaly', 'scored_at']
    history_record = scores[history_cols].copy()
    
    if history_path.exists():
        existing_history = pd.read_parquet(history_path)
        updated_history = pd.concat([existing_history, history_record], ignore_index=True)
    else:
        updated_history = history_record
    
    updated_history.to_parquet(history_path, index=False)
    print(f"  Updated score_history: {len(updated_history):,} total rows -> {history_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("05_batch_scoring.py - Daily Batch Scoring")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model path: {args.model_path}")
    print()
    
    # Load model
    print("Loading trained model...")
    detector = load_model(args.model_path)
    print(f"  Threshold: {detector.threshold:.2f}")
    
    # Load features
    print("\nLoading user features...")
    features = load_features()
    print(f"  Users to score: {len(features):,}")
    
    # Score users
    print("\nScoring users...")
    scores = score_users(features, detector)
    
    # Summary statistics
    n_anomalies = scores['is_anomaly'].sum()
    print(f"\nScoring summary:")
    print(f"  Total users: {len(scores):,}")
    print(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/len(scores):.2%})")
    print(f"  Score range: [{scores['anomaly_score'].min():.1f}, {scores['anomaly_score'].max():.1f}]")
    print(f"  Mean score: {scores['anomaly_score'].mean():.2f}")
    
    # Show top 10 anomalous users
    print("\nTop 10 anomalous users:")
    print(scores.head(10)[['user_id', 'anomaly_score', 'anomaly_rank', 'if_score', 'ae_score']].to_string(index=False))
    
    # Save scores
    print("\nSaving to gold layer...")
    save_scores(scores)
    
    print("\n" + "=" * 60)
    print("✅ Batch scoring complete!")
    print(f"Output: {GOLD_DATA_DIR / 'anomaly_scores.parquet'}")
    print("=" * 60)
    print("=" * 60)
