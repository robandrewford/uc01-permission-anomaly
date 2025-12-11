"""
03_feature_engineering.py - Feature Engineering (Gold Layer)

This notebook computes derived features from silver layer data.
Per UC-01 spec: fabric_workload_structure.notebook_workflow[2]

Gold layer tables:
- user_permission_features: Aggregated features per user for anomaly detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path for importing features module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import compute_permission_features, compute_peer_baselines

# ============================================================================
# Configuration
# ============================================================================

SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_DIR = PROJECT_ROOT / "data" / "gold"

# Encoding mappings per UC-01 spec
PERMISSION_LEVEL_ORDINAL = {
    'Read': 1,
    'Contribute': 2,
    'Full Control': 3
}


# ============================================================================
# Data Loading
# ============================================================================

def load_silver_layer() -> dict[str, pd.DataFrame]:
    """Load all silver layer tables."""
    return {
        "permissions": pd.read_parquet(SILVER_DATA_DIR / "cleaned_permissions.parquet"),
        "access_events": pd.read_parquet(SILVER_DATA_DIR / "cleaned_access_events.parquet"),
        "velocity": pd.read_parquet(SILVER_DATA_DIR / "permission_velocity.parquet"),
        "users": pd.read_parquet(SILVER_DATA_DIR / "dim_users.parquet"),
        "resources": pd.read_parquet(SILVER_DATA_DIR / "dim_resources.parquet"),
    }


# ============================================================================
# Additional Feature Functions
# ============================================================================

def compute_external_share_ratio(permissions: pd.DataFrame) -> pd.DataFrame:
    """Compute external share ratio per user.
    
    Formula from spec: COUNT(external_shares) / COUNT(total_shares)
    """
    if 'is_external' not in permissions.columns:
        print("  Warning: is_external column not found, returning zeros")
        return pd.DataFrame({
            'user_id': permissions['user_id'].unique(),
            'external_share_ratio': 0.0
        })
    
    share_stats = permissions.groupby('user_id').agg(
        total_shares=('is_external', 'count'),
        external_shares=('is_external', 'sum')
    ).reset_index()
    
    share_stats['external_share_ratio'] = (
        share_stats['external_shares'] / share_stats['total_shares']
    ).fillna(0)
    
    return share_stats[['user_id', 'external_share_ratio']]


def compute_off_hours_access_ratio(access_events: pd.DataFrame) -> pd.DataFrame:
    """Compute off-hours access ratio per user.
    
    Formula from spec: COUNT(access_outside_business_hours) / COUNT(total_access)
    Off-hours defined as outside 8am-6pm local time.
    """
    if 'is_off_hours' not in access_events.columns:
        # Compute from timestamp if not pre-computed
        access_events = access_events.copy()
        access_events['event_timestamp'] = pd.to_datetime(access_events['event_timestamp'])
        hour = access_events['event_timestamp'].dt.hour
        access_events['is_off_hours'] = (hour < 8) | (hour >= 18)
    
    access_stats = access_events.groupby('user_id').agg(
        total_access=('is_off_hours', 'count'),
        off_hours_access=('is_off_hours', 'sum')
    ).reset_index()
    
    access_stats['off_hours_access_ratio'] = (
        access_stats['off_hours_access'] / access_stats['total_access']
    ).fillna(0)
    
    return access_stats[['user_id', 'off_hours_access_ratio']]


def apply_encoding(features: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Apply encoding strategies per UC-01 spec.
    
    - user_role: Target encoding with smoothing (simplified to one-hot for now)
    - department: Target encoding with smoothing (simplified to one-hot for now)
    - permission_type: Ordinal encoding (Read < Contribute < Full Control)
    """
    # For now, we keep categorical columns as-is for model training
    # In production, apply target encoding with smoothing (m=10)
    
    # Add role from users table if not present
    if 'role' not in features.columns:
        user_roles = users[['user_id', 'role']].drop_duplicates()
        features = features.merge(user_roles, on='user_id', how='left')
    
    return features


def compute_all_features(
    permissions: pd.DataFrame,
    access_events: pd.DataFrame,
    users: pd.DataFrame,
    resources: pd.DataFrame
) -> pd.DataFrame:
    """Compute all derived features from spec."""
    
    # Get reference date from data
    permissions = permissions.copy()
    permissions['granted_date'] = pd.to_datetime(permissions['granted_date'])
    reference_date = permissions['granted_date'].max()
    
    print(f"  Reference date: {reference_date}")
    
    # Core features from features.py
    print("  Computing core permission features...")
    features = compute_permission_features(permissions, users, resources, reference_date)
    print(f"    Generated {len(features)} user feature rows")
    
    # Add peer baselines
    print("  Computing peer baselines...")
    features = compute_peer_baselines(features)
    
    # Add external share ratio (now we have is_external!)
    print("  Computing external share ratio...")
    external_ratio = compute_external_share_ratio(permissions)
    features = features.merge(external_ratio, on='user_id', how='left', suffixes=('', '_new'))
    # Use new computed value instead of placeholder
    if 'external_share_ratio_new' in features.columns:
        features['external_share_ratio'] = features['external_share_ratio_new']
        features = features.drop(columns=['external_share_ratio_new'])
    
    # Add off-hours access ratio
    print("  Computing off-hours access ratio...")
    off_hours_ratio = compute_off_hours_access_ratio(access_events)
    features = features.merge(off_hours_ratio, on='user_id', how='left')
    features['off_hours_access_ratio'] = features['off_hours_access_ratio'].fillna(0)
    
    # Apply encodings
    print("  Applying encodings...")
    features = apply_encoding(features, users)
    
    # Add metadata
    features['feature_computed_at'] = datetime.now()
    features['reference_date'] = reference_date
    
    return features


def save_gold_layer(features: pd.DataFrame) -> None:
    """Save feature table to gold layer."""
    GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = GOLD_DATA_DIR / "user_permission_features.parquet"
    features.to_parquet(output_path, index=False)
    print(f"  Saved user_permission_features: {len(features):,} rows -> {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("03_feature_engineering.py - Feature Engineering")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    print("Loading silver layer...")
    silver = load_silver_layer()
    for name, df in silver.items():
        print(f"  {name}: {len(df):,} rows")
    
    print("\nComputing features...")
    features = compute_all_features(
        permissions=silver['permissions'],
        access_events=silver['access_events'],
        users=silver['users'],
        resources=silver['resources']
    )
    
    print(f"\nFeature summary:")
    print(f"  Total users: {len(features):,}")
    print(f"  Feature columns: {len(features.columns)}")
    print(f"  Features: {list(features.columns)}")
    
    print("\nSaving to gold layer...")
    save_gold_layer(features)
    
    print("\n✅ Feature engineering complete!")
    print(f"Output directory: {GOLD_DATA_DIR}")
    
    # Show sample statistics
    print("\n" + "=" * 60)
    print("Feature Statistics (sample)")
    print("=" * 60)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    print(features[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])
