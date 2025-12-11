"""
06_alert_generation.py - Alert Queue Generation

Generates prioritized alert queue from anomaly scores.
Per spec: fabric_workload_structure.output_schema.alerts_table

Per spec: fabric_workload_structure.notebook_workflow[5]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import PermissionAnomalyDetector

# ============================================================================
# Configuration
# ============================================================================

GOLD_DATA_DIR = PROJECT_ROOT / "data" / "gold"
SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"

# Severity thresholds
SEVERITY_THRESHOLDS = {
    'Critical': 90,
    'High': 75,
    'Medium': 50,
    'Low': 0
}

# Anomaly type detection thresholds (based on feature z-scores)
ANOMALY_TYPE_RULES = {
    'permission_spike': ('permission_velocity_7d', 2.0),
    'sensitivity_escalation': ('highly_confidential_count', 2.0),
    'cross_department_anomaly': ('cross_department_ratio', 2.0),
    'external_share_spike': ('external_share_ratio', 2.0),
    'off_hours_anomaly': ('off_hours_access_ratio', 2.0),
    'admin_grant_anomaly': ('admin_permission_flag', 0.5)
}

# Recommended actions by anomaly type
RECOMMENDED_ACTIONS = {
    'permission_spike': 'Review recent permission grants and revoke unnecessary access',
    'sensitivity_escalation': 'Verify business justification for Highly Confidential access',
    'cross_department_anomaly': 'Confirm cross-department access is appropriate for role',
    'external_share_spike': 'Audit external sharing links and revoke if not justified',
    'off_hours_anomaly': 'Investigate off-hours access patterns for potential compromise',
    'admin_grant_anomaly': 'Verify admin permissions are appropriate for non-IT user',
    'general_anomaly': 'Review overall permission profile for this user'
}


# ============================================================================
# Data Loading
# ============================================================================

def load_scores() -> pd.DataFrame:
    """Load anomaly scores from gold layer."""
    return pd.read_parquet(GOLD_DATA_DIR / "anomaly_scores.parquet")


def load_user_features() -> pd.DataFrame:
    """Load user features for explanation generation."""
    return pd.read_parquet(GOLD_DATA_DIR / "user_permission_features.parquet")


def load_user_info() -> pd.DataFrame:
    """Load user dimension data for email and department."""
    return pd.read_parquet(SILVER_DATA_DIR / "dim_users.parquet")


# ============================================================================
# Alert Generation Functions
# ============================================================================

def determine_severity(score: float) -> str:
    """Determine alert severity based on score."""
    for severity, threshold in SEVERITY_THRESHOLDS.items():
        if score >= threshold:
            return severity
    return 'Low'


def determine_anomaly_type(
    user_features: pd.Series,
    dept_means: pd.Series,
    dept_stds: pd.Series
) -> tuple:
    """Determine primary anomaly type based on feature deviations.
    
    Returns:
        (primary_type, top_contributing_features)
    """
    feature_deviations = {}
    
    for anomaly_type, (feature, threshold) in ANOMALY_TYPE_RULES.items():
        if feature in user_features.index:
            value = user_features[feature]
            mean = dept_means.get(feature, 0)
            std = dept_stds.get(feature, 1)
            
            if std > 0:
                z_score = (value - mean) / std
            else:
                z_score = 0
            
            if z_score >= threshold:
                feature_deviations[anomaly_type] = z_score
    
    if feature_deviations:
        # Primary type is the one with highest z-score
        primary_type = max(feature_deviations, key=feature_deviations.get)
        
        # Top contributing features (sorted by deviation)
        top_features = sorted(
            [(ANOMALY_TYPE_RULES[t][0], d) for t, d in feature_deviations.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_feature_names = [f[0] for f in top_features]
        
        return primary_type, top_feature_names
    
    return 'general_anomaly', ['anomaly_score']


def generate_explanation(
    primary_type: str,
    user_features: pd.Series,
    top_features: list
) -> str:
    """Generate human-readable explanation for the alert."""
    explanations = {
        'permission_spike': f"User gained {user_features.get('permission_velocity_7d', 0)*7:.0f} new permissions in the last 7 days",
        'sensitivity_escalation': f"User has {user_features.get('highly_confidential_count', 0):.0f} Highly Confidential permissions",
        'cross_department_anomaly': f"User has {user_features.get('cross_department_ratio', 0)*100:.1f}% cross-department access",
        'external_share_spike': f"User has {user_features.get('external_share_ratio', 0)*100:.1f}% external sharing rate",
        'off_hours_anomaly': f"User has {user_features.get('off_hours_access_ratio', 0)*100:.1f}% off-hours access",
        'admin_grant_anomaly': "User has admin-level permissions outside IT department",
        'general_anomaly': f"User's permission profile deviates significantly from peers"
    }
    
    base = explanations.get(primary_type, "Anomalous permission pattern detected")
    
    if top_features and len(top_features) > 1:
        additional = f". Additional contributing factors: {', '.join(top_features[1:])}"
        return base + additional
    
    return base


def generate_alerts(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    user_info: pd.DataFrame,
    threshold: float = 50.0
) -> pd.DataFrame:
    """Generate alert queue from scores and features."""
    
    # Filter to anomalies above threshold
    anomalies = scores[scores['anomaly_score'] >= threshold].copy()
    
    if len(anomalies) == 0:
        print("  No anomalies detected above threshold")
        return pd.DataFrame()
    
    # Merge with features and user info
    anomalies = anomalies.merge(features, on='user_id', how='left')
    anomalies = anomalies.merge(
        user_info[['user_id', 'department', 'role']],
        on='user_id',
        how='left',
        suffixes=('', '_info')
    )
    
    # Compute department statistics for z-score calculation
    feature_cols = [c for c in features.columns if c in [r[0] for r in ANOMALY_TYPE_RULES.values()]]
    dept_stats = features.groupby('department')[feature_cols].agg(['mean', 'std'])
    
    # Generate alert records
    alerts = []
    for _, row in anomalies.iterrows():
        user_id = row['user_id']
        dept = row.get('department', 'Unknown')
        
        # Get department statistics
        if dept in dept_stats.index:
            dept_means = dept_stats.loc[dept].xs('mean', level=1)
            dept_stds = dept_stats.loc[dept].xs('std', level=1)
        else:
            dept_means = pd.Series()
            dept_stds = pd.Series()
        
        # Determine anomaly type and contributing features
        user_features = row[feature_cols] if len(feature_cols) > 0 else pd.Series()
        primary_type, top_features = determine_anomaly_type(user_features, dept_means, dept_stds)
        
        # Generate explanation
        explanation = generate_explanation(primary_type, row, top_features)
        
        # Get recommended action
        action = RECOMMENDED_ACTIONS.get(primary_type, RECOMMENDED_ACTIONS['general_anomaly'])
        
        alerts.append({
            'alert_id': str(uuid.uuid4()),
            'user_id': user_id,
            'user_email': f"{user_id[:8]}@example.com",  # Placeholder - would come from user directory
            'user_department': dept,
            'anomaly_score': row['anomaly_score'],
            'anomaly_rank': row['anomaly_rank'],
            'primary_anomaly_type': primary_type,
            'explanation': explanation,
            'top_contributing_features': top_features,
            'recommended_action': action,
            'created_timestamp': datetime.now(),
            'severity': determine_severity(row['anomaly_score'])
        })
    
    alert_df = pd.DataFrame(alerts)
    
    # Sort by score descending
    alert_df = alert_df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    
    return alert_df


def save_alerts(alerts: pd.DataFrame) -> None:
    """Save alert queue to gold layer."""
    GOLD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = GOLD_DATA_DIR / "alert_queue.parquet"
    alerts.to_parquet(output_path, index=False)
    print(f"  Saved alert_queue: {len(alerts):,} alerts -> {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("06_alert_generation.py - Alert Queue Generation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load data
    print("Loading data...")
    scores = load_scores()
    features = load_user_features()
    user_info = load_user_info()
    print(f"  Scores: {len(scores):,} users")
    print(f"  Features: {len(features):,} users")
    print(f"  User info: {len(user_info):,} users")
    
    # Get threshold from scores
    threshold = 50.0  # Default
    if 'is_anomaly' in scores.columns:
        # Use the implied threshold
        anomaly_scores = scores[scores['is_anomaly'] == 1]['anomaly_score']
        if len(anomaly_scores) > 0:
            threshold = anomaly_scores.min()
    
    print(f"\nAlert threshold: {threshold:.2f}")
    
    # Generate alerts
    print("\nGenerating alerts...")
    alerts = generate_alerts(scores, features, user_info, threshold)
    
    if len(alerts) > 0:
        # Summary by severity
        print(f"\nAlert summary:")
        print(f"  Total alerts: {len(alerts):,}")
        severity_counts = alerts['severity'].value_counts()
        for sev in ['Critical', 'High', 'Medium', 'Low']:
            if sev in severity_counts:
                print(f"    {sev}: {severity_counts[sev]:,}")
        
        # Summary by anomaly type
        print(f"\n  By anomaly type:")
        type_counts = alerts['primary_anomaly_type'].value_counts()
        for atype, count in type_counts.items():
            print(f"    {atype}: {count:,}")
        
        # Show top 5 alerts
        print("\nTop 5 alerts:")
        for _, alert in alerts.head(5).iterrows():
            print(f"  [{alert['severity']}] User {alert['user_id'][:8]}... Score: {alert['anomaly_score']:.1f}")
            print(f"    Type: {alert['primary_anomaly_type']}")
            print(f"    Explanation: {alert['explanation']}")
            print()
        
        # Save alerts
        print("Saving to gold layer...")
        save_alerts(alerts)
    else:
        print("No alerts generated (no anomalies above threshold)")
    
    print("\n" + "=" * 60)
    print("✅ Alert generation complete!")
    print(f"Output: {GOLD_DATA_DIR / 'alert_queue.parquet'}")
    print("=" * 60)
