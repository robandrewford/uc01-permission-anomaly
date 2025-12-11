"""Feature engineering for UC-01 per specification."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def compute_permission_features(
    permissions: pd.DataFrame,
    users: pd.DataFrame,
    resources: pd.DataFrame,
    reference_date: datetime = None
) -> pd.DataFrame:
    """Compute user-level features from UC-01 spec."""
    
    if reference_date is None:
        reference_date = datetime.now()
    
    # Merge for enrichment
    perm_enriched = permissions.merge(users, on='user_id').merge(
        resources, on='resource_id', suffixes=('_user', '_resource')
    )
    
    features = []
    
    for user_id, group in perm_enriched.groupby('user_id'):
        user_dept = group['department'].iloc[0]
        
        # Time-based filters
        granted_dates = pd.to_datetime(group['granted_date'])
        last_7d = granted_dates >= (reference_date - timedelta(days=7))
        last_30d = granted_dates >= (reference_date - timedelta(days=30))
        
        # Permission velocity (from spec)
        permission_velocity_7d = last_7d.sum() / 7
        permission_velocity_30d = last_30d.sum() / 30
        
        # Cross-department ratio (from spec)
        cross_dept = group['owning_department'] != user_dept
        cross_department_ratio = cross_dept.mean() if len(group) > 0 else 0
        
        # Sensitivity-weighted score (from spec)
        sensitivity_weights = {
            'Public': 1, 'Internal': 2, 'Confidential': 5, 'Highly Confidential': 10
        }
        group['sensitivity_weight'] = group['sensitivity_label'].map(sensitivity_weights)
        sensitivity_weighted_score = group['sensitivity_weight'].sum()
        
        # Permission breadth (from spec)
        permission_breadth = group['resource_id'].nunique()
        
        # Admin permission flag (from spec)
        admin_permission_flag = int((group['permission_level'] == 'Full Control').any())
        
        # External share ratio (simplified - would need sharing data)
        external_share_ratio = 0  # Placeholder
        
        features.append({
            'user_id': user_id,
            'department': user_dept,
            'total_permissions': len(group),
            'permission_velocity_7d': permission_velocity_7d,
            'permission_velocity_30d': permission_velocity_30d,
            'cross_department_ratio': cross_department_ratio,
            'sensitivity_weighted_score': sensitivity_weighted_score,
            'permission_breadth': permission_breadth,
            'admin_permission_flag': admin_permission_flag,
            'external_share_ratio': external_share_ratio,
            'pct_full_control': (group['permission_level'] == 'Full Control').mean(),
            'pct_direct_grants': (group['grant_type'] == 'Direct').mean(),
            'highly_confidential_count': (group['sensitivity_label'] == 'Highly Confidential').sum(),
            'recent_grant_ratio': last_7d.mean()
        })
    
    return pd.DataFrame(features)

def compute_peer_baselines(features: pd.DataFrame) -> pd.DataFrame:
    """Add peer comparison features."""
    
    # Department baselines
    dept_stats = features.groupby('department').agg({
        'total_permissions': ['mean', 'std'],
        'cross_department_ratio': 'mean',
        'sensitivity_weighted_score': ['mean', 'std']
    }).reset_index()
    dept_stats.columns = ['department', 'dept_perm_mean', 'dept_perm_std', 
                          'dept_cross_dept_mean', 'dept_sens_mean', 'dept_sens_std']
    
    features = features.merge(dept_stats, on='department')
    
    # Z-scores relative to department
    features['perm_zscore'] = (
        (features['total_permissions'] - features['dept_perm_mean']) / 
        features['dept_perm_std'].replace(0, 1)
    )
    features['sens_zscore'] = (
        (features['sensitivity_weighted_score'] - features['dept_sens_mean']) / 
        features['dept_sens_std'].replace(0, 1)
    )
    
    return features