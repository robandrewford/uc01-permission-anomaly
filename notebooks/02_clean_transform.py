"""
02_clean_transform.py - Bronze to Silver Layer Transformation

This notebook applies data quality transformations to bronze layer data.
Per UC-01 spec: fabric_workload_structure.notebook_workflow[1]

Silver layer tables:
- cleaned_permissions: Deduplicated, validated permissions
- cleaned_access_events: Validated access events
- permission_velocity: Time-series of permission changes per user
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BRONZE_DATA_DIR = PROJECT_ROOT / "data" / "bronze"
SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"


# ============================================================================
# Data Loading
# ============================================================================

def load_bronze_layer() -> dict[str, pd.DataFrame]:
    """Load all bronze layer tables."""
    return {
        "raw_users": pd.read_parquet(BRONZE_DATA_DIR / "raw_users.parquet"),
        "raw_resources": pd.read_parquet(BRONZE_DATA_DIR / "raw_resources.parquet"),
        "raw_permissions": pd.read_parquet(BRONZE_DATA_DIR / "raw_permissions.parquet"),
        "raw_access_events": pd.read_parquet(BRONZE_DATA_DIR / "raw_access_events.parquet"),
    }


# ============================================================================
# Transformation Functions
# ============================================================================

def deduplicate_permissions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate permission records.
    
    Duplicates can occur from overlapping API snapshots or inherited permissions.
    Keep the most recent record for each (user_id, resource_id, permission_level) combo.
    """
    df = df.sort_values('granted_date', ascending=False)
    df = df.drop_duplicates(
        subset=['user_id', 'resource_id', 'permission_level'],
        keep='first'
    )
    return df


def validate_foreign_keys(
    perms: pd.DataFrame,
    users: pd.DataFrame,
    resources: pd.DataFrame
) -> pd.DataFrame:
    """Filter out permissions with invalid foreign keys."""
    valid_user_ids = set(users['user_id'])
    valid_resource_ids = set(resources['resource_id'])
    
    initial_count = len(perms)
    perms = perms[perms['user_id'].isin(valid_user_ids)]
    perms = perms[perms['resource_id'].isin(valid_resource_ids)]
    
    removed = initial_count - len(perms)
    if removed > 0:
        print(f"  Removed {removed} permissions with invalid FKs")
    
    return perms


def standardize_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Ensure date columns are proper datetime types."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def compute_permission_velocity(perms: pd.DataFrame) -> pd.DataFrame:
    """Compute permission change velocity per user over time windows.
    
    Creates a time-series of permission grants per user at different windows:
    - 7 day rolling count
    - 30 day rolling count
    - 90 day rolling count
    """
    # Get the reference date (latest date in data)
    perms = perms.copy()
    perms['granted_date'] = pd.to_datetime(perms['granted_date'])
    reference_date = perms['granted_date'].max()
    
    velocity_records = []
    
    for user_id, user_perms in perms.groupby('user_id'):
        granted_dates = user_perms['granted_date']
        
        # Count permissions in each window
        last_7d = (granted_dates >= reference_date - timedelta(days=7)).sum()
        last_30d = (granted_dates >= reference_date - timedelta(days=30)).sum()
        last_90d = (granted_dates >= reference_date - timedelta(days=90)).sum()
        
        velocity_records.append({
            'user_id': user_id,
            'reference_date': reference_date,
            'permissions_7d': last_7d,
            'permissions_30d': last_30d,
            'permissions_90d': last_90d,
            'velocity_7d': last_7d / 7,
            'velocity_30d': last_30d / 30,
            'velocity_90d': last_90d / 90,
        })
    
    return pd.DataFrame(velocity_records)


def clean_access_events(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate access events."""
    # Validate user_id FK
    valid_user_ids = set(users['user_id'])
    events = events[events['user_id'].isin(valid_user_ids)]
    
    # Standardize timestamps
    events = events.copy()
    events['event_timestamp'] = pd.to_datetime(events['event_timestamp'])
    
    return events


def save_silver_layer(data: dict[str, pd.DataFrame]) -> None:
    """Save all tables to silver layer."""
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in data.items():
        output_path = SILVER_DATA_DIR / f"{table_name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved {table_name}: {len(df):,} rows -> {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("02_clean_transform.py - Silver Layer Transformation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    print("Loading bronze layer...")
    bronze = load_bronze_layer()
    for name, df in bronze.items():
        print(f"  {name}: {len(df):,} rows")
    
    print("\nApplying transformations...")
    
    # Clean permissions
    print("\n  Deduplicating permissions...")
    cleaned_perms = deduplicate_permissions(bronze['raw_permissions'])
    print(f"    Before: {len(bronze['raw_permissions']):,} -> After: {len(cleaned_perms):,}")
    
    print("  Validating foreign keys...")
    cleaned_perms = validate_foreign_keys(
        cleaned_perms,
        bronze['raw_users'],
        bronze['raw_resources']
    )
    
    print("  Standardizing dates...")
    cleaned_perms = standardize_dates(cleaned_perms, ['granted_date'])
    
    # Compute velocity
    print("  Computing permission velocity...")
    velocity = compute_permission_velocity(cleaned_perms)
    
    # Clean access events
    print("  Cleaning access events...")
    cleaned_events = clean_access_events(
        bronze['raw_access_events'],
        bronze['raw_users']
    )
    
    # Prepare silver layer
    silver_data = {
        "cleaned_permissions": cleaned_perms,
        "cleaned_access_events": cleaned_events,
        "permission_velocity": velocity,
        # Pass through dimension tables (already clean)
        "dim_users": bronze['raw_users'],
        "dim_resources": bronze['raw_resources'],
    }
    
    print("\nSaving to silver layer...")
    save_silver_layer(silver_data)
    
    print("\n✅ Silver layer transformation complete!")
    print(f"Output directory: {SILVER_DATA_DIR}")
