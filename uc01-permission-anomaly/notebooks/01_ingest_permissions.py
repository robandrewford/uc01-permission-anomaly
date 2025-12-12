"""
01_ingest_permissions.py - Bronze Layer Ingestion

This notebook pulls data from source (synthetic or AvePoint API) to bronze layer.
Per UC-01 spec: fabric_workload_structure.notebook_workflow[0]

Bronze layer tables:
- raw_permissions: Permission snapshots from AvePoint API
- raw_access_events: Access audit logs
- raw_users: User directory snapshot
- raw_resources: Resource inventory
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# For local development, use synthetic data
# In Fabric, this would connect to AvePoint APIs
SOURCE_TYPE = "synthetic"  # Options: "synthetic", "avepoint_api"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
BRONZE_DATA_DIR = PROJECT_ROOT / "data" / "bronze"


# ============================================================================
# Ingestion Functions
# ============================================================================

def ingest_users() -> pd.DataFrame:
    """Ingest user dimension data to bronze layer."""
    if SOURCE_TYPE == "synthetic":
        df = pd.read_parquet(SYNTHETIC_DATA_DIR / "dim_users.parquet")
    else:
        # TODO: Implement AvePoint API connection
        # df = avepoint_client.get_users()
        raise NotImplementedError("AvePoint API ingestion not implemented")
    
    # Add ingestion metadata
    df['_ingested_at'] = datetime.now()
    df['_source'] = SOURCE_TYPE
    return df


def ingest_resources() -> pd.DataFrame:
    """Ingest resource dimension data to bronze layer."""
    if SOURCE_TYPE == "synthetic":
        df = pd.read_parquet(SYNTHETIC_DATA_DIR / "dim_resources.parquet")
    else:
        raise NotImplementedError("AvePoint API ingestion not implemented")
    
    df['_ingested_at'] = datetime.now()
    df['_source'] = SOURCE_TYPE
    return df


def ingest_permissions() -> pd.DataFrame:
    """Ingest permission fact data to bronze layer."""
    if SOURCE_TYPE == "synthetic":
        df = pd.read_parquet(SYNTHETIC_DATA_DIR / "fact_permissions.parquet")
    else:
        # Production: AvePoint Permission Inventory API
        raise NotImplementedError("AvePoint API ingestion not implemented")
    
    df['_ingested_at'] = datetime.now()
    df['_source'] = SOURCE_TYPE
    return df


def ingest_access_events() -> pd.DataFrame:
    """Ingest access event fact data to bronze layer."""
    if SOURCE_TYPE == "synthetic":
        df = pd.read_parquet(SYNTHETIC_DATA_DIR / "fact_access_events.parquet")
    else:
        # Production: AvePoint Access Audit API
        raise NotImplementedError("AvePoint API ingestion not implemented")
    
    df['_ingested_at'] = datetime.now()
    df['_source'] = SOURCE_TYPE
    return df


def save_bronze_layer(data: dict[str, pd.DataFrame]) -> None:
    """Save all tables to bronze layer."""
    BRONZE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in data.items():
        output_path = BRONZE_DATA_DIR / f"{table_name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved {table_name}: {len(df):,} rows -> {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("01_ingest_permissions.py - Bronze Layer Ingestion")
    print("=" * 60)
    print(f"Source: {SOURCE_TYPE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    print("Ingesting data...")
    bronze_data = {
        "raw_users": ingest_users(),
        "raw_resources": ingest_resources(),
        "raw_permissions": ingest_permissions(),
        "raw_access_events": ingest_access_events(),
    }
    
    print("\nSaving to bronze layer...")
    save_bronze_layer(bronze_data)
    
    print("\n✅ Bronze layer ingestion complete!")
    print(f"Output directory: {BRONZE_DATA_DIR}")
