"""Tests for FK referential integrity of synthetic data."""
import pytest
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"


@pytest.fixture(scope="module")
def dim_users():
    """Load dim_users dimension table."""
    return pd.read_parquet(DATA_DIR / "dim_users.parquet")


@pytest.fixture(scope="module")
def dim_resources():
    """Load dim_resources dimension table."""
    return pd.read_parquet(DATA_DIR / "dim_resources.parquet")


@pytest.fixture(scope="module")
def fact_permissions():
    """Load fact_permissions fact table."""
    return pd.read_parquet(DATA_DIR / "fact_permissions.parquet")


@pytest.fixture(scope="module")
def fact_access_events():
    """Load fact_access_events fact table."""
    return pd.read_parquet(DATA_DIR / "fact_access_events.parquet")


class TestFKIntegrity:
    """Test FK referential integrity between fact and dimension tables."""

    def test_permissions_user_id_references_dim_users(self, fact_permissions, dim_users):
        """All user_ids in fact_permissions must exist in dim_users."""
        invalid_user_ids = fact_permissions[
            ~fact_permissions['user_id'].isin(dim_users['user_id'])
        ]
        assert len(invalid_user_ids) == 0, (
            f"Found {len(invalid_user_ids)} permissions with invalid user_id"
        )

    def test_permissions_resource_id_references_dim_resources(self, fact_permissions, dim_resources):
        """All resource_ids in fact_permissions must exist in dim_resources."""
        invalid_resource_ids = fact_permissions[
            ~fact_permissions['resource_id'].isin(dim_resources['resource_id'])
        ]
        assert len(invalid_resource_ids) == 0, (
            f"Found {len(invalid_resource_ids)} permissions with invalid resource_id"
        )

    def test_access_events_user_id_references_dim_users(self, fact_access_events, dim_users):
        """All user_ids in fact_access_events must exist in dim_users."""
        invalid_user_ids = fact_access_events[
            ~fact_access_events['user_id'].isin(dim_users['user_id'])
        ]
        assert len(invalid_user_ids) == 0, (
            f"Found {len(invalid_user_ids)} access events with invalid user_id"
        )

    def test_access_events_resource_id_references_dim_resources(self, fact_access_events, dim_resources):
        """All resource_ids in fact_access_events must exist in dim_resources."""
        invalid_resource_ids = fact_access_events[
            ~fact_access_events['resource_id'].isin(dim_resources['resource_id'])
        ]
        assert len(invalid_resource_ids) == 0, (
            f"Found {len(invalid_resource_ids)} access events with invalid resource_id"
        )
