"""Generate synthetic dataset per UC-01 schema."""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)

# Parameters from spec
N_USERS = 10000
N_RESOURCES = 50000
N_PERMISSIONS = 500000
N_ACCESS_EVENTS = 500000  # Reduced from 5M for local dev; scale up for production
ANOMALY_RATE = 0.01

def generate_users(n: int) -> pd.DataFrame:
    departments = ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Legal', 'IT', 'Operations']
    roles = ['Individual Contributor', 'Manager', 'Director', 'VP', 'Executive']
    
    return pd.DataFrame({
        'user_id': [fake.uuid4() for _ in range(n)],
        'department': np.random.choice(departments, n),
        'role': np.random.choice(roles, n, p=[0.6, 0.25, 0.1, 0.04, 0.01]),
        'hire_date': [fake.date_between(start_date='-5y', end_date='today') for _ in range(n)],
        'location': [fake.city() for _ in range(n)]
    })

def generate_resources(n: int) -> pd.DataFrame:
    sensitivity = ['Public', 'Internal', 'Confidential', 'Highly Confidential']
    resource_types = ['Site', 'Library', 'Folder', 'File', 'Team', 'Group']
    
    return pd.DataFrame({
        'resource_id': [fake.uuid4() for _ in range(n)],
        'resource_type': np.random.choice(resource_types, n),
        'sensitivity_label': np.random.choice(sensitivity, n, p=[0.3, 0.5, 0.15, 0.05]),
        'owning_department': np.random.choice(
            ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Legal', 'IT', 'Operations'], n
        ),
        'created_date': [fake.date_between(start_date='-3y', end_date='today') for _ in range(n)]
    })

def generate_permissions(users: pd.DataFrame, resources: pd.DataFrame, n: int) -> pd.DataFrame:
    """Generate permissions using vectorized operations for speed."""
    # Pre-compute resource lookup by department for same-dept bias
    dept_to_resources = resources.groupby('owning_department')['resource_id'].apply(list).to_dict()
    all_resource_ids = resources['resource_id'].values
    
    # Vectorized approach: generate all n permissions at once
    user_indices = np.random.choice(len(users), n)
    user_data = users.iloc[user_indices]
    
    # For each permission, decide if same-dept (70%) or cross-dept (30%)
    is_same_dept = np.random.random(n) < 0.7
    
    resource_ids = []
    for i, (is_same, dept) in enumerate(zip(is_same_dept, user_data['department'].values)):
        if is_same and dept in dept_to_resources:
            resource_ids.append(np.random.choice(dept_to_resources[dept]))
        else:
            resource_ids.append(np.random.choice(all_resource_ids))
    
    # Generate dates in batch
    now = datetime.now()
    days_ago = np.random.uniform(0, 730, n)  # 0-2 years
    granted_dates = [now - timedelta(days=int(d), hours=np.random.randint(0, 24)) for d in days_ago]
    
    return pd.DataFrame({
        'permission_id': [fake.uuid4() for _ in range(n)],
        'user_id': user_data['user_id'].values,
        'resource_id': resource_ids,
        'permission_level': np.random.choice(['Read', 'Contribute', 'Full Control'], n, p=[0.6, 0.3, 0.1]),
        'grant_type': np.random.choice(['Direct', 'Inherited', 'Sharing Link', 'Group'], n, p=[0.3, 0.4, 0.2, 0.1]),
        'granted_date': granted_dates,
        'is_external': np.random.choice([True, False], n, p=[0.05, 0.95]),
        'is_anomaly': False
    })


def generate_access_events(users: pd.DataFrame, resources: pd.DataFrame, n: int) -> pd.DataFrame:
    """Generate access events per UC-01 spec with off-hours patterns."""
    user_ids = users['user_id'].values
    resource_ids = resources['resource_id'].values
    event_types = ['View', 'Edit', 'Download', 'Share']
    
    # Pre-generate all random data for performance
    records = {
        'event_id': [fake.uuid4() for _ in range(n)],
        'user_id': np.random.choice(user_ids, n),
        'resource_id': np.random.choice(resource_ids, n),
        'event_type': np.random.choice(event_types, n, p=[0.6, 0.25, 0.1, 0.05]),
        'client_ip': [fake.ipv4() for _ in range(n)],
        'user_agent': np.random.choice(['Browser', 'Desktop App', 'Mobile App', 'API'], n, p=[0.5, 0.3, 0.15, 0.05])
    }
    
    # Generate timestamps with 80% business hours (8am-6pm), 20% off-hours
    base_dates = [fake.date_time_between(start_date='-90d', end_date='now') for _ in range(n)]
    timestamps = []
    for dt in base_dates:
        if np.random.random() < 0.8:  # Business hours
            hour = np.random.randint(8, 18)
        else:  # Off-hours
            hour = np.random.choice(list(range(0, 8)) + list(range(18, 24)))
        timestamps.append(dt.replace(hour=hour, minute=np.random.randint(0, 60)))
    
    records['event_timestamp'] = timestamps
    records['is_off_hours'] = [t.hour < 8 or t.hour >= 18 for t in timestamps]
    records['is_anomaly'] = False
    
    return pd.DataFrame(records)

def inject_anomalies(permissions: pd.DataFrame, users: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    """Inject anomaly patterns per UC-01 spec."""
    anomaly_users = users.sample(int(len(users) * ANOMALY_RATE))
    
    anomaly_records = []
    for _, user in anomaly_users.iterrows():
        anomaly_type = np.random.choice([
            'permission_spike',       # 50+ permissions in 7 days
            'cross_department',       # Access to 5+ new departments  
            'sensitivity_escalation', # Highly Confidential without history
            'admin_grant',            # Non-IT user gets admin
            'external_share_spike'    # High external sharing rate
        ])
        
        if anomaly_type == 'permission_spike':
            # Add 50+ permissions in recent 7 days
            for _ in range(np.random.randint(50, 100)):
                anomaly_records.append({
                    'permission_id': fake.uuid4(),
                    'user_id': user['user_id'],
                    'resource_id': resources.sample(1)['resource_id'].values[0],
                    'permission_level': np.random.choice(['Read', 'Contribute', 'Full Control']),
                    'grant_type': 'Direct',
                    'granted_date': fake.date_time_between(start_date='-7d', end_date='now'),
                    'is_external': np.random.choice([True, False], p=[0.1, 0.9]),
                    'is_anomaly': True,
                    'anomaly_type': 'permission_spike'
                })
        
        elif anomaly_type == 'sensitivity_escalation':
            # Grant Highly Confidential access
            hc_resources = resources[resources['sensitivity_label'] == 'Highly Confidential']
            for _ in range(np.random.randint(5, 15)):
                anomaly_records.append({
                    'permission_id': fake.uuid4(),
                    'user_id': user['user_id'],
                    'resource_id': hc_resources.sample(1)['resource_id'].values[0],
                    'permission_level': 'Full Control',
                    'grant_type': 'Direct',
                    'granted_date': fake.date_time_between(start_date='-3d', end_date='now'),
                    'is_external': False,
                    'is_anomaly': True,
                    'anomaly_type': 'sensitivity_escalation'
                })
        
        elif anomaly_type == 'external_share_spike':
            # High external sharing - 40%+ external recipients
            for _ in range(np.random.randint(20, 50)):
                anomaly_records.append({
                    'permission_id': fake.uuid4(),
                    'user_id': user['user_id'],
                    'resource_id': resources.sample(1)['resource_id'].values[0],
                    'permission_level': np.random.choice(['Read', 'Contribute']),
                    'grant_type': 'Sharing Link',
                    'granted_date': fake.date_time_between(start_date='-7d', end_date='now'),
                    'is_external': True,  # Anomalously high external sharing
                    'is_anomaly': True,
                    'anomaly_type': 'external_share_spike'
                })
    
    anomaly_df = pd.DataFrame(anomaly_records)
    return pd.concat([permissions, anomaly_df], ignore_index=True)

# Generate datasets
print("Generating users...")
users = generate_users(N_USERS)

print("Generating resources...")
resources = generate_resources(N_RESOURCES)

print("Generating permissions...")
permissions = generate_permissions(users, resources, N_PERMISSIONS)

print("Generating access events...")
access_events = generate_access_events(users, resources, N_ACCESS_EVENTS)

print("Injecting anomalies...")
permissions = inject_anomalies(permissions, users, resources)

# Save
users.to_parquet('data/synthetic/dim_users.parquet')
resources.to_parquet('data/synthetic/dim_resources.parquet')
permissions.to_parquet('data/synthetic/fact_permissions.parquet')
access_events.to_parquet('data/synthetic/fact_access_events.parquet')

print(f"Generated: {len(users)} users, {len(resources)} resources, {len(permissions)} permissions, {len(access_events)} access events")
print(f"Permission anomaly rate: {permissions['is_anomaly'].mean():.2%}")
print(f"Off-hours access rate: {access_events['is_off_hours'].mean():.2%}")