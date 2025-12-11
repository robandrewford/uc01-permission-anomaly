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
    # Normal permissions (log-normal distribution per user)
    permissions_per_user = np.random.lognormal(mean=3.5, sigma=0.8, size=len(users)).astype(int)
    permissions_per_user = np.clip(permissions_per_user, 1, 200)
    
    records = []
    for i, (_, user) in enumerate(users.iterrows()):
        n_perms = min(permissions_per_user[i], n // len(users) * 2)
        
        # Bias toward same-department resources
        dept_resources = resources[resources['owning_department'] == user['department']]
        other_resources = resources[resources['owning_department'] != user['department']]
        
        same_dept_count = int(n_perms * 0.7)
        cross_dept_count = n_perms - same_dept_count
        
        selected_resources = pd.concat([
            dept_resources.sample(min(same_dept_count, len(dept_resources)), replace=True),
            other_resources.sample(min(cross_dept_count, len(other_resources)), replace=True)
        ])
        
        for _, resource in selected_resources.iterrows():
            records.append({
                'permission_id': fake.uuid4(),
                'user_id': user['user_id'],
                'resource_id': resource['resource_id'],
                'permission_level': np.random.choice(['Read', 'Contribute', 'Full Control'], p=[0.6, 0.3, 0.1]),
                'grant_type': np.random.choice(['Direct', 'Inherited', 'Sharing Link', 'Group'], p=[0.3, 0.4, 0.2, 0.1]),
                'granted_date': fake.date_time_between(start_date='-2y', end_date='now'),
                'is_anomaly': False
            })
        
        if len(records) >= n:
            break
    
    return pd.DataFrame(records[:n])

def inject_anomalies(permissions: pd.DataFrame, users: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    """Inject anomaly patterns per UC-01 spec."""
    anomaly_users = users.sample(int(len(users) * ANOMALY_RATE))
    
    anomaly_records = []
    for _, user in anomaly_users.iterrows():
        anomaly_type = np.random.choice([
            'permission_spike',      # 50+ permissions in 7 days
            'cross_department',      # Access to 5+ new departments
            'sensitivity_escalation', # Highly Confidential without history
            'admin_grant'            # Non-IT user gets admin
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
                    'is_anomaly': True,
                    'anomaly_type': 'sensitivity_escalation'
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

print("Injecting anomalies...")
permissions = inject_anomalies(permissions, users, resources)

# Save
users.to_parquet('data/synthetic/dim_users.parquet')
resources.to_parquet('data/synthetic/dim_resources.parquet')
permissions.to_parquet('data/synthetic/fact_permissions.parquet')

print(f"Generated: {len(users)} users, {len(resources)} resources, {len(permissions)} permissions")
print(f"Anomaly rate: {permissions['is_anomaly'].mean():.2%}")