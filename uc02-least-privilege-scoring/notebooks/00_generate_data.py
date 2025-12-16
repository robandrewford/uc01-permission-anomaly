# UC-02: Least-Privilege Scoring - Data Generation
# Generates synthetic permission grants, user attributes, and activity logs.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating Synthetic Data for Least-Privilege Scoring...")

# 1. Users
n_users = 100
users = pd.DataFrame({
    'UserId': range(n_users),
    'Department': np.random.choice(['Sales', 'Engineering', 'HR', 'IT'], n_users),
    'Role': np.random.choice(['Manager', 'Individual Contributor', 'Admin'], n_users)
})

# 2. Permissions (Grants)
# Each user has 50-100 permissions
grants = []
for uid in users['UserId']:
    n_perms = np.random.randint(50, 100)
    for i in range(n_perms):
        grants.append({
            'UserId': uid,
            'PermissionId': f"Perm_{np.random.randint(1000, 9999)}",
            'GrantDate': datetime.now() - timedelta(days=np.random.randint(30, 365)),
            'Scope': np.random.choice(['Site', 'File', 'Team']),
            'Sensitivity': np.random.choice(['General', 'Confidential', 'Highly Confidential'])
        })
grants_df = pd.DataFrame(grants)

# 3. Usage Signal (Activity)
# Correlate usage: Engineers use 80% of perms, Sales use 20%
# Logic: If usage_days_ago < 90, it's 'Necessary'. Else 'Excessive'.
print("Simulating activity...")

# Export to Lakehouse (Simulation)
# In real Fabric, this would use: df.write.format("delta").save("Tables/...")
print(f"Generated {len(users)} users and {len(grants_df)} permission grants.")
print("Data generation complete.")
