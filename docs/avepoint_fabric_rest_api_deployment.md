# AvePoint Fabric REST API Deployment

## Step 1: Get Fabric REST API Token

```python
# Using Azure Identity
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
token = credential.get_token("https://api.fabric.microsoft.com/.default")
```

## Step 2: Create Lakehouse via API

```python
import requests

WORKSPACE_ID = "your-workspace-id"
BASE_URL = f"https://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}"

headers = {
    "Authorization": f"Bearer {token.token}",
    "Content-Type": "application/json"
}

# Create Lakehouse
lakehouse_payload = {
    "displayName": "UC01_PermissionAnomaly_Lakehouse",
    "type": "Lakehouse"
}

response = requests.post(f"{BASE_URL}/items", json=lakehouse_payload, headers=headers)
lakehouse_id = response.json()["id"]
print(f"Created Lakehouse: {lakehouse_id}")
```

## Step 3: Create Notebook via API

```python
import requests

WORKSPACE_ID = "your-workspace-id"
BASE_URL = f"https://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}"

headers = {
    "Authorization": f"Bearer {token.token}",
    "Content-Type": "application/json"
}

# Create Notebook
```python
import base64

# Read notebook content
with open("notebooks/04_model_training.py", "r") as f:
    notebook_content = f.read()

# Convert to Fabric notebook format
notebook_payload = {
    "displayName": "UC01_Model_Training",
    "type": "Notebook",
    "definition": {
        "format": "ipynb",
        "parts": [
            {
                "path": "notebook-content.py",
                "payload": base64.b64encode(notebook_content.encode()).decode(),
                "payloadType": "InlineBase64"
            }
        ]
    }
}

response = requests.post(f"{BASE_URL}/items", json=notebook_payload, headers=headers)
notebook_id = response.json()["id"]
print(f"Created Notebook: {notebook_id}")
```

## Step 4: Create Data Pipeline

```python

pipeline_definition = {
    "name": "UC01_Daily_Scoring",
    "activities": [
        {
            "name": "Run Feature Engineering",
            "type": "TridentNotebook",
            "typeProperties": {
                "notebookId": "uc01-feature-engineering-notebook-id"
            }
        },
        {
            "name": "Run Scoring",
            "type": "TridentNotebook",
            "dependsOn": [{"activity": "Run Feature Engineering", "dependencyConditions": ["Succeeded"]}],
            "typeProperties": {
                "notebookId": "uc01-scoring-notebook-id"
            }
        }
    ]
}

pipeline_payload = {
    "displayName": "UC01_Daily_Pipeline",
    "type": "DataPipeline",
    "definition": {
        "parts": [
            {
                "path": "pipeline-content.json",
                "payload": base64.b64encode(json.dumps(pipeline_definition).encode()).decode(),
                "payloadType": "InlineBase64"
            }
        ]
    }
}

response = requests.post(f"{BASE_URL}/items", json=pipeline_payload, headers=headers)
```

---

## Step 5: Complete UC-01 Fabric Workload Structure
```md
Fabric Workspace: AvePoint_ML_Production
│
├── Lakehouses
│   └── UC01_PermissionAnomaly_Lakehouse
│       ├── Tables/
│       │   ├── bronze_permissions      (raw AvePoint API data)
│       │   ├── bronze_users            (raw user directory)
│       │   ├── bronze_access_events    (raw audit logs)
│       │   ├── silver_cleaned_permissions
│       │   ├── silver_user_features
│       │   ├── gold_anomaly_scores
│       │   ├── gold_alert_queue
│       │   └── gold_score_history
│       └── Files/
│           └── models/
│               ├── isolation_forest_v1.pkl
│               └── scaler_v1.pkl
│
├── Notebooks
│   ├── UC00_00_Generate_Synthetic_Data
│   ├── UC01_01_Ingest_Permissions
│   ├── UC01_02_Clean_Transform
│   ├── UC01_03_Feature_Engineering
│   ├── UC01_04_Model_Training
│   ├── UC01_05_Batch_Scoring
│   └── UC01_06_Alert_Generation
│
├── Data Pipelines
│   ├── UC01_Daily_Ingestion (02:00 UTC)
│   ├── UC01_Daily_Scoring (05:00 UTC)
│   └── UC01_Weekly_Retrain (Sunday 06:00 UTC)
│
└── Power BI Reports
    └── UC01_Anomaly_Dashboard
```

## Quickstart Commands

```bash
# 1. Initialize project
mkdir uc01-permission-anomaly && cd uc01-permission-anomaly
git init && uv init

# 2. Tell Claude:
# "Read the UC-01 spec from avepoint_implementation_specs_complete.json 
#  and generate the synthetic data script"

# 3. Generate synthetic data
uv run python notebooks/00_generate_synthetic_data.py

# 4. Tell Claude:
# "Create the feature engineering module based on UC-01 feature_engineering spec"

# 5. Train models
uv run python notebooks/04_model_training.py

# 6. Export to Fabric
uv run python scripts/export_to_fabric.py
        
# 7. Push to Git-connected Fabric workspace
git add . && git commit -m "UC01 implementation" && git push
```

### TODO: Add Power BI Report
