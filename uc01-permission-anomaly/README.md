# UC-01 Implementation Guide: Permission Anomaly Detection

## Local Development → Fabric Export

### Phase 1: Project Setup in VSCode

#### Create project structure
```sh
mkdir -p avepoint-ml/uc01-permission-anomaly/{notebooks,src,data,models,tests}
cd avepoint-ml/uc01-permission-anomaly
```

#### Initialize
```sh
git init
uv init  # or: python -m venv .venv && source .venv/bin/activate
```

#### Install dependencies from UC-01 spec
```sh
uv add scikit-learn pyod tensorflow pandas polars mlflow shap matplotlib seaborn
```

#### Project Structure
```sh
uc01-permission-anomaly/
├── notebooks/
│   ├── 00_generate_synthetic_data.py # Synthetic raw data generation
│   ├── 01_ingest_permissions.py      # Bronze layer
│   ├── 02_clean_transform.py         # Silver layer
│   ├── 03_feature_engineering.py     # Feature computation
│   ├── 04_model_training.py          # Train anomaly models
│   ├── 05_batch_scoring.py           # Score users
│   └── 06_alert_generation.py        # Generate alerts
├── src/
│   ├── features.py                   # Feature engineering functions
│   ├── models.py                     # Model classes
│   └── evaluation.py                 # Metrics computation
├── data/
│   ├── bronze/                    # Raw data
│   ├── silver/                    # Cleaned fact and dimension tables
│   └── gold/                      # Sample AvePoint exports
├── models/                           # Trained model artifacts
└── tests/
    └── test_features.py
    └── test_models.py
    └── test_evaluation.py
```
