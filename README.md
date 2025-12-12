# AvePoint AI/ML Usecase Tools

This repository contains the Machine Learning solutions and tools for AvePoint's security and compliance use cases.

## Solutions

### [UC-01: Permission Anomaly Detection](./uc01-permission-anomaly)

An end-to-end Machine Learning pipeline designed to detect anomalous user permissions and access patterns in SharePoint-like environments.

**Key Features:**
- **Synthetic Data Generation**: Creates realistic user, resource, and permission graphs with injected anomalies.
- **Medallion Architecture**: Structured ETL pipeline (Bronze ingestion → Silver cleaning → Gold features).
- **Unsupervised Learning**: Utilizes an ensemble of Isolation Forest and Autoencoders to score anomaly likelihood.
- **Model Versioning**: robust artifact management and experiment tracking using MLflow.
- **DuckDB Integration**: SQL interface for exploring data and results.

## Architecture

The project follows a modular design:

```
avepoint-ml/
└── uc01-permission-anomaly/
    ├── notebooks/            # ETL, Training, Scoring, Alerting
    ├── src/                  # Shared libraries (features, models, evaluation)
    ├── data/                 # Local data storage (parquet)
    ├── models/               # Versioned model artifacts
    └── scripts/              # Utility scripts (e.g., DuckDB shell)
```

### Data Pipeline
1. **Bronze**: Raw ingestion of Users, Resources, Permissions, and Access Logs.
2. **Silver**: Cleaned data, foreign key validation, and relational structuring.
3. **Gold**: Feature vectors (e.g., permission counts, velocity, velocity) and anomaly scores.

## Getting Started

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Installation

```bash
cd avepoint-ml/uc01-permission-anomaly
uv sync
```

## Usage Guide

### 1. Run the Full Pipeline

Execute the notebooks in order to generate data, train the model, and score users.

```bash
# 1. Generate Synthetic Data
uv run python notebooks/00_generate_synthetic_data.py

# 2. Ingest & Transform (Bronze -> Silver)
uv run python notebooks/01_ingest_permissions.py
uv run python notebooks/02_clean_transform.py

# 3. Feature Engineering (Silver -> Gold)
uv run python notebooks/03_feature_engineering.py

# 4. Train Model
uv run python notebooks/04_model_training.py

# 5. Batch Scoring
uv run python notebooks/05_batch_scoring.py
```

### 2. Model Management

Models are versioned in `models/<type>_<timestamp>_run_<id>`.
A `models/latest` symlink always points to the most recently trained model.

**Load a specific version for scoring:**
```bash
uv run python notebooks/05_batch_scoring.py --model-path models/ensemble_20251211_154317_run_bd7db5b1
```

### 3. Data Exploration (DuckDB)

Use the interactive DuckDB shell to query your local parquet data using SQL.

```bash
uv run python scripts/duckdb_shell.py
```

**Example Query:**
```sql
SELECT user_id, anomaly_score, is_anomaly 
FROM anomaly_scores 
ORDER BY anomaly_score DESC 
LIMIT 5;
```

## License
Proprietary & Confidential.
