#!/usr/bin/env bash
# chmod +x init_uc.sh 
# Usage: ./init_uc.sh <base_project_dir>
# Example: ./init_uc.sh uc01-permission-anomaly

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <base_project_dir>"
  echo "Example: $0 uc01-permission-anomaly"
  exit 1
fi

base="$1"

mkdir -p "$base"/notebooks
mkdir -p "$base"/src
mkdir -p "$base"/data/synthetic
mkdir -p "$base"/data/raw
mkdir -p "$base"/models
mkdir -p "$base"/tests
mkdir -p "$base"/scripts

touch "$base"/notebooks/00_generate_synthetic_data.py
touch "$base"/notebooks/01_ingest_permissions.py
touch "$base"/notebooks/02_clean_transform.py
touch "$base"/notebooks/03_feature_engineering.py
touch "$base"/notebooks/04_model_training.py
touch "$base"/notebooks/05_batch_scoring.py
touch "$base"/notebooks/06_alert_generation.py

touch "$base"/src/features.py
touch "$base"/src/models.py
touch "$base"/src/evaluation.py

touch "$base"/tests/test_features.py
touch "$base"/tests/test_models.py
touch "$base"/tests/test_evaluation.py