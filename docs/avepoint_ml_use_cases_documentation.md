# AvePoint ML/GenAI Use Cases for Microsoft Fabric Workloads

## Executive Summary

This document defines 20 MECE (Mutually Exclusive, Collectively Exhaustive) use cases for AvePoint to offer ML and GenAI capabilities to its customers. Each use case is organized using a unified three-framework structure:

- **Framework C (Data Domain)**: The AvePoint telemetry source providing input data
- **Framework A (ML Capability)**: The technical model type/technique implemented  
- **Framework B (Business Outcome)**: The commercial value delivered (1:many mapping)

---

## Framework Mapping Summary

### Coverage by Data Domain (Framework C)

| Data Domain | Use Case Count | Use Cases |
|-------------|----------------|-----------|
| Access & Permissions | 4 | UC-01 through UC-04 |
| Content & Documents | 4 | UC-05 through UC-08 |
| User Behavior & Activity | 4 | UC-09 through UC-12 |
| Policies & Configuration | 4 | UC-13 through UC-16 |
| Lifecycle & Storage | 4 | UC-17 through UC-20 |

### Coverage by ML Capability (Framework A)

| ML Capability | Count | Use Cases |
|---------------|-------|-----------|
| Anomaly Detection | 5 | UC-01, UC-04, UC-10, UC-11, UC-14 |
| Classification | 8 | UC-02, UC-03, UC-05, UC-07, UC-13, UC-16, UC-19, UC-20 |
| Clustering | 4 | UC-06, UC-09, UC-12, UC-20 |
| Time-Series Prediction | 3 | UC-04, UC-18 |
| NLP / GenAI | 4 | UC-05, UC-08, UC-15, UC-16 |
| Recommendation / Optimization | 2 | UC-15, UC-17 |

### Coverage by Business Outcome (Framework B)

| Primary Outcome | Count | Use Cases |
|-----------------|-------|-----------|
| Reduce Security Risk | 6 | UC-01, UC-02, UC-03, UC-10, UC-11, UC-14 |
| Ensure Compliance | 5 | UC-04, UC-05, UC-13, UC-15, UC-16 |
| Optimize Cost | 5 | UC-06, UC-07, UC-17, UC-18, UC-19 |
| Improve Productivity | 3 | UC-08, UC-12, UC-20 |
| Protect Revenue | 1 | UC-09 |

---

## Use Case Index

| ID | Name | Domain | ML Capability | Primary Outcome |
|----|------|--------|---------------|-----------------|
| UC-01 | Permission Anomaly Detection | Access & Permissions | Anomaly Detection | Reduce Security Risk |
| UC-02 | Least-Privilege Scoring | Access & Permissions | Classification + Risk Scoring | Reduce Security Risk |
| UC-03 | External Sharing Risk Model | Access & Permissions | Classification | Reduce Security Risk |
| UC-04 | Permission Drift Forecasting | Access & Permissions | Time-Series Prediction | Ensure Compliance |
| UC-05 | Sensitive Data Classification | Content & Documents | NLP Classification | Ensure Compliance |
| UC-06 | Duplicate/Near-Duplicate Detection | Content & Documents | Clustering + Similarity | Optimize Cost |
| UC-07 | ROT Content Scoring | Content & Documents | Classification + NLP | Optimize Cost |
| UC-08 | Document Summarization & Auto-Tagging | Content & Documents | NLP / GenAI | Improve Productivity |
| UC-09 | Adoption Health Scoring | User Behavior & Activity | Regression + Clustering | Protect Revenue |
| UC-10 | Shadow IT / Unapproved App Detection | User Behavior & Activity | Anomaly Detection | Reduce Security Risk |
| UC-11 | Data Exfiltration Risk Model | User Behavior & Activity | Anomaly Detection + Classification | Reduce Security Risk |
| UC-12 | User Segmentation for Enablement | User Behavior & Activity | Clustering | Improve Productivity |
| UC-13 | Policy Compliance Prediction | Policies & Configuration | Classification + Forecasting | Ensure Compliance |
| UC-14 | Configuration Drift Detection | Policies & Configuration | Anomaly Detection | Reduce Security Risk |
| UC-15 | Policy Recommendation Engine | Policies & Configuration | Recommendation / GenAI | Ensure Compliance |
| UC-16 | Regulatory Readiness Scoring | Policies & Configuration | Classification + NLP | Ensure Compliance |
| UC-17 | Retention Rule Optimization | Lifecycle & Storage | Optimization + Prediction | Optimize Cost |
| UC-18 | Storage Growth Forecasting | Lifecycle & Storage | Time-Series Prediction | Optimize Cost |
| UC-19 | Archive Candidate Scoring | Lifecycle & Storage | Classification | Optimize Cost |
| UC-20 | Migration Readiness Assessment | Lifecycle & Storage | Clustering + Scoring | Improve Productivity |

---

## Metaprompt Usage Guide

Each use case includes a discrete metaprompt designed to generate a detailed implementation specification. The metaprompt output includes:

1. **Model Architecture**: Algorithm selection, ensemble strategies, hyperparameters
2. **Feature Engineering**: Raw and derived features from AvePoint telemetry
3. **Synthetic Dataset Schema**: Table structures for training/evaluation data generation
4. **Experiment Design**: MLflow experiment configuration
5. **Evaluation Metrics**: Success criteria with target thresholds
6. **Fabric Workload Structure**: Lakehouse tables, pipelines, scheduling
7. **Python Packages**: Required dependencies with versions
8. **Commercial Outcome Specification**: Business deliverables and ROI metrics

### How to Use

```python
import json

# Load the use case definitions
with open('avepoint_ml_use_cases.json', 'r') as f:
    data = json.load(f)

# Access a specific use case
use_case = data['use_cases'][0]  # UC-01

# Get the metaprompt for implementation
metaprompt = use_case['metaprompt']

# Send to LLM for detailed specification generation
# response = llm.generate(metaprompt)
```

### Metaprompt Evolution

Each metaprompt is stored as a discrete string to enable independent evolution:
- Version control individual prompts
- A/B test prompt variations
- Customize for specific customer contexts
- Extend with additional output sections as needed

---

## Fabric Workload Mapping

Each use case maps to a Microsoft Fabric Workload with standard structure:

```
/workload_{UC_ID}/
├── bronze/
│   └── raw ingestion tables
├── silver/
│   └── cleaned, joined, feature tables
├── gold/
│   └── aggregated, scored, output tables
├── notebooks/
│   ├── 01_data_ingestion.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_model_evaluation.py
│   └── 05_batch_scoring.py
├── experiments/
│   └── mlflow experiment configs
└── outputs/
    └── dashboards, alerts, integrations
```

---

## Next Steps

1. **Step 3**: Execute each metaprompt to generate detailed implementation specifications
2. **Synthetic Data Generation**: Create datasets for each use case
3. **Pilot Selection**: Choose 3-5 use cases for initial implementation
4. **Fabric Workspace Setup**: Configure development environments
5. **Model Development**: Iterative build-evaluate-refine cycle

---

## File Manifest

| File | Description |
|------|-------------|
| `avepoint_ml_use_cases.json` | Structured data with all 20 use cases, discussions, TL;DR, and metaprompts |
| `avepoint_ml_use_cases_documentation.md` | This documentation file |

