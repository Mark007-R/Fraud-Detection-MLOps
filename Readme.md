# SENTINEL Fraud Detection

End-to-end fraud detection project for payment transactions with:
- reproducible ML pipelines using DVC
- XGBoost model training and evaluation
- optional Amazon Fraud Dataset Benchmark (FDB) integration
- multi-page Streamlit dashboard for prediction and analytics

## What This Project Does

SENTINEL combines PaySim and Sparkov-style transaction data into a common schema, engineers fraud-focused features, trains an XGBoost classifier, evaluates performance, and benchmarks against published baselines.

The repository includes both:
- command-line pipeline stages (in src/)
- a Streamlit app (app.py + pages/) for interactive use

## Key Features

- DVC pipeline with dependency tracking and reproducible outputs
- Unified schema mapping for PaySim and Sparkov variants
- Feature engineering for amount, time, and balance behavior
- XGBoost training with configurable hyperparameters in params.yaml
- Metrics + report artifacts (scores + SVG confusion matrix and ROC curve)
- Single and batch fraud prediction (CLI + UI)
- Optional FDB baseline comparison output in JSON

## Repository Structure

```text
Sentinel/
|-- app.py
|-- dvc.yaml
|-- params.yaml
|-- requirements.txt
|-- Readme.md
|-- src/
|   |-- config.py
|   |-- load_fdb.py
|   |-- combine_datasets.py
|   |-- preprocess.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- benchmark_fdb.py
|-- pages/
|   |-- 1_Predict.py
|   |-- 2_Performance.py
|   `-- 3_Transactions.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- metrics/
|-- screenshots/
`-- notebooks/
```

## Data Flow

```text
data/raw/paysim.csv + data/raw/sparkov.csv
                |
                v
src/combine_datasets.py
                |
                v
data/processed/combined_transactions.csv
                |
                v
src/preprocess.py
                |
                v
data/processed/features.csv
                |
                v
src/train.py -----------------------------> models/fraud_model.pkl
                |                           data/processed/X_test.csv
                |                           data/processed/y_test.csv
                v
src/evaluate.py --------------------------> metrics/scores.json
                                            reports/confusion_matrix.svg
                                            reports/roc_curve.svg

(optional)
src/benchmark_fdb.py --------------------> metrics/fdb_benchmark.json
```

## Pipeline Stages (DVC)

Defined in dvc.yaml:

1. load_fdb
- Script: src/load_fdb.py
- Purpose: fetch Sparkov train split from FDB (with local fallback)
- Output: data/raw/sparkov.csv

2. combine
- Script: src/combine_datasets.py
- Purpose: normalize PaySim and Sparkov columns into one schema and shuffle
- Output: data/processed/combined_transactions.csv

3. preprocess
- Script: src/preprocess.py
- Purpose: engineer model features and one-hot encode categoricals
- Output: data/processed/features.csv

4. train
- Script: src/train.py
- Purpose: train XGBoost model and persist holdout test split
- Outputs: models/fraud_model.pkl, data/processed/X_test.csv, data/processed/y_test.csv

5. evaluate
- Script: src/evaluate.py
- Purpose: compute metrics and generate report visualizations
- Metrics: metrics/scores.json
- Plots: reports/confusion_matrix.svg, reports/roc_curve.svg

6. benchmark_fdb
- Script: src/benchmark_fdb.py
- Purpose: compare model AUC against fixed baseline references
- Metrics: metrics/fdb_benchmark.json

## Installation

### Prerequisites

- Python 3.9+
- Git
- DVC

### Setup

```bash
git clone <your-repo-url>
cd Sentinel

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Optional (for FDB integration):

```bash
pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git
```

## Running the Project

### Run full DVC pipeline

```bash
dvc repro
```

### Run individual stages

```bash
dvc repro load_fdb
dvc repro combine
dvc repro preprocess
dvc repro train
dvc repro evaluate
dvc repro benchmark_fdb
```

### Run scripts directly

```bash
python src/load_fdb.py
python src/combine_datasets.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python -m src.benchmark_fdb
```

## Inference

### CLI prediction

Single transaction JSON:

```bash
python src/predict.py --input-json "{\"amount\":250000,\"transaction_type\":\"TRANSFER\",\"hour_of_day\":2,\"day_of_month\":14,\"balance_change_orig\":240000,\"balance_ratio\":0.05,\"has_balance_info\":1,\"source\":\"paysim\"}"
```

Batch CSV:

```bash
python src/predict.py --input-csv data/processed/X_test.csv --output-csv predictions.csv
```

## Streamlit App

Launch:

```bash
streamlit run app.py
```

Pages:
- Home (overview, quick KPIs)
- Predict Fraud (single input + batch CSV upload)
- Performance (metrics dashboard and charts)
- Transactions (feature exploration and correlations)

## Configuration

All main config is in params.yaml.

Important sections:
- data: data/model paths
- fdb: FDB dataset key and cache settings
- combine: paths for PaySim/Sparkov merge
- preprocess: input/output for features
- train: model hyperparameters and split settings
- evaluate: metrics/report output paths
- benchmark: benchmark output path

## Outputs

Primary artifacts created by the pipeline:
- data/processed/combined_transactions.csv
- data/processed/features.csv
- data/processed/X_test.csv
- data/processed/y_test.csv
- models/fraud_model.pkl
- metrics/scores.json
- metrics/fdb_benchmark.json
- reports/confusion_matrix.svg
- reports/roc_curve.svg

### Streamlit App Screenshots

**Home Dashboard — KPIs & Model Status**
![Home Dashboard - KPIs](screenshots/home.png)

**Home Dashboard — Capabilities & Quick Start Guide**
![Home Dashboard - Guide](screenshots/home2.png)

**Fraud Prediction**
![Fraud Prediction](screenshots/predict.png)

**Model Performance — Key Metrics**
![Model Performance - Metrics](screenshots/performance.png)

**Model Performance — Charts & Class Distribution**
![Model Performance - Charts](screenshots/performance2.png)

**Transaction Analysis — Amount Distribution**
![Transaction Analysis - Amounts](screenshots/transactions.png)

**Transaction Analysis — Pattern & Anomaly Detection**
![Transaction Analysis - Patterns](screenshots/transactions2.png)

## Current Metrics Snapshot

From metrics/scores.json in this workspace:
- precision: 0.3370
- recall: 0.9622
- f1_score: 0.4992
- accuracy: 0.9960
- auc_roc: 0.9990
- average_precision: 0.8628

Interpretation:
- very high recall and AUC indicate strong ranking/detection capability
- lower precision indicates many false positives at default threshold

## Notes and Limitations

- Lower precision (0.34) at the default 0.5 threshold indicates many false positives; adjusting the classification threshold can improve precision at the cost of recall.
- The FDB benchmark stage requires the optional `fraud-dataset-benchmark` package and will fall back to local Sparkov data if unavailable.

## Useful Commands

```bash
dvc metrics show
dvc params diff
dvc metrics diff
```

## Tech Stack

- Python, pandas, numpy
- scikit-learn, XGBoost, imbalanced-learn
- DVC
- Streamlit, Plotly, Altair
- (optional) Amazon Fraud Dataset Benchmark

## License

This project is licensed under the MIT License.

SPDX identifier: MIT

Copyright (c) 2026 Rodrigues Mark Oliver

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies.

See the full license text in LICENSE.
