# SENTINEL Fraud Detection Pipeline

An end-to-end fraud detection system for digital payment transactions using **Dask** for scalable data processing and **DVC** for reproducible ML pipelines. Built with XGBoost, featuring automated feature engineering, experiment tracking, and model versioning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Using the Web UI](#using-the-web-ui)
- [Pipeline Stages](#pipeline-stages)
- [Running the Pipeline](#running-the-pipeline)
- [Experiment Tracking](#experiment-tracking)
- [Results](#results)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Digital payment platforms process millions of transactions daily, making manual fraud detection impossible. This project builds a **scalable, reproducible ML pipeline** that:

- Processes large-scale transaction data using **Dask** (parallel & out-of-core computing)
- Engineers fraud-indicative features like transaction velocity, amount deviation, and time-based patterns
- Trains an **XGBoost** classifier to flag fraudulent transactions
- Tracks every data version, parameter change, and model artifact using **DVC**
- Enables easy experiment comparison and full pipeline reproducibility

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Preprocessing  │────▶│    Training     │
│ (transactions)  │     │  (Dask-based)   │     │   (XGBoost)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │   Evaluation    │◀─────────────┘
                        │ (Metrics/Plots) │
                        └─────────────────┘

── Entire pipeline versioned & tracked by DVC ──
```

---

## Tech Stack

| Component              | Tool                        |
| ---------------------- | --------------------------- |
| Data Processing        | Dask                        |
| Model Training         | XGBoost, Scikit-learn       |
| Pipeline Orchestration | DVC                         |
| Data/Model Versioning  | DVC + Git                   |
| Visualization (Static) | Matplotlib, Seaborn         |
| Visualization (Web UI) | Streamlit, Plotly, Altair   |
| Language               | Python 3.9+                 |

---

## Project Structure

```
sentinel-fraud-detection-pipeline/
│
├── data/
│   ├── raw/                        # Original dataset (DVC tracked)
│   │   └── transactions.csv
│   └── processed/                  # Feature-engineered data
│       └── features.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Dask-based cleaning & feature engineering
│   ├── train.py                    # Model training
│   ├── evaluate.py                 # Metrics & confusion matrix
│   └── predict.py                  # Inference on new data
│
├── pages/                          # Streamlit UI pages
│   ├── 1_Predict.py               # Single & batch fraud prediction interface
│   ├── 2_Performance.py            # Model evaluation metrics & visualizations
│   └── 3_Transactions.py           # Transaction analysis & insights
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
│
├── models/
│   └── fraud_model.pkl             # Trained model (DVC tracked)
│
├── metrics/
│   └── scores.json                 # Precision, Recall, F1, AUC
│
├── app.py                          # Streamlit main app entry point
├── ui_utils.py                     # UI utility functions & helpers
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # Pipeline lock file
├── params.yaml                     # Hyperparameters & config
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentinel-fraud-detection-pipeline.git
cd sentinel-fraud-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Download Dataset

```bash
# Option 1 — If using DVC remote storage
dvc pull

# Option 2 — Manual download
# Download the dataset from Kaggle (see Dataset section below)
# Place it in data/raw/transactions.csv
```

---

## Using the Web UI

### 🛡️ SENTINEL Fraud Detection Dashboard

A modern, interactive web interface built with **Streamlit** for fraud prediction, model evaluation, and transaction analysis.

### Features

- **🔍 Fraud Prediction** — Detect fraudulent transactions in real-time
  - Single transaction prediction with instant results
  - Batch upload for multiple transactions
  - Fraud probability scores and confidence levels
  
- **📊 Performance Dashboard** — Comprehensive model metrics
  - Classification metrics (Accuracy, Precision, Recall, F1)
  - Confusion matrix visualization
  - ROC curve and AUC analysis
  - Feature importance rankings
  
- **📈 Transaction Analysis** — Data insights and visualizations
  - Transaction amount distributions
  - Pattern detection and anomalies
  - Time-based trends
  - Feature correlations
  - Data export functionality

### Running the UI

```bash
# Ensure you're in the project directory and virtual environment is activated
# Install Streamlit (should already be in requirements.txt)
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### UI Structure

```
SENTINEL Fraud Detection Dashboard
├── 🏠 Home
│   └── Quick overview & latest metrics
├── 🔍 Predict Fraud
│   ├── Single Transaction Prediction
│   └── Batch Upload (CSV)
├── 📊 Performance
│   ├── Classification Metrics
│   ├── Confusion Matrix
│   ├── ROC Curve
│   ├── Feature Importance
│   └── Detailed Report
└── 📈 Transactions
    ├── Amount Analysis
    ├── Pattern Detection
    ├── Time-based Trends
    ├── Feature Distributions
    └── Correlations
```

### Example Workflow

1. **Train the Model** (if not already trained)
   ```bash
   dvc repro
   ```

2. **Launch the UI**
   ```bash
   streamlit run app.py
   ```

3. **Make Predictions**
   - Go to "Predict Fraud" tab
   - Enter transaction details or upload CSV
   - Get instant fraud probability

4. **Review Performance**
   - Go to "Performance" tab
   - View model metrics and visualizations

5. **Analyze Transactions**
   - Go to "Transactions" tab
   - Explore patterns and anomalies

### Customizing the UI

**Theme & Colors** — Edit the CSS styles in `app.py`:
```python
st.markdown("""
<style>
    .fraud-alert { background-color: #fee2e2; ... }
    .safe-transaction { background-color: #dcfce7; ... }
</style>
""", unsafe_allow_html=True)
```

**Model Path** — Change the default model path in prediction pages:
```python
model_path = "models/fraud_model.pkl"  # Edit this
```

**UI Utils** — Helper functions available in `ui_utils.py`:
```python
from ui_utils import (
    load_model_metrics,
    load_features_data,
    validate_transaction_data,
    calculate_fraud_statistics
)
```

### Performance Tips

- **Batch predictions**: Upload CSV for faster processing of multiple transactions
- **Caching**: The UI caches metrics and features data automatically
- **Large datasets**: Use `nrows` parameter to load subsets for faster exploration

---

## Pipeline Stages

### Stage 1 — Preprocessing (`src/preprocess.py`)

Uses **Dask** to handle large-scale transaction data with parallel processing.

**Operations performed:**
- Load raw CSV using `dask.dataframe` for out-of-core reading
- Handle missing values and data type conversions
- Engineer fraud-indicative features:
  - `tx_frequency` — number of transactions per user in a rolling window
  - `amount_deviation` — how far a transaction amount deviates from user's average
  - `time_since_last_tx` — seconds elapsed since the user's previous transaction
  - `hour_of_day`, `day_of_week` — temporal patterns
  - `is_high_amount` — flag for transactions above a threshold
- Output cleaned feature set to `data/processed/features.csv`

### Stage 2 — Training (`src/train.py`)

- Loads processed features
- Splits data into train/test sets
- Trains an **XGBoost** classifier with parameters from `params.yaml`
- Saves the trained model to `models/fraud_model.pkl`

### Stage 3 — Evaluation (`src/evaluate.py`)

- Loads trained model and test data
- Computes **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**
- Generates confusion matrix plot
- Saves metrics to `metrics/scores.json`

---

## Running the Pipeline

### Run the full pipeline

```bash
dvc repro
```

This executes all three stages in order. DVC automatically skips stages where inputs haven't changed.

### Run individual stages

```bash
dvc repro preprocess
dvc repro train
dvc repro evaluate
```

### View metrics

```bash
dvc metrics show
```

### Compare experiments

```bash
# After changing params.yaml and re-running
dvc params diff
dvc metrics diff
```

---

## Experiment Tracking

DVC makes it easy to iterate and compare experiments:

```bash
# 1. Modify hyperparameters in params.yaml
# 2. Re-run the pipeline
dvc repro

# 3. Compare metrics with previous run
dvc metrics diff

# 4. View parameter changes
dvc params diff

# 5. Commit the experiment
git add .
git commit -m "experiment: increase n_estimators to 300"
```

Example metrics output:

```json
{
  "precision": 0.96,
  "recall": 0.91,
  "f1_score": 0.93,
  "auc_roc": 0.98
}
```

---

## Results

| Metric    | Score |
| --------- | ----- |
| Precision | 0.96  |
| Recall    | 0.91  |
| F1-Score  | 0.93  |
| AUC-ROC   | 0.98  |

> *Results based on the IEEE-CIS Fraud Detection dataset. Actual results may vary based on dataset and hyperparameters.*

---

## Dataset

This project supports any tabular transaction dataset. Recommended public datasets:

- [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284K transactions, 492 frauds)
- [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (590K transactions)

Place the downloaded CSV in `data/raw/transactions.csv` and run:

```bash
dvc add data/raw/transactions.csv
git add data/raw/transactions.csv.dvc .gitignore
git commit -m "track raw dataset with DVC"
```

---

## Why Dask + DVC?

| Challenge                          | Solution                                              |
| ---------------------------------- | ----------------------------------------------------- |
| Dataset too large for pandas       | **Dask** — parallel, chunked, out-of-core processing  |
| Can't reproduce previous results   | **DVC** — tracks data, params, and model versions     |
| No way to compare experiments      | **DVC metrics** — diff params and scores across runs  |
| Large files can't go in Git        | **DVC remote** — stores large files outside Git       |
| Pipeline breaks when stages change | **DVC pipeline** — dependency-aware stage execution    |

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Dask Documentation](https://docs.dask.org/)
- [DVC Documentation](https://dvc.org/doc)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle Fraud Detection Datasets](https://www.kaggle.com/)