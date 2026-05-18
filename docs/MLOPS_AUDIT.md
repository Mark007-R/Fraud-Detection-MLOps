# Sentinel — MLOps Audit (Day 1, 2026-05-18)

This document inventories Sentinel's pre-sprint state and calls out the two
leakage bugs that inflated the 0.921 sparkov AUC claim. Both are fixed today.

## 1. Existing 6-stage DVC pipeline (dvc.yaml)

| Stage          | Script                          | Inputs                                                     | Outputs                                                                |
|----------------|---------------------------------|------------------------------------------------------------|------------------------------------------------------------------------|
| load_fdb       | `src/load_fdb.py`               | (FDB API or local sparkov)                                 | `data/raw/sparkov.csv`                                                 |
| combine        | `src/combine_datasets.py`       | `data/raw/{paysim,sparkov}.csv`                            | `data/processed/combined_transactions.csv`                             |
| preprocess     | `src/preprocess.py`             | combined_transactions.csv                                  | `data/processed/{features.csv, feature_thresholds.json}`               |
| train          | `src/train.py`                  | features.csv + feature_thresholds.json                     | `models/fraud_model.pkl`, `X_test.csv`, `y_test.csv`                   |
| evaluate       | `src/evaluate.py`               | model + X_test/y_test                                      | `metrics/scores.json`, `reports/confusion_matrix.svg`, `roc_curve.svg` |
| benchmark_fdb  | `src/benchmark_fdb.py`          | model + (FDB sparkov split or local sparkov)               | `metrics/fdb_benchmark.json`                                           |

Model: XGBoost (`scale_pos_weight=50`, n_estimators=200, max_depth=6, lr=0.1).
Optional SMOTE on training split.

## 2. Pre-fix state (what was being reported)

`metrics/fdb_benchmark.json` (pre-fix snapshot) advertised:

```
our_model_auc:  0.9210  (sparkov)
AutoGluon:      0.952   (BELOW by 0.031)
H2O AutoML:     0.947   (BELOW by 0.026)
AutoSklearn:    0.931   (BELOW by 0.010)
```

That number is **not honest**. Two leakage paths produced it.

## 3. Leakage Bug #1 — random `train_test_split` in `src/train.py:47-53`

Original code:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state,
    stratify=y if y.nunique() > 1 else None,
)
```

For fraud-detection on time-series transactions, this is wrong:

- Fraud patterns evolve over time (new merchant categories, new compromise
  vectors, seasonal velocity changes).
- A random split puts March 2020 txns in train AND March 2020 txns in test
  — the model gets to "look up the answer" via correlated features it just
  saw a week earlier in the same period.
- The honest production setup is: train on the past, evaluate on the
  strictly-later future. Anything else inflates test metrics.

**Fix:** `temporal_split_per_source(...)` in `src/train.py`. For each
`source_*` one-hot group, sort by `txn_timestamp` and take the LAST
`test_size` fraction as test. Per-source because paysim's `step` timeline
and sparkov's unix timestamps live on different scales and can't be merged
into a single sortable axis.

## 4. Leakage Bug #2 — benchmark fell back to the training file

`src/benchmark_fdb.py` originally fell back to `data/raw/sparkov.csv` when
the FDB package was unavailable. But:

- `data/raw/sparkov.csv` is the SAME file consumed by the combine stage.
  So a row in sparkov.csv is in both train.py's training data AND the
  benchmark's "test" data.
- That makes `benchmark_fdb.json`'s 0.921 a measurement of memorisation,
  not generalisation.

There's a clean alternative right there on disk:
`data/raw/sparkov_test.csv` (555,719 rows, 2020-06-21 → 2020-12-31). It's
chronologically after sparkov.csv's range (2019-01-01 → 2020-06-21) and
was never read by the train pipeline.

**Fix:** `src/benchmark_fdb.py` now prefers `sparkov_test.csv` and only
falls back to `sparkov.csv` with a printed WARNING.

## 5. Post-fix honest numbers (2026-05-18)

| Measurement                                                      | AUC      | AP      | Notes                                                  |
|------------------------------------------------------------------|----------|---------|--------------------------------------------------------|
| Pre-fix benchmark (random split + train file as test)            | 0.9210   | —       | Retired today. Inflated by stacked leakage.            |
| Post-fix combined temporal test (1.53M rows, paysim+sparkov)     | 0.9989   | 0.8280  | Dominated by paysim's near-deterministic features.     |
| Post-fix paysim-only train-time temporal test (1.27M rows)       | 0.9997   | 0.9170  | paysim signal doesn't decay in-distribution.           |
| Post-fix sparkov-only train-time temporal test (259k rows)       | 0.9966   | 0.8090  | Final 20% of training file. Same data-collection regime.|
| **Post-fix benchmark on held-out sparkov_test.csv (Jun–Dec 2020)** | **0.7949** | —     | **The honest sparkov AUC. -0.157 vs AutoGluon 0.952.** |

The ~0.20 AUC drop between the train-time sparkov test slice and the
held-out sparkov_test.csv file is the real **distribution shift** the
rest of the sprint has to close — and it's exactly what Day-3's drift
detector + auto-retrain will be measured against.

## 6. MLflow tracking (Day-1 addition)

- Local sqlite store at `mlflow.db` (gitignored).
- Experiment: `sentinel-day01-temporal-split`.
- Every `python -m src.train` run is now wrapped in `mlflow.start_run()`.
- Params, metrics, and the model artifact log automatically. View with:
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## 7. Where the upgrade is going

| Day | Add                                                                                          |
|-----|----------------------------------------------------------------------------------------------|
| 2   | Dask-based feature engineering (`src/features/engineer.py`) + MLflow registry promote/rollback |
| 3   | Drift detector (`src/drift/detector.py`) + auto-retrain trigger                              |
| 4   | Production refactor: shadow deploy, Postgres telemetry, FastAPI                              |
| 5   | Optuna sweep + failure-mode analysis                                                         |
| 6   | LLM-judged-fraud negative result + frontier comparison                                       |
| 7   | Docker Compose stack + Streamlit ops dashboard + tests + README                              |

Differentiation guard: Sentinel's resume claim is **MLOps discipline at
scale** (drift response time, throughput, rollback latency). The joint
Fraud Detection project owns AUPRC/ensemble/SHAP — Sentinel does not
compete there.
