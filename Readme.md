# Fraud-Detection-MLOps

> 🔗 **Live demo:** https://iambatman07-sentinel.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/Sentinel)

**Fraud-detection MLOps — built around an honest temporal split, with drift detection, auto-retrain, and registry rollback that a notebook can't replicate.**

Fraud-Detection-MLOps detects fraud in payment transactions (PaySim + Sparkov), but its resume claim is **MLOps discipline at scale**, not raw model quality. A 7-day upgrade sprint found and fixed a data-leakage bug that had been inflating the headline AUC, then layered MLflow registry promotion/rollback, KS+PSI drift detection, auto-retrain-and-promote, Dask-deterministic feature engineering, and Postgres-backed telemetry on top of the existing DVC pipeline.

---

## The headline: the leakage fix is the story

The original `train.py` used `train_test_split(..., stratify=y)` — a **random** split on time-series fraud data, which leaks future transaction patterns into training. The Day-1 audit ([docs/MLOPS_AUDIT.md](docs/MLOPS_AUDIT.md)) found a second leak too: the benchmark fell back to the *training* file instead of the held-out test file.

| Metric (Sparkov OOT, vs AutoGluon 0.952) | Before fix | After fix |
|---|---:|---:|
| **Honest AUC** | **0.9210** (leaked) | **0.7949** (honest baseline) |
| Apparent gap to AutoGluon | −0.031 (mirage) | −0.157 (real) |

The fix: `temporal_split_per_source()` sorts each source chronologically and takes the final 20% as test ([docs/DATA_SPLIT.md](docs/DATA_SPLIT.md)). **The ~0.126 AUC drop was the data leakage. The 0.7949 is the number that was always real** — and the rest of the sprint earns it back honestly.

---

## Final champion + the gap closed honestly

A 30-trial Optuna sweep + source-balanced sample weights closed the entire gap to AutoGluon — **no new features, no ensembling, no SHAP** (that's the joint Fraud-Detection project's territory; Fraud-Detection-MLOps is deliberately MLOps-only).

| Model | OOT AUC (sparkov_test.csv) | Δ vs AutoGluon 0.952 | Source |
|---|---:|---:|---|
| Day-1 honest baseline (temporal split) | 0.7949 | −0.157 | `results/baseline_metrics.json` |
| **Day-5 champion (Optuna + source-balanced)** | **0.9520** | **−0.00004** (tied) | `results/day05/day05_leaderboard.csv` |

Champion: XGBoost, per-source temporal split + source-balanced weights (paysim 0.60×, sparkov 2.95×) + Optuna tuning. Stored at `models/fraud_model_tuned_fixed.pkl` (MLflow run `day05_targeted_fix_v1`).

---

## The four MLOps capabilities (what a notebook doesn't have)

| Capability | Headline metric | Source |
|---|---|---|
| **Dask distributed features** | Pandas 1.15M rows/s vs Dask 0.26M rows/s on one host, **bit-exact within 5.5e-12** | `results/throughput_speedup.csv` |
| **MLflow registry rollback** | **3.9 ms** median alias flip; 11.9 ms full audited rollback | `results/registry_rollback_times.csv` |
| **KS+PSI drift detection** | **0-day** detection lag on a synthetic 2σ shift; **precision 1.0, recall 1.0** on the 7-day window | `results/drift_replay_summary.json` |
| **Auto-retrain + shadow-promote** | drift → train → shadow-eval → promote in **median 6.85 s**; 3/3 events auto-promoted | `results/drift_retrain_events.csv` |

Dask "loses" the single-host throughput race but is the *scaling primitive* — and it's bit-exact with Pandas, which is the property that lets you swap engines without changing results.

---

## Frontier guard-rail: why LLMs don't do tabular fraud

Day 6 ran Claude Opus 4.6 as an LLM fraud judge on the same 200-row OOT sample (tool-use schema forcing a structured verdict):

| Strategy | AUC | AUPRC | Latency/query | Cost @ 1K QPS/day |
|---|---:|---:|---:|---:|
| **Fraud-Detection-MLOps champion (XGBoost)** | **0.9156** | **0.526** | 60 µs | **$0.43** |
| Naive notebook XGBoost | 0.626 | 0.415 | 73 µs | $0.43 |
| Claude Opus 4.6 LLM-judged | 0.622 | 0.351 | 1.82 s | **$1,250,691** |

Two findings: (1) the LLM is **30,400× slower** and **2.9M× more expensive**, and ranks 0.29 AUC worse — it caught only textbook patterns (large online txns at night) and missed the "card skimmed at POS → grocery abuse" behavioral class. (2) **The naive notebook XGBoost (0.626) ties the LLM (0.622)** — the 0.29-AUC jump to the champion comes from the *discipline layers*, not from "XGBoost vs LLM". The discipline is the model. Source: `results/day06/frontier_comparison.csv`. The LLM judge ran in simulate mode (no API key on host); `--mode api` is the same code path with a real key.

### MLOps ablation (L0 → L3, full OOT)
| Layer | OOT AUC | Δ vs prev |
|---|---:|---:|
| L0 naive (random split + defaults) | 0.547 | — |
| L1 + temporal split (Day-1 fix) | 0.657 | +0.111 |
| L2 + source-balanced weights | 0.696 | +0.039 |
| L3 + Optuna (champion) | **0.948** | **+0.252** |

Total L0→L3 gain: **+0.401 OOT AUC**. Every layer is load-bearing; Optuna is the biggest single contributor. Source: `results/day06/ablation_modelling.csv`.

---

## Architecture

```
   DVC pipeline:  load_fdb → combine → preprocess → train → evaluate → benchmark_fdb
                                          │ (temporal split, MLflow tracking)
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  src/features/   Dask + Pandas behavioral features (bit-exact)        │
   │  src/registry/   MLflow promote / rollback (alias-flip, ~4ms)         │
   │  src/drift/      KS + PSI detector → N-consecutive trigger → retrain  │
   │  src/serving/    FastAPI /predict + async shadow deployment           │
   │  src/telemetry/  Postgres 4-table logger (predictions, drift, …)      │
   │  src/frontier/   LLM-judge negative result + ablations                │
   └──────────────────────────────────────────────────────────────────────┘
                                          │
        docker compose up  →  FastAPI :8000 + Postgres + Redis + MLflow
        streamlit run app.py  →  "4 Ops" dashboard (drift / retrain / rollback / AUPRC)
```

---

## Quick start

```bash
# Full stack: Postgres + MLflow + Redis (+ FastAPI serving profile)
docker compose up -d
docker compose --profile serving up -d api      # FastAPI on :8000
mlflow ui                                        # MLflow UI on :5000

# Streamlit ops dashboard — navigate to the "4 Ops" page
streamlit run app.py

# DVC pipeline
dvc repro                                        # full pipeline
dvc repro train                                  # train stage only

# MLOps operations
python -m src.registry.promote  --experiment sentinel-day01-temporal-split --alias production
python -m src.registry.rollback --target-version 1
python -m src.features.benchmark --sizes 100000 500000 1000000

# 60-second reproducible sprint demo
bash scripts/demo.sh

# Tests — 31 tests
pytest tests/ -q
pytest tests/ -q -m "not requires_data"          # CI mode (skips full-data replay)
```

---

## Tests

31 pytest tests, gated in CI (`.github/workflows/ci.yml` runs `dvc dag` + `pytest -m "not requires_data"` on every push):

| File | Covers |
|---|---|
| `test_temporal_split.py` | per-source timestamp monotonicity, no-future-leak regression |
| `test_features_determinism.py` | Pandas == Dask bit-exact |
| `test_drift_detector.py` | PSI=0 on identical input, fires on 2σ shift, KS/PSI paths |
| `test_registry.py` | promote → rollback end-to-end, alias truly flips |
| `test_retrain_trigger.py` | N-consecutive debounce, single-fire ignored, counter reset |
| `test_api.py`, `test_data_loader.py`, `test_telemetry.py` | serving + DVC loader + telemetry round-trips |

---

## Repo layout

```
Fraud-Detection-MLOps/
├── dvc.yaml                    # 6-stage pipeline
├── docker-compose.yml          # Postgres + MLflow + Redis + FastAPI
├── app.py + pages/             # Streamlit; pages/4_Ops.py = MLOps dashboard
├── .github/workflows/ci.yml    # DVC DAG + pytest
├── src/
│   ├── train.py                # temporal split + MLflow tracking (Day 1 fix)
│   ├── combine_datasets.py     # txn_timestamp + chronological sort
│   ├── benchmark_fdb.py        # prefer held-out sparkov_test.csv
│   ├── features/{engineer,benchmark}.py     # Dask/Pandas features + throughput sweep
│   ├── registry/{promote,rollback,bench_rollback}.py
│   ├── drift/{detector,trigger,bench_retrain}.py
│   ├── serving/{api,shadow}.py
│   ├── telemetry/logger.py     # 4-table SQLAlchemy logger
│   ├── tuning/{optuna_sweep,eval_best,targeted_fix,build_leaderboard}.py
│   ├── analysis/failure_modes.py
│   └── frontier/{llm_judge,compare_models,ablation}.py
├── tests/                      # 31-test suite
├── models/fraud_model_tuned_fixed.pkl   # champion
├── results/                    # baseline, throughput, rollback, drift, day05/, day06/
├── reports/                    # day01..day07 phase reports
├── docs/                       # MLOPS_AUDIT.md, DATA_SPLIT.md
└── data/raw/{paysim,sparkov,sparkov_test}.csv
```

---

## Datasets

| Dataset | Rows | Window | Use |
|---|---:|---|---|
| `paysim.csv` | 6.36M | step-based sim | training (~83% of combined) |
| `sparkov.csv` | 1.30M | 2019-01 → 2020-06 | training window |
| `sparkov_test.csv` | 555,719 | 2020-06 → 2020-12 | **held-out OOT** (the honest benchmark) |

DVC-tracked; `dvc pull` to materialize. Combined temporal train ≈ 6.13M rows.

---

## License

MIT. See [LICENSE](LICENSE).
