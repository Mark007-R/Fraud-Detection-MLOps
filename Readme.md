# Fraud-Detection-MLOps

> 🔗 **Live demo:** https://iambatman07-fraud-detection-mlops.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/Fraud-Detection-MLOps)

Fraud detection on payment transactions (PaySim + Sparkov), built as an MLOps pipeline: a temporal train/test split, MLflow registry promotion and rollback, KS+PSI drift detection, auto-retrain-and-promote, Dask feature engineering, and Postgres-backed telemetry on a DVC pipeline.

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

# Demo script
bash scripts/demo.sh

# Tests
pytest tests/ -q
pytest tests/ -q -m "not requires_data"          # CI mode (skips full-data replay)
```

---

## Datasets

| Dataset | Rows | Window | Use |
|---|---:|---|---|
| `paysim.csv` | 6.36M | step-based sim | training |
| `sparkov.csv` | 1.30M | 2019-01 → 2020-06 | training window |
| `sparkov_test.csv` | 555,719 | 2020-06 → 2020-12 | held-out out-of-time |

DVC-tracked; `dvc pull` to materialize. The split is chronological per source — the final 20% of each is held out.

---

## License

MIT. See [LICENSE](LICENSE).
