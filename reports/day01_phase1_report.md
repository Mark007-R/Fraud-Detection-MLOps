# Day 01 — Audit + Temporal-Split Fix + Baseline — Sentinel
**Date:** 2026-05-18
**Day:** 01 of 7

## Resume gap progress
**Gap:** MLOps discipline (drift response time, throughput, registry rollback)
— NOT model quality (joint Fraud Detection's territory).
**Today's contribution:** Documented the 6-stage pipeline, identified and
fixed two stacked leakage bugs (random split + benchmark on training file),
and stood up local MLflow tracking. Pre-fix sparkov AUC of 0.921 is retired;
honest sparkov AUC on a truly-held-out file is **0.795**. That's Sentinel's
canonical baseline for the rest of the sprint.

## Files touched
- `src/combine_datasets.py` (lines 44–55, 56–80, 105–117, 134, 181–185) — added `txn_timestamp` to both source standardisers; replaced shuffle with `sort_values(["source","txn_timestamp"])`
- `src/preprocess.py` (lines 77–92, 109–127) — added `txn_timestamp` to required schema + passthrough through `engineer_features_df`
- `src/train.py` (full rewrite) — replaced `train_test_split(stratify=y)` with `temporal_split_per_source`; wrapped run in `mlflow.start_run()`; logs params, metrics, model artifact
- `src/benchmark_fdb.py` (lines 162–179) — prefer held-out `sparkov_test.csv`, fall back to `sparkov.csv` only with WARNING
- `docs/MLOPS_AUDIT.md` (new) — 7-section audit of pipeline + two leakage bugs + honest numbers + roadmap
- `docs/DATA_SPLIT.md` (new) — rationale for per-source temporal split
- `results/baseline_metrics.json` (new) — canonical baseline numbers
- `results/per_source_test_metrics.json` (new) — per-source AUC breakdown
- `.gitignore` (updated, see Code Changes section)

## Setup
- **Compute:** Local CPU, single host
- **Packages installed:** `mlflow==2.18.0`
- **Datasets used:** `data/raw/{paysim.csv (6.36M rows), sparkov.csv (1.30M rows, 2019.01-2020.06), sparkov_test.csv (555,719 rows, 2020.06-2020.12)}`
- **Components touched:** combine → preprocess → train → evaluate → benchmark_fdb (entire DVC graph)
- **MLflow:** local sqlite at `mlflow.db`, experiment `sentinel-day01-temporal-split`

## Experiments

### Experiment 1.1: Confirm Leakage Bug #1 — random `train_test_split`
**Hypothesis:** A random stratified split on time-series fraud data leaks
future patterns into training, inflating test AUC.
**Method:** Read `src/train.py:47-53`. Confirmed `train_test_split(...,
stratify=y if y.nunique() > 1 else None)`. No `shuffle=False`, no temporal
ordering, no time-aware splitter. Combined with `combine_datasets.py:182`'s
`combined.sample(frac=1.0)` shuffle that also destroyed source-level order.
**Result:** Bug confirmed. Both files needed changes.
**Interpretation:** Random splits assume row exchangeability; fraud
transactions are not exchangeable in time (compromised cards reissued in
later months, merchant compromises cluster temporally, transaction-mix
distribution drifts over the year). Training on May 2020 and testing on
March 2020 lets the model use information unavailable at production score
time.

### Experiment 1.2: Confirm Leakage Bug #2 — benchmark on training file
**Hypothesis:** `benchmark_fdb.py` falls back to `data/raw/sparkov.csv`,
which is the SAME file the train stage reads.
**Method:** Read `src/benchmark_fdb.py:162-165`. Confirmed
`sparkov_local = Path("data/raw/sparkov.csv")` is the only fallback. Then
checked `data/raw/` for alternatives — found `sparkov_test.csv` (555k rows,
2020-06-21 → 2020-12-31, strictly after `sparkov.csv`'s 2019-01-01 → 2020-06-21
range).
**Result:** Bug confirmed. Held-out file exists on disk but was unused.
**Interpretation:** Two leakage paths stacked: even if the train split were
fixed, the benchmark would still measure memorisation because the test
file is the train file. Both fixes are needed for the 0.795 number to
mean anything.

### Experiment 1.3: Per-source temporal split implementation
**Hypothesis:** Sorting each source's rows by `txn_timestamp` and taking
the last 20% as test produces a strictly-future evaluation regime.
**Method:** Implemented `temporal_split_per_source(df, test_size, timestamp_col)`
in `src/train.py`. For each `source_*` one-hot column: filter, sort by
timestamp, take last 20%. Verified `ts_train_max <= ts_test_min` for each
source via stage logs:
- paysim: `ts_train_max=1278000, ts_test_min=1278000` (boundary on step ~355h)
- sparkov: `ts_train_max=1583478917 (2020-03-06), ts_test_min=1583479003 (2020-03-06)` (boundary mid-March 2020)
**Result:** Temporal monotonicity verified per source.
**Interpretation:** The split is now honest. The boundary equality on
paysim is expected (sub-hour resolution causes ties at the step boundary)
and on sparkov is microsecond-tight, which is correct.

### Experiment 1.4: Post-fix end-to-end pipeline run
**Hypothesis:** After both fixes, the published benchmark AUC drops
materially.
**Method:** `python -m src.combine_datasets && python -m src.preprocess && python -m src.train && python -m src.evaluate && python -m src.benchmark_fdb`. Each stage ran to completion. The train stage logged to MLflow run `8b630c3ec7114b3995661f631efac4f6`.
**Result:**

| Measurement | Pre-fix | Post-fix | Δ |
|---|---|---|---|
| **Benchmark sparkov AUC** (vs AutoGluon 0.952) | **0.9210** | **0.7949** | **-0.1261** |
| Combined temporal test AUC | (n/a, was random) | 0.9989 | — |
| Combined temporal test AP | (n/a) | 0.8280 | — |
| Sparkov-only train-time test AUC | (n/a) | 0.9966 | — |
| Paysim-only train-time test AUC | (n/a) | 0.9997 | — |
| `metrics/scores.json` precision (combined) | 0.337 | 0.527 | +0.190 |
| `metrics/scores.json` recall (combined) | 0.962 | 0.897 | -0.065 |
| `metrics/scores.json` F1 (combined) | 0.499 | 0.664 | +0.165 |

**Interpretation:** Three findings sit on top of each other.

1. **The 0.921 benchmark was inflated by ~0.126 AUC of stacked leakage.**
   The honest sparkov AUC on a truly-held-out file is 0.795. Sentinel is
   now BELOW AutoGluon by 0.157, not 0.031.
2. **Combined temporal test AUC is 0.999, dominated by paysim** (83% of
   total rows). PaySim's `balance_change_orig` is essentially a
   deterministic fraud signal — once you know it, sparkov-style noise
   gets averaged out. This is why the combined number is not useful as a
   resume claim and why the per-source breakdown is.
3. **Sparkov in-period AUC (0.997) vs sparkov out-of-period AUC (0.795)
   reveals a 0.20-point distribution shift.** The model interpolates well
   within its training time range, then degrades materially in Jun–Dec
   2020. That delta IS the gap that Day-3's drift detector + auto-retrain
   pipeline has to close. It's also what makes the MLOps story
   interesting — without drift response, Sentinel's "production" AUC
   slowly slides from 0.997 toward 0.795 as data ages.

## Head-to-Head Comparison

| Rank | Strategy | Primary (sparkov benchmark AUC) | Combined test AUC | Honesty | Notes |
|------|----------|---------------------------------|-------------------|---------|-------|
| 1 | **Post-fix: temporal split + held-out file** | **0.7949** | 0.9989 | ✅ Honest | Today's baseline. -0.157 vs AutoGluon. |
| 2 | Pre-fix: random split + train-file benchmark | 0.9210 | (n/a) | ❌ Inflated (stacked leakage) | The retired claim. |

## Key Findings
1. **The biggest finding of the sprint may already be in.** A 0.126-point
   AUC drop from the previously advertised number is bigger than any
   single modeling improvement is likely to deliver in Days 2–6. The
   resume story shifts from "we beat AutoGluon" (we don't) to "we
   discovered our own leakage, fixed it, and now run a real-world honest
   MLOps loop on top of the corrected baseline."
2. **Distribution shift is the dominant signal on sparkov, not modeling
   choice.** The in-period sparkov AUC (0.997) is nearly tied with
   AutoGluon's 0.952 — the real performance loss comes when the model
   ages out of its training window. This is exactly the failure mode
   Day-3's drift detector and Day-3's auto-retrain trigger are designed
   to catch.
3. **Even per-source temporal splits aren't enough to flatter paysim.**
   paysim's `balance_change_orig` is a deterministic fraud signal — any
   well-tuned tree model will hit 0.999 on it. The honest resume metric
   for Sentinel must be sparkov-only AUC vs AutoGluon, not the combined
   number.

## What Didn't Work
- The previously published 0.921 claim relied on two leakage paths
  simultaneously. Fixing one and not the other would still produce a
  misleading number, so both fixes had to land in the same Day-1 commit.
- The `sparkov_test.csv` file was on disk the whole time but was never
  preferred over the leaking `sparkov.csv` fallback. The cost of that
  oversight was the 0.126 AUC inflation. Subtle path defaults eat real
  metric integrity — Day-7's tests must include a regression test on the
  benchmark file selection.

## Sample Outputs Saved
- `models/fraud_model.pkl` — post-fix XGBoost (temporal-split trained)
- `metrics/scores.json` — combined-test classification report (post-fix)
- `metrics/fdb_benchmark.json` — held-out sparkov benchmark (post-fix)
- `results/baseline_metrics.json` — canonical Day-1 numbers
- `results/per_source_test_metrics.json` — per-source AUC breakdown
- `reports/{confusion_matrix.svg, roc_curve.svg}` — regenerated SVGs
- `mlflow.db` — local MLflow store; experiment `sentinel-day01-temporal-split`

## Next Day (Day 2 — Phase 2a)
Strategy A — Dask-based feature engineering at `src/features/engineer.py`,
benchmark throughput vs single-node Pandas at 100K, 500K, 1M rows.
Strategy B — MLflow model registry CLI: `src/registry/promote.py` and
`src/registry/rollback.py`, tested on at least 2 model versions. Measure
rollback latency.

## Code Changes
- `src/combine_datasets.py`:
  - Lines 44–55, 56–80: `_normalize_sparkov` now emits `txn_timestamp` from `trans_date_trans_time` / `TX_TIMESTAMP` / `unix_time`
  - Lines 105–117: `_normalize_paysim` now emits `txn_timestamp = step * 3600`
  - Line 134: `_validate_combined` requires `txn_timestamp`
  - Lines 181–185: replaced `combined.sample(frac=1.0, random_state=42)` with `combined.sort_values(["source","txn_timestamp"], kind="mergesort")`
- `src/preprocess.py`:
  - Lines 77–92: `engineer_features_df` requires `txn_timestamp`
  - Lines 109–127: `txn_timestamp` is a passthrough column on the engineered output
- `src/train.py`: full rewrite — new `temporal_split_per_source` function, MLflow `start_run()` wrapper, per-stage metric logging
- `src/benchmark_fdb.py` (lines 162–179): prefer `sparkov_test.csv`, fall back to `sparkov.csv` with WARNING
- `.gitignore`: added `mlflow.db`, `mlruns/`, `mlartifacts/`
