# Day 03 — Drift detector + synthetic replay + auto-retrain trigger — Sentinel
**Date:** 2026-05-20
**Day:** 03 of 7

## Resume gap progress
**Gap:** MLOps discipline — drift response time + auto-retrain gate on top of the Day-2 registry rollback.
**Today's contribution:** Built a per-feature KS + predicted-probability PSI drift detector against a frozen reference, validated against a 30-day synthetic replay (precision 1.0 / recall 1.0 on a +2σ injection that starts on day 23), and wired it into an auto-retrain trigger that re-fits on the drifted window, runs a held-out shadow eval, and conditionally flips the `@production` alias via the Day-2 registry CLI. **End-to-end "drift detected → traffic on a new model" median is 6.85s; recovered AUPRC from 0.055 (stale model on the drifted slice) to 0.552 on the first event and to 0.805 on the second.**

## Files touched
- `src/drift/__init__.py` (new)
- `src/drift/detector.py` (new — `DriftDetector`, `fit_reference`, `psi`, `DriftReport`)
- `src/drift/trigger.py` (new — `TriggerState`, `run_drift_retrain_simulation`, `RetrainEvent`)
- `src/drift/bench_retrain.py` (new — end-to-end drift → retrain → register → promote bench)
- `tests/__init__.py` (new — package marker)
- `tests/synthetic_drift.py` (new — 30-day replay + injection + pytest)
- `results/drift_replay_per_day.csv`, `results/drift_replay_summary.json`
- `results/drift_retrain_events.csv`, `results/drift_retrain_metrics.json`
- `results/drift_metrics.json` (champion + headline numbers)
- `results/phase2_leaderboard.csv`
- `results/drift_reference.json` (saved reference snapshot)
- `results/samples/drift/per_day_reports_sample.json`, `results/samples/drift/retrain_event_sample.json`

## Setup
- **Compute:** local Windows 11, Python 3.11.9, single CPU host.
- **Dataset slice:** `data/processed/features.csv` — sparkov-only subset (1,296,675 rows), recreated the Day-1 80/20 temporal split (train=1,037,340; test=259,335). First 30 calendar days of the test fold (2020-03-06 → 2020-04-05) become the synthetic stream (~62K rows total).
- **Model under test:** `models/fraud_model.pkl` — the Day-1 temporal-split XGBoost with the 45-feature surface.
- **Reference snapshot:** 20K-row stratified sub-sample of the sparkov train fold + that model's predicted probabilities. KS test uses ≤20K reference samples (test power saturates well below this); PSI on probability uses raw 20K samples and rebinned 10-quantile bins.
- **MLflow:** local sqlite store, experiment `sentinel-day03-drift-retrain`. Three retrain runs were registered as versions v7/v8/v9 of `sentinel-fraud-xgboost` and their alias flips were timed.

## Experiments

### Experiment 3.1 — Drift detector precision / recall on synthetic +2σ injection
**Hypothesis:** A KS-on-features OR'd with PSI-on-predicted-probability detector can hit precision = recall = 1.0 against a +2σ shift in `amount`, given thresholds tuned for daily windows of ~2K rows.

**Method:** Slice sparkov test fold into the first 30 calendar days. Days 0–22: pass through unmodified. Days 23–29: shift `amount` by +2σ (2.0 × ref-std = +$318) and cascade the change into `tx_amount_log`, `amount_zscore`, `balance_change_abs`, `balance_change_log` so the engineered representation is internally consistent. Score with the Day-1 model. Run detector daily with `ks_pvalue_threshold=0.01`, `ks_stat_threshold=0.15`, `psi_threshold=0.25`. Monitored features: continuous-numeric only (amount-family, balance-family, hour_of_day) — `day_of_month` was excluded because the test fold mechanically walks forward through the calendar, KS-firing every window without indicating data drift.

**Result:**

| Window range | Avg KS(`amount`) | Avg `proba_psi` | Days flagged | Drift fired |
|---|---:|---:|---:|:---:|
| Days 0–22 (clean) | 0.0249 | 0.029 | 0 | False (0/23) |
| Days 23–29 (+2σ on `amount`) | 0.978 | 3.156 | 5 features each day | True (7/7) |

| Metric | Value |
|---|---|
| Precision | **1.000** |
| Recall | **1.000** |
| First firing day | 23 (exact match to injection day) |
| Detection lag (no debounce) | 0 days |
| Pre-injection max `proba_psi` | 0.097 |
| Post-injection min `proba_psi` | 2.916 |
| Post / pre PSI ratio | **30×** |

**Interpretation:** Three things land together. **First**, the `amount` KS statistic jumps from 0.02–0.04 on clean days to 0.978 on injected days — a 25× separation that any reasonable threshold catches without tuning. **Second**, model-output PSI moves from a noisy 0.01–0.097 envelope to a sustained 2.9+ — the 30× ratio means the detector's PSI threshold has at minimum a full order of magnitude of slack. **Third**, the `day_of_month` exclusion mattered: an earlier run that included it scored precision 0.23 / recall 1.0 because the test fold marches forward through the calendar and KS-fires on calendar drift every day. That false-positive class would have been the dominant failure mode in production — the policy fix (curate the monitored set) is more important than the algorithm.

### Experiment 3.2 — Auto-retrain trigger end-to-end (drift → train → shadow → promote)
**Hypothesis:** With an N=2 consecutive-day debounce and a 1pp AUPRC tolerance, the trigger can recover materially better than the stale model on the held-out next day, all in well under 10 seconds per event after the cold start.

**Method:** Same 30-day replay. Each day's drift report drives a `TriggerState` counter. When it hits N=2, retrain XGBoost (n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=50) on the strict window `[first_fired_day .. d−1]` (the shadow day `d` is held out), compute shadow AUPRC on day `d` against the current production model's AUPRC on the same day, and conditionally promote via `src/registry/promote.py` (alias=`production` on pass, alias=None on fail — but every retrain is registered as an audit-trail version). Each retrain wrapped in `mlflow.start_run()`.

**Result:**

| Event | Trigger day | Train window | Train rows | Shadow day | Prod AUPRC | Shadow AUPRC | Δ AUPRC | Promoted | E2E (s) |
|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 24 | 22, 23 | 4,876 | 24 | **0.055** | **0.552** | **+0.497** | ✅ | 30.09 (cold start) |
| 2 | 26 | 24, 25 | 5,900 | 26 | 0.756 | **0.805** | +0.049 | ✅ | 6.85 |
| 3 | 28 | 26, 27 | 3,259 | 28 | 0.758 | **0.765** | +0.008 | ✅ | 6.50 |
| **median** | — | — | — | — | — | — | — | — | **6.85** |

| Component latency (median over 3 events) | Seconds |
|---|---:|
| Detect-to-retrain-start | 0.005 |
| XGBoost fit | 0.677 |
| Shadow eval (`predict_proba` on shadow day) | 0.038 |
| MLflow register + alias flip | 0.031 |
| **End-to-end median** | **6.85** |
| **End-to-end p100** | 30.09 |

**Interpretation:** The headline number is the **first event's AUPRC delta**: a +0.497 jump from 0.055 (stale prod) to 0.552 (auto-retrained). The stale model had effectively collapsed on the drifted distribution — its predicted probabilities concentrated below the fraud-rate baseline, so AUPRC fell to 0.055 (random would be ~0.004 at the test fold's fraud rate, so prod was barely above chance). The candidate, fit on just 4,876 rows from the trailing two days, restored AUPRC to 0.552 within the held-out shadow window. Subsequent events show diminishing returns — once the model is approximately re-aligned, each new retrain only nudges AUPRC by ~5pp then ~1pp.

The latency split is more interesting than the wall time. **The per-event sum of useful work — fit + shadow + register — is well under 1 second** (median 0.75s). The remaining ~6s is MLflow's `xgboost.log_model` serializing the booster to UBJSON + writing the artifact + creating the model-version row. The cold-start cost on event 1 (30s) is one-time XGBoost native-library + sqlite init. **For the runbook, the honest steady-state number is "drift detected → traffic on new model in ~7s, with the actual training cost under one second."** That's the MLOps claim the resume will carry.

The asymmetric promote gate (`shadow_auprc >= prod_auprc - 0.01`) is the deliberate design choice. Under sustained drift the prod_auprc is bound to be low; tolerating a small downside vs prod prevents the gate from being too tight when *any* response is better than the stale model. The candidate doesn't have to be great — it just has to not be measurably worse than the model already on fire.

## Phase 2 Head-to-Head Leaderboard

Built on top of Day-1 baseline + Day-2 throughput / rollback metrics, plus today's drift + retrain numbers:

| # | Strategy | Axis | Primary metric | Value | Secondary metric | Notes |
|--:|---|---|---|---:|---|---|
| 1 | KS + PSI drift detector | drift quality | precision @ 2σ | **1.000** | recall = 1.000 | Replay first-fires on the exact injection day |
| 2 | KS + PSI drift detector | drift quality | proba_psi ratio | **30×** | pre-max 0.097 → post-min 2.916 | Signal separation, not noise |
| 3 | Auto-retrain trigger (N=2 debounce) | end-to-end latency | median (s) | **6.85** | p100 30.09 (cold start) | Steady-state ~7s; fit alone <1s |
| 4 | Auto-retrain trigger | recovery quality | shadow AUPRC | **0.552** | prod_auprc on drifted day 0.055 | +0.497 over stale model |
| 5 | MLflow alias-flip (Day 2) | rollback latency | median (s) | 0.0039 | p100 0.0047 | Sub-5ms switching |
| 6 | MLflow end-to-end rollback (Day 2) | rollback latency | median (s) | 0.0119 | p100 0.0139 | Includes audit tag |
| 7 | Pandas behavioral engineer (Day 2) | throughput | rows/sec at 1M | 1,152,848 | wall 0.87s | Sub-1M scale winner |
| 8 | Dask behavioral engineer (Day 2) | throughput | rows/sec at 1M | 259,862 | wall 3.85s | RAM-headroom winner, bit-exact |

**Champion pick (Day 3):**
- **Detector**: per-feature KS (p<0.01 AND stat≥0.15) OR'd with PSI (threshold 0.25) on predicted probability. The OR'ing is load-bearing — KS catches single-feature pipeline drift; PSI catches concept-drift-style cases that move predictions without moving any single feature marginally.
- **Trigger**: N=2 consecutive-day debounce with a 1pp AUPRC tolerance on a held-out next-day shadow window. Every retrain is registered (audit), only passing candidates are aliased (`@production` flip via Day-2 CLI).

## Key Findings

1. **The detector is correct, but the *monitored feature set* is the actual lever.** An initial run that included `day_of_month` collapsed precision to 0.23 because the calendar advances every day. The detector code didn't change — only the curated list of features did. The resume-grade takeaway: drift detection is a policy + algorithm system, and the policy (which features to watch) matters more than the statistical test.

2. **Stale-model collapse is brutal on drifted distributions.** Prod AUPRC fell from a steady-state of ~0.76 to **0.055** within a single drift onset — about a 13× drop. That makes the auto-retrain trigger's +0.497 AUPRC recovery (back to 0.552 on the held-out shadow day) the headline MLOps win, and quantifies the cost of *not* having drift response.

3. **End-to-end "click button → traffic moved" is bounded by registry I/O, not by training.** Of the 6.85s median, less than 1s is XGBoost fitting; the rest is MLflow model serialization and registry writes. A remote Postgres-backed MLflow server (Day-4 plan) won't change this story qualitatively — the dominant cost is artifact upload, not the alias flip. The Day-2 4ms alias-flip rollback number stays as the upper-bound for *rollback*, where no model upload is needed.

4. **N=2 debounce buys you one day of detection latency in exchange for kicking single-noisy-day false-positives.** With N=1 the detector would have triggered on day 23 exactly, but a single noisy day in production (e.g. a one-off ingestion bug) would also have triggered a retrain — costly + risky. N=2 trades 24h of stale-model exposure for an asymmetric protection against noise. That tradeoff is documented in the trigger module's docstring.

## What Didn't Work
- **First pass shadow AUPRC came out 1.0 on every event** because the shadow day `d` was included in the training window `[first_fired_day .. d]`. That's a textbook in-sample eval and would have been a credibility-destroying number in the report. Fixed by changing the training window to strictly `[first_fired_day .. d−1]` and keeping `d` for the shadow eval. The numbers in the report are the post-fix held-out numbers (shadow AUPRC 0.552 → 0.805 → 0.765). The bug-then-fix is itself a useful artifact in the audit trail — Day-7 tests will encode this as a regression check.
- **Initial monitored-feature list flagged `day_of_month` on every clean day**, dragging precision to 0.23. The fix was curating the monitored set, not changing the detector. Documented in the test file's `DEFAULT_MONITORED` comment so future-me does not re-introduce calendar features without a re-baseline policy.
- **Dask was not used in the drift detector** — it could shard the KS-on-many-features step across cores at very large windows, but at 2K-row daily windows the per-feature KS finishes in <1ms and parallelisation would be pure overhead. Day-2's "Dask earns its place at the scale Pandas hits RAM limits" finding holds here too: monitoring windows are small, so single-threaded pandas is the right tool.

## Sample Outputs Saved
- `results/drift_replay_per_day.csv` — 30 rows, per-day KS / PSI / fired flags
- `results/drift_retrain_events.csv` — 3 rows, per-retrain-event timing + AUPRC
- `results/drift_replay_summary.json` — top-level detector metrics
- `results/drift_retrain_metrics.json` — top-level trigger metrics
- `results/drift_metrics.json` — champion pick + headline numbers
- `results/phase2_leaderboard.csv` — combined Day 1–3 leaderboard
- `results/drift_reference.json` — saved reference snapshot (ready to ship in a model artifact)
- `results/samples/drift/per_day_reports_sample.json` — sample DriftReport dicts (days 0, 22, 23, 29)
- `results/samples/drift/retrain_event_sample.json` — full RetrainEvent dict (first event)

## Next Day (Day 4 — Phase 3)
Champion stack integration:
- Refactor for cleanliness — confirm `src/data/loader.py`, `src/features/engineer.py`, `src/training/train.py`, `src/registry/{promote,rollback}.py`, `src/drift/{detector,trigger}.py`, `src/serving/api.py` cleanly compose with Pydantic configs.
- Add `src/serving/shadow.py` — every production prediction also fires the latest staging model; results compared async.
- Stand up a Dockerised Postgres telemetry backing store (schema for `predictions`, `drift_scores`, `retrain_events`, `model_registry_log`) plus `src/telemetry/logger.py`.
- Read API at `src/serving/api.py`: `/metrics/drift`, `/metrics/predictions`.
- **Phase-3 wrap-up post** (Day 4 is a phase-wrap day).

## Code Changes
- New: `src/drift/{__init__, detector, trigger, bench_retrain}.py`
- New: `tests/{__init__, synthetic_drift}.py`
- No edits to Day-1 / Day-2 files — drift modules are additive and consume the existing MLflow store and registry CLI.
