# Sentinel — Train/Test Split Rationale

## Decision

Sentinel uses a **per-source temporal split**: for each source (paysim,
sparkov), rows are sorted ascending by `txn_timestamp` and the last
`test_size` (default 20%) fraction becomes the test set. The two source
test sets are concatenated to form the combined evaluation set.

## Why per-source temporal — not random, not stratified, not k-fold

### Random / stratified random (the prior approach) leaks the future

Random splits assume rows are exchangeable. Fraud transactions are not:

- Reissued cards land in subsequent months, not the same hour.
- Compromised merchants tend to cluster in time.
- The mix of `transaction_type` shifts over a 6-month window
  (e.g., online/POS share changes in 2020 due to COVID).
- Velocity-based features (per-card spending rate, time-since-last-txn)
  are time-correlated by construction.

With `train_test_split(..., stratify=y)`, the model sees a March 2020
fraud burst in training and is then asked to predict on March 2020 frauds
in test that share temporally adjacent context (same merchants, same
compromised regions). It can effectively interpolate within the period.
Production scoring at runtime cannot — it sees strictly-newer txns.

### Per-source temporal is closer to production reality

In production, a model trained yesterday scores transactions today. The
strictly-future evaluation regime matches that. Per-source preserves each
dataset's own chronology:

- **paysim**: `step` column = simulated hour. We map to `step * 3600` to
  produce a sortable per-source timestamp.
- **sparkov**: `trans_date_trans_time` parsed to unix seconds. (Also
  available pre-parsed as `unix_time`.)

Cross-source merging on a single timeline would be meaningless (paysim's
synthetic clock isn't comparable to sparkov's calendar dates), so we
split each source independently and concat.

### Why not k-fold time-series CV

A single forward chronological split is the cleanest match for the
benchmark setup (compare to AutoGluon's published number on a fixed
held-out sparkov split). Time-series CV (TimeSeriesSplit) is the right
tool for **hyperparameter tuning** — that's Day-5's job, not Day-1.

## Combine stage: chronological sort, not shuffle

`src/combine_datasets.py` previously ran
`combined.sample(frac=1.0, random_state=42)` after concat, destroying
all time order. That's incompatible with a temporal split. Day-1 fix:
replace the shuffle with `sort_values(["source", "txn_timestamp"])` so
each source block in `combined_transactions.csv` is monotonic in time.

## Benchmark file: sparkov_test.csv, not sparkov.csv

The held-out sparkov benchmark file (`data/raw/sparkov_test.csv`,
2020-06-21 → 2020-12-31) is strictly after the training file's date
range (2019-01-01 → 2020-06-21). Using it for `benchmark_fdb.py` gives
an honest AutoGluon-comparable AUC. The prior code fell back to
`sparkov.csv` itself (= the training file) when the FDB Python package
wasn't installed, which is the leak that produced the 0.921 claim.

## Honest numbers (post-fix, 2026-05-18)

- **Combined temporal test AUC**: 0.999 (dominated by paysim balance signals)
- **Sparkov-only temporal test AUC** (final 20% of sparkov.csv): 0.997
- **Sparkov-only held-out file AUC** (sparkov_test.csv): **0.795**

The 0.20 AUC drop from in-period test to truly-out-of-period file is
**real distribution shift**, not modeling failure. That delta is the gap
that Day-3's drift-triggered retrain has to cover.
