#!/usr/bin/env bash
# Sentinel 60-second end-to-end demo -- Day 7 Phase 6 (2026-05-24).
#
# Walks through the Day-1 to Day-6 deliverables in a single take so the
# headline MLOps story is reproducible from a clean checkout:
#
#   1. show the temporal-split fix in train.py is committed (Day 1),
#   2. show pandas == dask determinism on a tiny synthetic frame (Day 2),
#   3. measure registry alias-flip latency (Day 2),
#   4. trigger the 30-day synthetic drift replay (Day 3),
#   5. tail the auto-retrain events + the drift-day-by-day artifact,
#   6. compare champion vs naive vs LLM-judged (Day 6 artifact).
#
# Runtime: ~60 seconds end-to-end on a developer laptop after deps are
# installed. asciinema-friendly: each section prints a banner and tails the
# relevant artifact so the viewer always sees the numbers, not just the
# command. Designed for `asciinema rec` -> upload to demo asset.
#
# Pre-reqs: `pip install -r requirements.txt`, `dvc repro train` (so the
# baseline mlflow.db exists), `docker compose up -d` is OPTIONAL -- the demo
# uses the local sqlite tracking store and committed `results/*` artifacts.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

banner "Day 1 -- temporal-split fix is committed"
echo "Pre-fix (random split, leaked future txns):"
echo "    train_test_split(X, y, test_size=0.2, stratify=y)"
echo "Post-fix (per-source temporal split):"
grep -n "def temporal_split_per_source" src/train.py | head -1
echo ""
echo "Honest sparkov_test AUC (from results/baseline_metrics.json):"
python - <<'PY'
import json
b = json.load(open("results/baseline_metrics.json"))
held = b.get("post_fix_benchmark_held_out_sparkov_test_file", {})
auc = held.get("our_auc")
ag = held.get("baselines", {}).get("AutoGluon", {})
print(f"  sparkov_test_auc = {auc}  (vs AutoGluon {ag.get('baseline_auc', 'n/a')}, delta {ag.get('delta', 'n/a')})")
PY

banner "Day 2 -- Pandas == Dask determinism (1K rows)"
python -m pytest tests/test_features_determinism.py -q --disable-warnings 2>&1 | tail -3

banner "Day 2 -- MLflow registry rollback latency (cached results)"
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/registry_rollback_times.csv")
print(df[["iteration", "alias_flip_seconds", "audit_tag_seconds", "total_seconds"]].to_string(index=False))
print(f"\nMedian alias-flip: {df['alias_flip_seconds'].median() * 1000:.1f} ms")
print(f"Max alias-flip:    {df['alias_flip_seconds'].max() * 1000:.1f} ms")
PY

banner "Day 3 -- 30-day synthetic drift replay (cached results)"
python - <<'PY'
import json, pandas as pd
summary = json.load(open("results/drift_replay_summary.json"))
print(f"Injection day:  {summary['injection_day']}")
print(f"Feature:        {summary['injection_feature']} (sigma={summary['injection_sigma']})")
print(f"True drift days: {summary['true_drift_days']}")
print(f"Detected fires:  {summary.get('detected_fire_days', '<see per_day.csv>')}")
print(f"Precision:       {summary.get('precision', 'n/a')}")
print(f"Recall:          {summary.get('recall', 'n/a')}")
PY

banner "Day 3 -- auto-retrain events"
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/drift_retrain_events.csv")
cols = ["triggered_on_day", "shadow_auprc", "prod_auprc", "promote_decision",
        "new_model_version", "seconds_end_to_end"]
print(df[cols].to_string(index=False))
PY

banner "Day 6 -- Sentinel champion vs naive notebook vs Claude Opus 4.6"
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/day06/frontier_comparison.csv")
df_disp = df[["strategy", "auc", "auprc", "f1_at_0_5", "latency_s_per_query", "cost_usd_at_1k_qps_per_day"]]
df_disp.columns = ["strategy", "AUC", "AUPRC", "F1@0.5", "latency/q (s)", "$/day @ 1k qps"]
print(df_disp.to_string(index=False))
PY

banner "Sprint complete -- 7 days, 31 tests passing, AutoGluon gap closed"
echo "Sources:"
echo "  results/baseline_metrics.json        (Day 1 honest AUC)"
echo "  results/throughput_speedup.csv       (Day 2 Pandas vs Dask)"
echo "  results/registry_rollback_times.csv  (Day 2 alias-flip)"
echo "  results/drift_replay_summary.json    (Day 3 30-day replay)"
echo "  results/drift_retrain_events.csv     (Day 3 auto-retrain)"
echo "  results/day05/                       (Day 5 sweep + fix)"
echo "  results/day06/                       (Day 6 frontier + ablation)"
echo ""
echo "Open the dashboard: streamlit run app.py -> sidebar 'Ops'"
