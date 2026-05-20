"""End-to-end drift -> auto-retrain bench (Day 3 Phase 2b).

Replays the same 30-day synthetic-drift stream as ``tests/synthetic_drift.py``,
but on each day it (a) scores the production model, (b) feeds the fire/no-fire
signal into the ``TriggerState`` debouncer, (c) when the debouncer trips,
retrains on the drifted window, runs shadow eval on the most recent day, and
conditionally promotes via the Day-2 MLflow registry CLI.

Outputs
-------
- ``results/drift_retrain_events.csv`` — one row per retrain event
- ``results/drift_retrain_metrics.json`` — summary numbers (end-to-end seconds,
  promote count, shadow AUPRC vs prod, alias-flip latency)
- ``results/samples/drift/retrain_event_sample.json`` — full event dict for the
  first event (audit trail)
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT
from src.drift.detector import DriftDetector, fit_reference
from src.drift.trigger import run_drift_retrain_simulation
from tests.synthetic_drift import (
    DEFAULT_MONITORED,
    INJECTION_DAY,
    INJECTION_FEATURE,
    INJECTION_SIGMA,
    inject_amount_drift,
    load_sparkov_train_test_folds,
    slice_first_n_days,
)


MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = RESULTS_DIR / "samples" / "drift"


def _build_daily_windows(
    test_df: pd.DataFrame,
    *,
    n_days: int,
    injection_day: int,
    injection_sigma: float,
    ref_amount_std: float,
    feature_cols: Iterable[str],
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """Return (daily_X, daily_y) sequences for the trigger replay."""
    feature_cols = list(feature_cols)
    stream = slice_first_n_days(test_df, n_days=n_days)
    daily_X, daily_y = [], []
    for d in range(n_days):
        day_df = stream[stream["_day"] == d].drop(columns=["_day"])
        if len(day_df) == 0:
            daily_X.append(pd.DataFrame(columns=feature_cols))
            daily_y.append(pd.Series(dtype=int))
            continue
        if d >= injection_day:
            day_df = inject_amount_drift(
                day_df, ref_amount_std=ref_amount_std, n_sigma=injection_sigma
            )
        daily_X.append(day_df[feature_cols].reset_index(drop=True))
        daily_y.append(day_df["is_fraud"].astype(int).reset_index(drop=True))
    return daily_X, daily_y


def main(
    *,
    n_days: int = 30,
    n_consecutive_days: int = 2,
    promote_tolerance: float = 0.01,
) -> dict:
    artifact = joblib.load(MODEL_PATH)
    feature_cols: list[str] = list(artifact["feature_columns"])

    train_df, test_df = load_sparkov_train_test_folds()

    rng = np.random.default_rng(42)
    ref_idx = rng.choice(len(train_df), size=min(50_000, len(train_df)), replace=False)
    ref_X = train_df.iloc[ref_idx][feature_cols].reset_index(drop=True)
    ref_proba = artifact["model"].predict_proba(ref_X)[:, 1]
    monitored = [c for c in DEFAULT_MONITORED if c in feature_cols]
    reference = fit_reference(ref_X, monitored, proba=ref_proba, max_reference_sample=20_000)
    ref_amount_std = float(np.std(reference["feature_samples"][INJECTION_FEATURE]))

    daily_X, daily_y = _build_daily_windows(
        test_df,
        n_days=n_days,
        injection_day=INJECTION_DAY,
        injection_sigma=INJECTION_SIGMA,
        ref_amount_std=ref_amount_std,
        feature_cols=feature_cols,
    )

    detector = DriftDetector(
        reference,
        ks_pvalue_threshold=0.01,
        ks_stat_threshold=0.15,
        psi_threshold=0.25,
        min_flag_features=1,
        monitored_features=monitored,
    )

    print(
        f"[Bench] Replaying {n_days} days, injection_day={INJECTION_DAY}, "
        f"sigma={INJECTION_SIGMA}, debounce N={n_consecutive_days}"
    )
    t0 = time.perf_counter()
    state, events = run_drift_retrain_simulation(
        daily_X=daily_X,
        daily_y=daily_y,
        detector=detector,
        prod_model_path=MODEL_PATH,
        n_consecutive_days=n_consecutive_days,
        promote_tolerance=promote_tolerance,
    )
    wall_seconds = time.perf_counter() - t0

    if not events:
        raise RuntimeError(
            "[Bench] No retrain events fired — check detector or debounce parameters."
        )

    # CSV per-event.
    rows = [e.to_dict() for e in events]
    df = pd.DataFrame(rows)
    # train_window_days is a list — stringify so CSV is round-trippable.
    df["train_window_days"] = df["train_window_days"].apply(lambda lst: ";".join(map(str, lst)))
    out_csv = RESULTS_DIR / "drift_retrain_events.csv"
    df.to_csv(out_csv, index=False)

    promoted = [e for e in events if e.promote_decision]
    not_promoted = [e for e in events if not e.promote_decision]

    e2e = [e.seconds_end_to_end for e in events]
    fits = [e.seconds_train for e in events]
    shadows = [e.seconds_shadow_eval for e in events]
    regs = [e.seconds_register_and_alias for e in events]

    detection_latency_days = events[0].triggered_on_day - INJECTION_DAY  # debounce cost
    summary = {
        "captured_on": pd.Timestamp.utcnow().isoformat(),
        "day": 3,
        "phase": "2b — drift -> auto-retrain bench",
        "n_days": n_days,
        "injection_day": INJECTION_DAY,
        "injection_sigma": INJECTION_SIGMA,
        "n_consecutive_days": n_consecutive_days,
        "promote_tolerance": promote_tolerance,
        "n_retrain_events": len(events),
        "n_promoted": len(promoted),
        "n_not_promoted": len(not_promoted),
        "first_trigger_day": events[0].triggered_on_day,
        "detection_latency_days": detection_latency_days,
        "fire_history": state.history,
        "end_to_end_seconds": {
            "min": round(min(e2e), 4),
            "median": round(statistics.median(e2e), 4),
            "max": round(max(e2e), 4),
            "mean": round(statistics.mean(e2e), 4),
        },
        "fit_seconds": {
            "min": round(min(fits), 4),
            "median": round(statistics.median(fits), 4),
            "max": round(max(fits), 4),
        },
        "shadow_eval_seconds": {
            "min": round(min(shadows), 4),
            "median": round(statistics.median(shadows), 4),
            "max": round(max(shadows), 4),
        },
        "register_alias_seconds": {
            "min": round(min(regs), 4),
            "median": round(statistics.median(regs), 4),
            "max": round(max(regs), 4),
        },
        "shadow_auprc_first_event": events[0].shadow_auprc,
        "prod_auprc_first_event": events[0].prod_auprc,
        "shadow_minus_prod_auprc": round(events[0].shadow_auprc - events[0].prod_auprc, 4),
        "wall_seconds_total": round(wall_seconds, 2),
    }

    out_json = RESULTS_DIR / "drift_retrain_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLES_DIR / "retrain_event_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(events[0].to_dict(), f, indent=2)

    print(f"[Bench] events={len(events)} promoted={len(promoted)} wall={wall_seconds:.2f}s")
    print(f"[Bench] end-to-end median {summary['end_to_end_seconds']['median']}s")
    print(f"[Bench] Wrote {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"[Bench] Wrote {out_json.relative_to(PROJECT_ROOT)}")

    return summary


if __name__ == "__main__":
    main()
