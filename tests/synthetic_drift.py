"""Synthetic 30-day drift replay against the Day-1 production model.

Day 3 Phase 2b (2026-05-20). The drift detector built in
``src/drift/detector.py`` is only as useful as its precision/recall against a
known drift event. This script:

    1. Loads ``data/processed/features.csv`` and takes the sparkov-only
       temporal test fold (the same 20% slice that ``src/train.py`` derives).
    2. Slices the first 30 contiguous calendar days as the monitoring stream.
    3. Fits the drift detector reference on the sparkov-only TRAIN fold
       (the same 80% the production model saw).
    4. Replays the stream day-by-day. Days 0–22 are passed through unmodified;
       days 23–29 are injected with a +2σ shift in ``amount`` (and the
       cascading log / z-score columns engineered from it).
    5. Asserts the detector fires only on days 23+, and reports per-day KS
       statistics + PSI on probability for every day. Computes precision /
       recall of detection vs the true drift window {23, 24, ..., 29}.

The injection is sized to the *engineered* amount column on the reference
distribution so the magnitude is reproducible across runs, not relative to
the daily window. This is the canonical "data-pipeline drift, not concept
drift" failure mode that motivates Sentinel's Day-3 detector — a benign
upstream change (currency conversion, sensor recalibration, a feed switching
from cents to dollars) bumps a single feature distribution by ~2σ.

Run as a script:
    python -m tests.synthetic_drift

Run as a pytest:
    pytest tests/synthetic_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.drift.detector import DriftDetector, fit_reference


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = RESULTS_DIR / "samples" / "drift"
REFERENCE_PATH = RESULTS_DIR / "drift_reference.json"


# Continuous features worth monitoring — skip one-hot category columns where
# KS is uninformative. These are the columns whose marginal distribution can
# realistically drift in serving (amount-related, balance-related,
# time-of-day signals).
DEFAULT_MONITORED = [
    "amount",
    "tx_amount_log",
    "amount_zscore",
    "balance_change_orig",
    "balance_ratio",
    "balance_change_abs",
    "balance_change_log",
    "hour_of_day",
    # Intentionally exclude `day_of_month` — it mechanically shifts every calendar
    # day in the replay (the test fold steps through Mar 6 -> Apr 5) so it KS-fires
    # on every window without indicating real data drift. The detector code stays
    # general; the monitoring set is the policy knob.
]

INJECTION_DAY = 23
INJECTION_FEATURE = "amount"
INJECTION_SIGMA = 2.0


# ----- Data preparation ------------------------------------------------------


def load_sparkov_train_test_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the Day-1 sparkov-only train/test fold from features.csv."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"[Drift] {FEATURES_PATH} missing — run `python -m src.preprocess` first"
        )
    pdf = pd.read_csv(FEATURES_PATH)
    sp = pdf[pdf["source_sparkov"] == 1].sort_values("txn_timestamp", kind="mergesort")
    sp = sp.reset_index(drop=True)
    n_test = int(np.ceil(len(sp) * 0.2))
    train_df = sp.iloc[:-n_test].reset_index(drop=True)
    test_df = sp.iloc[-n_test:].reset_index(drop=True)
    return train_df, test_df


def slice_first_n_days(test_df: pd.DataFrame, n_days: int = 30) -> pd.DataFrame:
    """Bucket the test fold by calendar day from its first row and keep the first ``n_days``."""
    ts0 = int(test_df["txn_timestamp"].iloc[0])
    days = (test_df["txn_timestamp"].astype("int64") - ts0) // 86400
    out = test_df.assign(_day=days)
    return out[out["_day"] < n_days].reset_index(drop=True)


def inject_amount_drift(
    daily_df: pd.DataFrame, ref_amount_std: float, *, n_sigma: float = INJECTION_SIGMA
) -> pd.DataFrame:
    """Shift the ``amount``-derived columns by ``n_sigma * ref_amount_std``.

    The shift is applied to every column engineered FROM ``amount`` so the
    drift is internally consistent — otherwise the KS test on derived
    columns would lag, and PSI on model probability would not move.
    """
    shift = float(n_sigma * ref_amount_std)
    out = daily_df.copy()
    if "amount" in out.columns:
        out["amount"] = out["amount"].astype("float64") + shift
    if "tx_amount_log" in out.columns:
        out["tx_amount_log"] = np.log1p(np.clip(out["amount"], 0.0, None))
    if "amount_zscore" in out.columns:
        # Use the same ref-std for the rescale so values stay comparable.
        z_shift = shift / max(ref_amount_std, 1e-6)
        out["amount_zscore"] = out["amount_zscore"].astype("float64") + z_shift
    if "balance_change_abs" in out.columns:
        out["balance_change_abs"] = out["balance_change_abs"].astype("float64") + shift
    if "balance_change_log" in out.columns:
        out["balance_change_log"] = np.log1p(np.clip(out["balance_change_abs"], 0.0, None))
    return out


# ----- The replay -----------------------------------------------------------


def replay(
    *,
    n_days: int = 30,
    injection_day: int = INJECTION_DAY,
    injection_feature: str = INJECTION_FEATURE,
    injection_sigma: float = INJECTION_SIGMA,
    monitored_features: Iterable[str] = DEFAULT_MONITORED,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the 30-day replay and return a results dictionary."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"[Drift] {MODEL_PATH} missing — run `python -m src.train` first"
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols: list[str] = list(artifact["feature_columns"])

    train_df, test_df = load_sparkov_train_test_folds()
    if verbose:
        print(
            f"[Drift] Sparkov fold sizes: train={len(train_df)}, test={len(test_df)}"
        )

    # Build the reference snapshot from the train fold + train predictions.
    train_X = train_df[feature_cols]
    rng = np.random.default_rng(42)
    # Cap reference proba sample to avoid scoring 1M rows on Windows CPU.
    ref_idx = rng.choice(len(train_X), size=min(50_000, len(train_X)), replace=False)
    ref_X = train_X.iloc[ref_idx].reset_index(drop=True)
    ref_proba = model.predict_proba(ref_X)[:, 1]

    monitored_features = [c for c in monitored_features if c in feature_cols]
    reference = fit_reference(
        ref_X, monitored_features, proba=ref_proba, max_reference_sample=20_000
    )
    ref_amount_std = float(np.std(reference["feature_samples"].get(injection_feature, [1.0])))
    if verbose:
        print(
            f"[Drift] Reference std({injection_feature})={ref_amount_std:.4f}; "
            f"injection shift = {injection_sigma:.2f}*sigma = {injection_sigma * ref_amount_std:.4f}"
        )

    if save_artifacts:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DriftDetector.save_reference(reference, REFERENCE_PATH)

    detector = DriftDetector(
        reference,
        ks_pvalue_threshold=0.01,
        ks_stat_threshold=0.15,
        psi_threshold=0.25,
        min_flag_features=1,
        monitored_features=monitored_features,
    )

    stream = slice_first_n_days(test_df, n_days=n_days)
    per_day_rows: list[dict] = []
    true_drift = {d for d in range(injection_day, n_days)}
    predicted_drift: set[int] = set()
    sample_records: list[dict] = []

    for d in range(n_days):
        day_df = stream[stream["_day"] == d].drop(columns=["_day"])
        if len(day_df) == 0:
            continue

        if d >= injection_day:
            day_df = inject_amount_drift(
                day_df, ref_amount_std=ref_amount_std, n_sigma=injection_sigma
            )

        day_X = day_df[feature_cols]
        t0 = time.perf_counter()
        day_proba = model.predict_proba(day_X)[:, 1]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        report = detector.score(day_X, day_proba, window_label=f"day_{d:02d}")
        if report.drift_fired:
            predicted_drift.add(d)

        per_day_rows.append(
            {
                "day": d,
                "n_rows": report.n_rows,
                "n_features_flagged": len(report.flagged_features),
                "flagged_features": ";".join(report.flagged_features),
                "ks_amount": round(report.feature_ks_stats.get(injection_feature, 0.0), 4),
                "ks_amount_pvalue": float(
                    np.format_float_scientific(
                        report.feature_ks_pvalues.get(injection_feature, 1.0), precision=3
                    )
                ),
                "ks_amount_zscore": round(report.feature_ks_stats.get("amount_zscore", 0.0), 4),
                "ks_tx_amount_log": round(report.feature_ks_stats.get("tx_amount_log", 0.0), 4),
                "proba_psi": round(report.proba_psi, 4),
                "proba_psi_flag": bool(report.proba_psi_flag),
                "drift_fired": bool(report.drift_fired),
                "fire_reason": report.fire_reason,
                "inference_ms": round(infer_ms, 2),
                "injected": d >= injection_day,
            }
        )

        if d in (0, injection_day - 1, injection_day, n_days - 1):
            sample_records.append(report.to_dict())

    df = pd.DataFrame(per_day_rows)

    # Precision / recall / latency to detection.
    tp = len(predicted_drift & true_drift)
    fp = len(predicted_drift - true_drift)
    fn = len(true_drift - predicted_drift)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    first_fired = min(predicted_drift) if predicted_drift else None
    detection_lag_days = (
        None if first_fired is None else first_fired - injection_day
    )

    summary = {
        "captured_on": pd.Timestamp.utcnow().isoformat(),
        "day": 3,
        "phase": "2b — synthetic drift replay",
        "n_days": n_days,
        "injection_day": injection_day,
        "injection_feature": injection_feature,
        "injection_sigma": injection_sigma,
        "injection_shift": round(injection_sigma * ref_amount_std, 4),
        "monitored_features": monitored_features,
        "detector_params": {
            "ks_pvalue_threshold": 0.01,
            "ks_stat_threshold": 0.15,
            "psi_threshold": 0.25,
        },
        "true_drift_days": sorted(true_drift),
        "predicted_drift_days": sorted(predicted_drift),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "first_fired_day": first_fired,
        "detection_lag_days": detection_lag_days,
        "pre_injection_max_proba_psi": round(
            float(df[df["day"] < injection_day]["proba_psi"].max()), 4
        )
        if len(df[df["day"] < injection_day]) > 0
        else None,
        "post_injection_min_proba_psi": round(
            float(df[df["day"] >= injection_day]["proba_psi"].min()), 4
        )
        if len(df[df["day"] >= injection_day]) > 0
        else None,
    }

    if save_artifacts:
        out_csv = RESULTS_DIR / "drift_replay_per_day.csv"
        df.to_csv(out_csv, index=False)
        out_json = RESULTS_DIR / "drift_replay_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        sample_path = SAMPLES_DIR / "per_day_reports_sample.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_records, f, indent=2)
        if verbose:
            print(f"[Drift] Wrote {out_csv.relative_to(PROJECT_ROOT)}")
            print(f"[Drift] Wrote {out_json.relative_to(PROJECT_ROOT)}")
            print(f"[Drift] Wrote {sample_path.relative_to(PROJECT_ROOT)}")

    if verbose:
        print(
            f"[Drift] Precision={precision:.3f} Recall={recall:.3f} "
            f"first_fired_day={first_fired} lag={detection_lag_days}"
        )

    return {"summary": summary, "per_day": df}


# ----- pytest entry --------------------------------------------------------


def test_detector_fires_only_after_injection() -> None:
    """Detector must fire on day >= 23 and not before."""
    result = replay(verbose=False, save_artifacts=False)
    s = result["summary"]
    assert s["recall"] == 1.0, f"Detector missed at least one drift day: {s}"
    # Detector should not over-fire on the pre-injection window.
    pre_fires = [d for d in s["predicted_drift_days"] if d < INJECTION_DAY]
    assert pre_fires == [], f"Detector false-fired on pre-injection days {pre_fires}"
    # First fire is exactly the injection day — no measurable lag is the target.
    assert s["first_fired_day"] == INJECTION_DAY, (
        f"Detector fired late on day {s['first_fired_day']} instead of {INJECTION_DAY}"
    )


if __name__ == "__main__":
    replay()
