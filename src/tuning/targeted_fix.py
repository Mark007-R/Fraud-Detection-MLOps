"""Day 5 Phase 4 — Targeted fix based on failure-mode analysis.

Findings from src.analysis.failure_modes on sparkov_test.csv (OOT):
- Overall recall@0.5 = 0.043 (catches only 92 of 2145 frauds)
- Per-category AUC is 0.88-0.99 across nearly every merchant category, yet
  recall@0.5 collapses to 0.00 for 11/14 categories
- Diagnosis: AUC says the RANKING is fine; the failure is THRESHOLD
  CALIBRATION. Distribution shift over the +6-month window shifted the
  fraud-probability mass downward, so 0.5 is too high a cutoff
- Secondary: the model also over-weights paysim noise during training
  because paysim is 83% of the row count

Two-pronged targeted fix layered on top of the Day-5 Optuna best params:
  (a) Source-balanced sample weights — set per-row weight so each source
      contributes equal aggregate weight. Paysim no longer drowns sparkov.
  (b) Threshold calibration — fit the OPTIMAL decision threshold on the
      sparkov SLICE of the temporal test set (a held-out in-distribution
      sparkov window), then apply it to sparkov_test.csv (OOT).
      Maximises F1; reports precision/recall/F1 at both 0.5 and τ*.

These do NOT add new features or new architectures — they reuse the same
XGBoost + same feature set. The story is: failure-analysis-driven fixes
on the existing model recover most of the OOT calibration loss.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.benchmark_fdb import _sparkov_to_common_schema
from src.config import PROJECT_ROOT
from src.preprocess import align_feature_columns, engineer_features_pandas
from src.train import temporal_split_per_source


RANDOM_STATE = 42
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SPARKOV_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "sparkov_test.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "day05"
MODEL_OUT = PROJECT_ROOT / "models" / "fraud_model_tuned_fixed.pkl"


def _setup_mlflow() -> None:
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("sentinel-day05-targeted-fix")


def _compute_sample_weights(train_df: pd.DataFrame, mode: str) -> np.ndarray:
    """Return per-row sample weights.

    mode='source_balanced': each source's rows sum to the same total weight.
        Effectively up-weights sparkov rows ~5x because paysim is ~5x larger.
    mode='none': uniform weights (no-op).
    """
    n = len(train_df)
    if mode == "none":
        return np.ones(n, dtype="float32")

    if mode == "source_balanced":
        weights = np.ones(n, dtype="float32")
        source_cols = [c for c in train_df.columns if c.startswith("source_")]
        n_sources = len(source_cols)
        for src in source_cols:
            mask = (train_df[src] == 1).to_numpy()
            n_src = int(mask.sum())
            if n_src == 0:
                continue
            # Each source contributes n/n_sources total weight.
            per_row_weight = (n / n_sources) / n_src
            weights[mask] = per_row_weight
        return weights

    raise ValueError(f"Unknown sample-weight mode: {mode}")


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises F1 on the given scores. Returns (tau, f1)."""
    precs, recs, thrs = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one fewer threshold than P/R points.
    if len(thrs) == 0:
        return 0.5, 0.0
    f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thrs[best_idx]), float(f1s[best_idx])


def main() -> None:
    _setup_mlflow()

    # Load Optuna best params (the Day-5 hyperparameter baseline).
    with open(RESULTS_DIR / "optuna_best_params.json", "r", encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]
    print(f"[Fix] Using Optuna best params: {best_params}")

    print(f"[Fix] Loading {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    train_df, test_df = temporal_split_per_source(df, test_size=0.2)
    feature_cols = [c for c in df.columns if c not in ("is_fraud", "txn_timestamp")]

    X_train = train_df[feature_cols].astype("float32")
    y_train = train_df["is_fraud"].astype(int).to_numpy()
    print(f"[Fix] Train rows: {len(X_train):,}")

    # In-distribution sparkov slice — used for threshold calibration.
    sparkov_test_indist = test_df[test_df["source_sparkov"] == 1]
    X_indist = sparkov_test_indist[feature_cols].astype("float32")
    y_indist = sparkov_test_indist["is_fraud"].astype(int).to_numpy()

    # Load thresholds sidecar.
    thresholds_path = FEATURES_PATH.parent / "feature_thresholds.json"
    feat_thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            feat_thresholds = json.load(f)

    # Load OOT sparkov_test.csv features.
    print(f"[Fix] Loading OOT {SPARKOV_TEST_PATH}")
    raw_oot = pd.read_csv(SPARKOV_TEST_PATH)
    common_oot = _sparkov_to_common_schema(raw_oot)
    y_oot = common_oot["is_fraud"].astype(int).to_numpy()
    engineered_oot = engineer_features_pandas(common_oot, thresholds=feat_thresholds or None)
    if "is_fraud" in engineered_oot.columns:
        engineered_oot = engineered_oot.drop(columns=["is_fraud"])
    X_oot = align_feature_columns(engineered_oot, feature_cols).astype("float32")

    # Train with source-balanced sample weights.
    weights = _compute_sample_weights(train_df, mode="source_balanced")
    print(
        f"[Fix] Sample-weight mode: source_balanced -- "
        f"paysim row weight={weights[train_df['source_paysim'].to_numpy()==1].mean():.4f}, "
        f"sparkov row weight={weights[train_df['source_sparkov'].to_numpy()==1].mean():.4f}"
    )

    with mlflow.start_run(run_name="day05_targeted_fix_v1") as run:
        print(f"[Fix] MLflow run_id={run.info.run_id}")
        mlflow.log_params(
            {
                **best_params,
                "sample_weight_mode": "source_balanced",
                "training_scope": "full_temporal_split_train",
                "threshold_calibration": "f1_on_sparkov_indist",
                "fix_layer": "Optuna best params + source-balanced weights + tau* threshold",
            }
        )

        t0 = time.time()
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        clf.fit(X_train, y_train, sample_weight=weights)
        fit_seconds = time.time() - t0
        print(f"[Fix] Trained in {fit_seconds:.1f}s")

        # Step 1: in-distribution sparkov slice for threshold tuning.
        p_indist = clf.predict_proba(X_indist)[:, 1]
        tau_star, f1_at_tau = _find_best_threshold(y_indist, p_indist)
        print(f"[Fix] Tuned threshold tau*={tau_star:.4f} (in-dist F1={f1_at_tau:.4f})")

        # Step 2: score OOT sparkov_test.csv with BOTH thresholds.
        p_oot = clf.predict_proba(X_oot)[:, 1]
        yp_05 = (p_oot >= 0.5).astype(int)
        yp_tau = (p_oot >= tau_star).astype(int)

        auc_oot = float(roc_auc_score(y_oot, p_oot))
        ap_oot = float(average_precision_score(y_oot, p_oot))

        def _metric_dict(yt, yp):
            return {
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "confusion_matrix": confusion_matrix(yt, yp).tolist(),
            }

        at_05 = _metric_dict(y_oot, yp_05)
        at_tau = _metric_dict(y_oot, yp_tau)

        # In-distribution per-source AUC sanity (paysim shouldn't regress).
        in_dist = {}
        for src in ("source_paysim", "source_sparkov"):
            sub = test_df[test_df[src] == 1]
            if len(sub) == 0:
                continue
            yt = sub["is_fraud"].astype(int).to_numpy()
            X = sub[feature_cols].astype("float32")
            yp = clf.predict_proba(X)[:, 1]
            in_dist[src] = {
                "rows": int(len(yt)),
                "auc": float(roc_auc_score(yt, yp)) if len(set(yt)) > 1 else 0.0,
                "average_precision": float(average_precision_score(yt, yp)) if len(set(yt)) > 1 else 0.0,
            }

        mlflow.log_metrics(
            {
                "test_oot_sparkov_auc": auc_oot,
                "test_oot_sparkov_ap": ap_oot,
                "recall_at_0.5": at_05["recall"],
                "recall_at_tau_star": at_tau["recall"],
                "f1_at_0.5": at_05["f1"],
                "f1_at_tau_star": at_tau["f1"],
                "tau_star": tau_star,
                "in_dist_paysim_auc": in_dist.get("source_paysim", {}).get("auc", 0.0),
                "in_dist_sparkov_auc": in_dist.get("source_sparkov", {}).get("auc", 0.0),
                "delta_vs_day1_honest": auc_oot - 0.794896,
                "delta_vs_autogluon": auc_oot - 0.952,
            }
        )

        # Persist the fixed artifact + headline payload.
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": clf,
                "feature_columns": list(feature_cols),
                "target_column": "is_fraud",
                "feature_thresholds": feat_thresholds,
                "split_strategy": "temporal_per_source",
                "tuned_params": best_params,
                "sample_weight_mode": "source_balanced",
                "decision_threshold": tau_star,
            },
            MODEL_OUT,
        )

        try:
            mlflow.xgboost.log_model(clf, artifact_path="xgboost_model_fixed")
        except Exception as exc:
            print(f"[Fix] mlflow.xgboost.log_model failed (non-fatal): {exc}")
        mlflow.log_artifact(str(MODEL_OUT))

        payload = {
            "model_artifact": str(MODEL_OUT.relative_to(PROJECT_ROOT)),
            "best_params": best_params,
            "sample_weight_mode": "source_balanced",
            "decision_threshold_tau_star": tau_star,
            "in_distribution_per_source": in_dist,
            "out_of_time_sparkov_test_csv": {
                "rows": int(len(y_oot)),
                "fraud_rate": float(y_oot.mean()),
                "auc": auc_oot,
                "average_precision": ap_oot,
                "at_threshold_0.5": at_05,
                "at_threshold_tau_star": at_tau,
            },
            "comparison": {
                "day1_honest_baseline_oot_auc": 0.794896,
                "day5_tuned_only_oot_auc": 0.915380,
                "day5_tuned_plus_fix_oot_auc": auc_oot,
                "autogluon_baseline_oot_auc": 0.952,
                "lift_vs_tuned_only": auc_oot - 0.915380,
                "lift_vs_day1": auc_oot - 0.794896,
                "delta_vs_autogluon": auc_oot - 0.952,
            },
            "recall_lift_at_tau_star": {
                "tuned_only_recall_at_0.5": 0.042890,
                "fixed_recall_at_0.5": at_05["recall"],
                "fixed_recall_at_tau_star": at_tau["recall"],
                "recall_uplift_pct_points": (at_tau["recall"] - 0.042890) * 100,
            },
        }
        out_path = RESULTS_DIR / "targeted_fix_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Fix] Saved -> {out_path}")

        print(
            f"\n[Fix] HEADLINE: OOT sparkov_test AUC = {auc_oot:.4f}, "
            f"recall@tau*={at_tau['recall']:.4f} (was {0.042890:.4f}), "
            f"f1@tau*={at_tau['f1']:.4f}, tau*={tau_star:.4f}"
        )


if __name__ == "__main__":
    main()
