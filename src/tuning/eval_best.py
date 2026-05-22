"""Day 5 Phase 4: Train Optuna's best params on FULL train data and evaluate.

Mirrors src/train.py's pipeline (per-source temporal split, MLflow tracking,
joblib artifact format) but uses the best hyperparameters from
src.tuning.optuna_sweep. Scores on the held-out sparkov_test.csv via the
same path src/benchmark_fdb.py uses, so the AUC is apples-to-apples with
the Day-1 honest baseline of 0.7949.

Outputs:
- models/fraud_model_tuned.pkl      — full artifact (model + thresholds)
- results/day05/tuned_eval.json     — headline metrics
- results/day05/oot_predictions.parquet — per-row predictions + raw fields
  for failure mode analysis in src.analysis.failure_modes
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
MODEL_OUT = PROJECT_ROOT / "models" / "fraud_model_tuned.pkl"


def _setup_mlflow() -> None:
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("sentinel-day05-tuned-final")


def load_best_params(label: str = "optuna_best_params.json") -> dict:
    path = RESULTS_DIR / label
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"[Eval] Loaded best params from {path}:")
    for k, v in payload["best_params"].items():
        print(f"  - {k}: {v}")
    return payload["best_params"]


def train_full(best_params: dict) -> tuple[XGBClassifier, list[str], dict, pd.DataFrame, pd.DataFrame]:
    """Train XGB with best params on FULL temporal-split train set."""
    print(f"[Eval] Loading {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    train_df, test_df = temporal_split_per_source(df, test_size=0.2)

    feature_cols = [c for c in df.columns if c not in ("is_fraud", "txn_timestamp")]
    X_train = train_df[feature_cols].astype("float32")
    y_train = train_df["is_fraud"].astype(int).to_numpy()
    X_test = test_df[feature_cols].astype("float32")
    y_test = test_df["is_fraud"].astype(int).to_numpy()

    print(f"[Eval] Train: {len(X_train):,}  Test (combined temporal): {len(X_test):,}")

    thresholds_path = FEATURES_PATH.parent / "feature_thresholds.json"
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    t0 = time.time()
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    clf.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    print(f"[Eval] Trained on full data in {fit_seconds:.1f}s")

    return clf, feature_cols, thresholds, test_df, pd.DataFrame({"y_test": y_test})


def score_oot(clf: XGBClassifier, feature_cols: list[str], thresholds: dict) -> dict:
    """Score held-out sparkov_test.csv. Save per-row predictions for failure analysis."""
    print(f"[Eval] Loading {SPARKOV_TEST_PATH}")
    raw = pd.read_csv(SPARKOV_TEST_PATH)
    common = _sparkov_to_common_schema(raw)
    y_true = common["is_fraud"].astype(int).to_numpy()

    engineered = engineer_features_pandas(common, thresholds=thresholds or None)
    if "is_fraud" in engineered.columns:
        engineered = engineered.drop(columns=["is_fraud"])
    X = align_feature_columns(engineered, feature_cols).astype("float32")

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Save per-row predictions joined with raw txn fields for failure analysis.
    raw["y_true"] = y_true
    raw["y_prob"] = y_prob
    raw["y_pred"] = y_pred
    oot_path = RESULTS_DIR / "oot_predictions.parquet"
    try:
        raw.to_parquet(oot_path, index=False)
    except Exception:
        oot_path = RESULTS_DIR / "oot_predictions.csv"
        raw.to_csv(oot_path, index=False)
    print(f"[Eval] Saved OOT predictions -> {oot_path}")

    return {
        "rows": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "auc": auc,
        "average_precision": ap,
        "accuracy": acc,
        "precision_at_0.5": prec,
        "recall_at_0.5": rec,
        "f1_at_0.5": f1,
        "confusion_matrix": cm,
        "predictions_path": str(oot_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    _setup_mlflow()
    label = os.environ.get("SENTINEL_BEST_PARAMS_LABEL", "optuna_best_params.json")
    out_name = os.environ.get("SENTINEL_EVAL_OUT", "tuned_eval.json")
    model_out_name = os.environ.get("SENTINEL_MODEL_OUT", "fraud_model_tuned.pkl")
    run_name = os.environ.get("SENTINEL_RUN_NAME", "day05_tuned_full_train")

    best_params = load_best_params(label)
    with mlflow.start_run(run_name=run_name) as run:
        print(f"[Eval] MLflow run_id={run.info.run_id}")
        mlflow.log_params({**best_params, "training_scope": "full_temporal_split_train"})

        clf, feature_cols, thresholds, test_df, _ = train_full(best_params)

        # In-distribution per-source metrics (sanity).
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
                "fraud_rate": float(yt.mean()),
                "auc": float(roc_auc_score(yt, yp)) if len(set(yt)) > 1 else 0.0,
                "average_precision": float(average_precision_score(yt, yp)) if len(set(yt)) > 1 else 0.0,
            }

        oot = score_oot(clf, feature_cols, thresholds)

        # Persist artifact in same format as src/train.py so existing inference paths work.
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        out_path = MODEL_OUT.parent / model_out_name
        joblib.dump(
            {
                "model": clf,
                "feature_columns": list(feature_cols),
                "target_column": "is_fraud",
                "feature_thresholds": thresholds,
                "split_strategy": "temporal_per_source",
                "tuned_params": best_params,
            },
            out_path,
        )
        print(f"[Eval] Saved tuned artifact -> {out_path}")

        mlflow.log_metrics(
            {
                "test_oot_sparkov_auc": oot["auc"],
                "test_oot_sparkov_ap": oot["average_precision"],
                "delta_vs_day1_honest": oot["auc"] - 0.794896,
                "delta_vs_autogluon": oot["auc"] - 0.952,
                **{f"in_dist_{k}_auc": v["auc"] for k, v in in_dist.items()},
            }
        )

        try:
            mlflow.xgboost.log_model(clf, artifact_path="xgboost_model")
        except Exception as exc:
            print(f"[Eval] mlflow.xgboost.log_model failed (non-fatal): {exc}")
        mlflow.log_artifact(str(out_path))

        payload = {
            "model_artifact": str(out_path.relative_to(PROJECT_ROOT)),
            "best_params": best_params,
            "in_distribution_per_source": in_dist,
            "out_of_time_sparkov_test_csv": oot,
            "comparison": {
                "day1_honest_baseline_oot_auc": 0.794896,
                "autogluon_baseline_oot_auc": 0.952,
                "delta_vs_day1_honest": oot["auc"] - 0.794896,
                "delta_vs_autogluon": oot["auc"] - 0.952,
            },
        }
        eval_path = RESULTS_DIR / out_name
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Eval] Saved metrics -> {eval_path}")

        print(
            f"\n[Eval] HEADLINE: OOT sparkov_test AUC = {oot['auc']:.4f} "
            f"(delta vs Day-1 0.7949 = {oot['auc'] - 0.794896:+.4f}, "
            f"delta vs AutoGluon 0.952 = {oot['auc'] - 0.952:+.4f})"
        )


if __name__ == "__main__":
    main()
