"""Stage 4: Train XGBoost fraud classifier with temporal split + MLflow tracking.

Day 1 Phase 1 fix (2026-05-18):
- The previous implementation used `train_test_split(stratify=y)` which is a
  RANDOM split. For fraud detection on time-series transactions this leaks
  future transaction patterns into training and inflates AUC.
- This implementation uses a per-source temporal split: each source is sorted
  by `txn_timestamp` and the last `test_size` fraction becomes the test set.
- All runs are wrapped in an MLflow run so params, metrics, and the model
  artifact are tracked. Tracking URI defaults to a local sqlite store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_params


def temporal_split_per_source(
    df: pd.DataFrame,
    test_size: float,
    timestamp_col: str = "txn_timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source temporal split.

    For each source (identified by `source_*` one-hot columns), sort rows by
    `timestamp_col` ascending and take the LAST `test_size` fraction as test.
    This preserves per-source chronology and prevents future-data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered feature dataframe. Must contain `txn_timestamp`,
        `is_fraud`, and at least one `source_*` one-hot column.
    test_size : float
        Fraction of each source's rows to allocate to the test set.
    timestamp_col : str
        Column name to sort by.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"[Train] Required column '{timestamp_col}' missing")

    source_cols = [c for c in df.columns if c.startswith("source_")]
    if not source_cols:
        raise KeyError("[Train] No source_* one-hot columns found")

    train_parts, test_parts = [], []
    for source_col in source_cols:
        sub = df[df[source_col] == 1].sort_values(timestamp_col, kind="mergesort")
        n = len(sub)
        if n == 0:
            continue
        n_test = int(np.ceil(n * test_size))
        n_train = n - n_test
        train_parts.append(sub.iloc[:n_train])
        test_parts.append(sub.iloc[n_train:])
        print(
            f"[Train] Source={source_col}: rows={n}, "
            f"train={n_train} ({train_parts[-1]['is_fraud'].mean():.4f} fraud), "
            f"test={n_test} ({test_parts[-1]['is_fraud'].mean():.4f} fraud), "
            f"ts_train_max={int(train_parts[-1][timestamp_col].max())}, "
            f"ts_test_min={int(test_parts[-1][timestamp_col].min())}"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def _setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow to write to a local sqlite store under the repo."""
    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def main() -> None:
    """Run Stage 4 training flow with temporal split + MLflow tracking."""
    params = load_params()
    data_cfg = params.get("data", {})
    cfg = params.get("train", {})

    features_path = Path(data_cfg.get("features_path", "data/processed/features.csv"))
    model_path = Path(data_cfg.get("model_path", "models/fraud_model.pkl"))
    x_test_path = Path(data_cfg.get("x_test_path", "data/processed/X_test.csv"))
    y_test_path = Path(data_cfg.get("y_test_path", "data/processed/y_test.csv"))

    if not features_path.exists():
        raise FileNotFoundError(f"[Train] Missing features file: {features_path}")

    print(f"[Train] Loading features from {features_path}")
    pdf = pd.read_csv(features_path)

    if "is_fraud" not in pdf.columns:
        raise KeyError("[Train] Target column 'is_fraud' not found in features dataframe.")
    if "txn_timestamp" not in pdf.columns:
        raise KeyError(
            "[Train] Target column 'txn_timestamp' not found — re-run preprocess "
            "stage to regenerate features.csv with timestamp preserved."
        )

    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    # Per-source temporal split (Day 1 fix).
    train_df, test_df = temporal_split_per_source(pdf, test_size=test_size)

    # txn_timestamp is a metadata column; drop from feature matrix.
    feature_cols = [c for c in pdf.columns if c not in ("is_fraud", "txn_timestamp")]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"].astype(int)

    if bool(cfg.get("use_smote", False)):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "[Train] SMOTE requested via use_smote=true but imbalanced-learn is not installed. "
                "Install with `pip install imbalanced-learn`, or set use_smote=false in params.yaml."
            ) from exc

        print("[Train] Applying SMOTE to training split")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    early_stopping = int(cfg.get("early_stopping_rounds", 0))

    _setup_mlflow("sentinel-day01-temporal-split")

    with mlflow.start_run(run_name="day01_phase1_temporal_split") as run:
        print(f"[Train] MLflow run_id={run.info.run_id}")

        mlflow.log_params(
            {
                "split_strategy": "temporal_per_source",
                "test_size": test_size,
                "random_state": random_state,
                "n_estimators": int(cfg.get("n_estimators", 200)),
                "max_depth": int(cfg.get("max_depth", 6)),
                "learning_rate": float(cfg.get("learning_rate", 0.1)),
                "scale_pos_weight": float(cfg.get("scale_pos_weight", 50)),
                "use_smote": bool(cfg.get("use_smote", False)),
                "early_stopping_rounds": early_stopping,
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "train_fraud_rate": float(y_train.mean()),
                "test_fraud_rate": float(y_test.mean()),
            }
        )

        model = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 200)),
            max_depth=int(cfg.get("max_depth", 6)),
            learning_rate=float(cfg.get("learning_rate", 0.1)),
            scale_pos_weight=float(cfg.get("scale_pos_weight", 50)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=early_stopping if early_stopping > 0 else None,
        )

        print(f"[Train] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"[Train] Test set: {X_test.shape[0]} samples")
        print(f"[Train] Fraud rate (train): {y_train.mean():.4f}")
        print(f"[Train] Fraud rate (test): {y_test.mean():.4f}")

        print("[Train] Fitting XGBoost model")
        fit_params = {}
        if early_stopping > 0:
            fit_params["eval_set"] = [(X_test, y_test)]
            fit_params["verbose"] = False
            print(f"[Train] Early stopping enabled: {early_stopping} rounds")

        model.fit(X_train, y_train, **fit_params)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else 0.0
        ap = float(average_precision_score(y_test, probs)) if y_test.nunique() > 1 else 0.0

        mlflow.log_metrics(
            {
                "test_accuracy": float(acc),
                "test_auc": auc,
                "test_average_precision": ap,
            }
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Load feature thresholds sidecar from preprocess stage so inference
        # reuses training-time percentiles/z-score stats (prevents leakage).
        thresholds: dict[str, float] = {}
        thresholds_path = features_path.parent / "feature_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            print(f"[Train] Loaded feature thresholds from {thresholds_path}")
        else:
            print(f"[Train] No thresholds sidecar at {thresholds_path}; artifact will omit them.")

        artifact = {
            "model": model,
            "feature_columns": list(feature_cols),
            "target_column": "is_fraud",
            "feature_thresholds": thresholds,
            "split_strategy": "temporal_per_source",
        }
        joblib.dump(artifact, model_path)

        x_test_path.parent.mkdir(parents=True, exist_ok=True)
        X_test.to_csv(x_test_path, index=False)
        pd.DataFrame({"is_fraud": y_test}).to_csv(y_test_path, index=False)

        try:
            mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        except Exception as exc:
            print(f"[Train] mlflow.xgboost.log_model failed (non-fatal): {exc}")
        mlflow.log_artifact(str(model_path))

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)

        print(f"[Train] Saved model to {model_path}")
        print(f"[Train] Saved X_test to {x_test_path}")
        print(f"[Train] Saved y_test to {y_test_path}")
        print(f"[Train] Accuracy: {acc:.6f}")
        print(f"[Train] Test AUC: {auc:.6f}")
        print(f"[Train] Test AP:  {ap:.6f}")
        print("[Train] Top 10 feature importances:")
        for feature, score in importances.items():
            print(f"  - {feature}: {float(score):.6f}")


if __name__ == "__main__":
    main()
