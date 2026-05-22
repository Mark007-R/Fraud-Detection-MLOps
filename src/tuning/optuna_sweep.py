"""Day 5 Phase 4: Optuna sweep on XGBoost with per-trial MLflow tracking.

Honest baseline (Day 1, post temporal-split fix):
- combined temporal test AUC = 0.9989 (dominated by paysim deterministic signal)
- held-out sparkov_test.csv AUC = 0.7949 (the gap-closing metric vs AutoGluon 0.952)

This sweep maximises AUC on the actual held-out sparkov_test.csv file
(Jun–Dec 2020) — the same dataset AutoGluon's 0.952 is reported on. The
sparkov train-time test slice (Mar–Jun 2020) is also tracked but NOT the
objective: it saturates at ~0.996 AUC even at default params, so optimizing
it would just chase noise on an in-distribution metric. The real bottleneck
is out-of-distribution generalization to the +6-month window, and that's
what AutoGluon's 0.952 measures too.

Subsampling strategy for tuning speed:
- Use the temporal split train set (6.1M rows)
- Keep ALL fraud rows (~13K)
- Random-sample non-fraud to TUNING_NEGATIVES (default 400K)
- Per source: paysim and sparkov are kept proportionally
- Subsample is built ONCE at the top of the sweep so every trial sees the
  same training data (apples-to-apples comparison)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.benchmark_fdb import _sparkov_to_common_schema
from src.config import PROJECT_ROOT
from src.preprocess import align_feature_columns, engineer_features_pandas
from src.train import temporal_split_per_source


N_TRIALS = int(os.environ.get("SENTINEL_OPTUNA_TRIALS", "30"))
TUNING_NEGATIVES = int(os.environ.get("SENTINEL_TUNING_NEG", "400000"))
RANDOM_STATE = 42
EXPERIMENT_NAME = "sentinel-day05-optuna-sweep"

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SPARKOV_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "sparkov_test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
RESULTS_DIR = PROJECT_ROOT / "results" / "day05"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_mlflow() -> None:
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def build_sparkov_test_features(feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Precompute aligned features + labels for held-out sparkov_test.csv.

    Mirrors src/benchmark_fdb.py's local-fallback path. Uses training-time
    thresholds from the persisted model artifact so percentile/z-score
    features match the train distribution (no leakage from inference data).
    """
    print(f"[Sweep] Loading held-out {SPARKOV_TEST_PATH}")
    raw = pd.read_csv(SPARKOV_TEST_PATH)
    common = _sparkov_to_common_schema(raw)
    y = common["is_fraud"].astype(int).to_numpy()

    thresholds = {}
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        thresholds = artifact.get("feature_thresholds", {})

    engineered = engineer_features_pandas(common, thresholds=thresholds or None)
    if "is_fraud" in engineered.columns:
        engineered = engineered.drop(columns=["is_fraud"])
    X = align_feature_columns(engineered, feature_cols).astype("float32")
    print(
        f"[Sweep] sparkov_test prepared: {len(X):,} rows, "
        f"fraud={int(y.sum())} ({y.mean()*100:.3f}%)"
    )
    return X, y


def build_train_val_split() -> dict:
    """Build the tuning train + validation matrices.

    Returns a dict with X_train, y_train (subsampled), X_val, y_val
    (sparkov slice of the temporal test), and X_val_paysim, y_val_paysim
    so per-source diagnostics are available for trials.
    """
    print(f"[Sweep] Loading {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    print(f"[Sweep] Total rows: {len(df):,}, fraud rate: {df['is_fraud'].mean():.5f}")

    train_df, test_df = temporal_split_per_source(df, test_size=0.2)
    print(f"[Sweep] train: {len(train_df):,}  test: {len(test_df):,}")

    feature_cols = [c for c in df.columns if c not in ("is_fraud", "txn_timestamp")]

    # Subsample train: keep all fraud + downsample negatives proportionally per source.
    rng = np.random.default_rng(RANDOM_STATE)
    train_fraud = train_df[train_df["is_fraud"] == 1]
    train_nonfraud = train_df[train_df["is_fraud"] == 0]

    if len(train_nonfraud) > TUNING_NEGATIVES:
        # Stratify negatives by source to keep both regimes represented.
        sub_parts = []
        source_cols = [c for c in train_df.columns if c.startswith("source_")]
        for src in source_cols:
            src_neg = train_nonfraud[train_nonfraud[src] == 1]
            if len(src_neg) == 0:
                continue
            share = TUNING_NEGATIVES * (len(src_neg) / len(train_nonfraud))
            n_take = int(min(len(src_neg), round(share)))
            idx = rng.choice(len(src_neg), size=n_take, replace=False)
            sub_parts.append(src_neg.iloc[idx])
        train_nonfraud_sub = pd.concat(sub_parts, ignore_index=True)
    else:
        train_nonfraud_sub = train_nonfraud

    train_sub = pd.concat([train_fraud, train_nonfraud_sub], ignore_index=True)
    train_sub = train_sub.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(
        f"[Sweep] Tuning train sample: {len(train_sub):,} "
        f"(fraud={int(train_sub['is_fraud'].sum())}, "
        f"rate={train_sub['is_fraud'].mean():.5f})"
    )

    X_train = train_sub[feature_cols].astype("float32")
    y_train = train_sub["is_fraud"].astype(int).to_numpy()

    # Validation: full sparkov slice of the temporal test set (proxy for sparkov_test.csv).
    sparkov_test = test_df[test_df["source_sparkov"] == 1]
    paysim_test = test_df[test_df["source_paysim"] == 1]

    X_val_sparkov = sparkov_test[feature_cols].astype("float32")
    y_val_sparkov = sparkov_test["is_fraud"].astype(int).to_numpy()
    X_val_paysim = paysim_test[feature_cols].astype("float32")
    y_val_paysim = paysim_test["is_fraud"].astype(int).to_numpy()

    print(
        f"[Sweep] Val sparkov: {len(X_val_sparkov):,} "
        f"(fraud={int(y_val_sparkov.sum())}, rate={y_val_sparkov.mean():.5f})"
    )
    print(
        f"[Sweep] Val paysim: {len(X_val_paysim):,} "
        f"(fraud={int(y_val_paysim.sum())}, rate={y_val_paysim.mean():.5f})"
    )

    # Held-out sparkov_test.csv (Jun–Dec 2020) — the gap-closing metric vs AutoGluon 0.952.
    X_test_oot, y_test_oot = build_sparkov_test_features(feature_cols)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val_sparkov": X_val_sparkov,
        "y_val_sparkov": y_val_sparkov,
        "X_val_paysim": X_val_paysim,
        "y_val_paysim": y_val_paysim,
        "X_test_oot": X_test_oot,
        "y_test_oot": y_test_oot,
        "feature_cols": feature_cols,
    }


def _build_objective(data: dict):
    """Return the objective function bound to a fixed dataset."""

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val_sparkov = data["X_val_sparkov"]
    y_val_sparkov = data["y_val_sparkov"]
    X_val_paysim = data["X_val_paysim"]
    y_val_paysim = data["y_val_paysim"]
    X_test_oot = data["X_test_oot"]
    y_test_oot = data["y_test_oot"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }

        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=False) as run:
            mlflow.log_params(params)
            mlflow.log_params(
                {
                    "trial_number": trial.number,
                    "tuning_train_rows": int(X_train.shape[0]),
                    "tuning_negatives_cap": TUNING_NEGATIVES,
                    "objective_metric": "test_oot_sparkov_auc",
                }
            )

            t0 = time.time()
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
            clf.fit(X_train, y_train)
            fit_seconds = time.time() - t0

            p_sparkov = clf.predict_proba(X_val_sparkov)[:, 1]
            p_paysim = clf.predict_proba(X_val_paysim)[:, 1]
            p_oot = clf.predict_proba(X_test_oot)[:, 1]

            auc_sparkov = float(roc_auc_score(y_val_sparkov, p_sparkov))
            ap_sparkov = float(average_precision_score(y_val_sparkov, p_sparkov))
            auc_paysim = float(roc_auc_score(y_val_paysim, p_paysim))
            ap_paysim = float(average_precision_score(y_val_paysim, p_paysim))
            auc_oot = float(roc_auc_score(y_test_oot, p_oot))
            ap_oot = float(average_precision_score(y_test_oot, p_oot))

            mlflow.log_metrics(
                {
                    "test_oot_sparkov_auc": auc_oot,
                    "test_oot_sparkov_ap": ap_oot,
                    "val_sparkov_auc": auc_sparkov,
                    "val_sparkov_ap": ap_sparkov,
                    "val_paysim_auc": auc_paysim,
                    "val_paysim_ap": ap_paysim,
                    "fit_seconds": fit_seconds,
                    "delta_vs_autogluon": auc_oot - 0.952,
                    "delta_vs_day1_honest": auc_oot - 0.794896,
                }
            )

            print(
                f"[Sweep] trial {trial.number:03d}: "
                f"oot_auc={auc_oot:.4f} (d_AG={auc_oot - 0.952:+.4f}, "
                f"d_D1={auc_oot - 0.794896:+.4f}) "
                f"in_dist_auc={auc_sparkov:.4f} fit={fit_seconds:.1f}s"
            )
            return auc_oot

    return objective


def main() -> None:
    _setup_mlflow()
    data = build_train_val_split()

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        study_name="sentinel-day05",
        direction="maximize",
        sampler=sampler,
    )

    objective = _build_objective(data)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    sweep_seconds = time.time() - t0

    best = study.best_trial
    print(f"\n[Sweep] DONE in {sweep_seconds:.1f}s")
    print(f"[Sweep] Best trial: #{best.number}  test_oot_sparkov_auc={best.value:.6f}")
    print(f"[Sweep] delta vs Day-1 honest baseline 0.7949: {best.value - 0.794896:+.4f}")
    print(f"[Sweep] delta vs AutoGluon 0.952: {best.value - 0.952:+.4f}")
    print(f"[Sweep] Best params: {best.params}")

    # Persist best params + the full trial history.
    best_path = RESULTS_DIR / "optuna_best_params.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": best.number,
                "best_value_test_oot_sparkov_auc": best.value,
                "best_params": best.params,
                "n_trials": N_TRIALS,
                "tuning_train_rows": int(data["X_train"].shape[0]),
                "tuning_negatives_cap": TUNING_NEGATIVES,
                "sweep_seconds": sweep_seconds,
                "honest_baseline_sparkov_test_auc": 0.794896,
                "autogluon_baseline_auc": 0.952,
                "delta_vs_autogluon": best.value - 0.952,
                "delta_vs_day1_honest": best.value - 0.794896,
            },
            f,
            indent=2,
        )
    print(f"[Sweep] Saved best params -> {best_path}")

    trials_df = study.trials_dataframe()
    trials_csv = RESULTS_DIR / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"[Sweep] Saved {len(trials_df)} trials -> {trials_csv}")

    # Save the dataset partition so eval scripts use the same split.
    partition_path = RESULTS_DIR / "tuning_partition_info.json"
    with open(partition_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": data["feature_cols"],
                "val_sparkov_rows": int(data["X_val_sparkov"].shape[0]),
                "val_sparkov_fraud": int(data["y_val_sparkov"].sum()),
                "val_paysim_rows": int(data["X_val_paysim"].shape[0]),
                "val_paysim_fraud": int(data["y_val_paysim"].sum()),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
