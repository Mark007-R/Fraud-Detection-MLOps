"""Day 6 — head-to-head on the same 200-row OOT sample.

Joins three score sources:

    1. ``results/day06/llm_predictions.csv``     — produced by ``llm_judge.py``.
    2. ``results/day05/oot_predictions.parquet`` — the Day-5 champion
       XGBoost (full pipeline: temporal split + source-balanced + tuned).
    3. A "naive notebook" XGBoost trained on a *random* (non-temporal) split
       of the in-distribution train data using stock defaults — represents
       the counterfactual where Sentinel's MLOps discipline never existed.

All three are scored on the same 200-row OOT sample.  Output:

    * ``results/day06/frontier_comparison.csv``
    * ``results/day06/frontier_comparison.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "day06"
DAY05_OOT = REPO_ROOT / "results" / "day05" / "oot_predictions.parquet"
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features.csv"
CHAMPION_PKL = REPO_ROOT / "models" / "fraud_model_tuned_fixed.pkl"

LOG = logging.getLogger("frontier.compare")


def _metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "auc": float(roc_auc_score(y_true, y_score)) if y_true.sum() and (y_true == 0).any() else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if y_true.sum() else float("nan"),
        "recall_at_0_5": recall,
        "precision_at_0_5": precision,
        "f1_at_0_5": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _build_naive_baseline(seed: int = 0) -> tuple[XGBClassifier, list[str]]:
    """A *deliberately* naive notebook-style baseline.

    This is what the project would have looked like with no MLOps:
    train on a random split of a *subsample* of the processed features,
    stock XGBoost defaults, no source balancing, no tuning, no temporal
    awareness.  Saved purely for the frontier-comparison and never
    promoted to the registry.
    """
    LOG.info("Training naive notebook baseline (random split, default XGB, no balancing)...")
    feats = pd.read_csv(FEATURES_CSV)
    # Subsample to keep the naive notebook tractable in seconds, mirroring
    # what a data scientist would actually do at the start of a project.
    sub = feats.sample(n=300_000, random_state=seed)
    feature_cols = [c for c in sub.columns if c not in {"is_fraud", "txn_timestamp"}]
    X = sub[feature_cols]
    y = sub["is_fraud"].astype(int)

    from sklearn.model_selection import train_test_split
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    LOG.info("Naive baseline trained in %.1fs", time.perf_counter() - t0)
    return model, feature_cols


def _score_naive_on_sample(naive: XGBClassifier, feature_cols: list[str], sample_trans_nums: list[str]) -> pd.DataFrame:
    """Pull feature rows for the sampled trans_nums and score with naive model."""
    LOG.info("Scoring naive baseline on %d sampled trans_nums...", len(sample_trans_nums))
    feats = pd.read_csv(FEATURES_CSV)
    # `features.csv` has no trans_num.  We can't join by trans_num, so we
    # instead use the Day-5 OOT predictions as the bridge: the Day-5 file
    # already pairs trans_num with the OOT row index.  But the naive model
    # was trained on a random subsample of `features.csv`, not `sparkov_test`,
    # so we need to score it on the OOT feature space.  Day-5 used
    # `src/benchmark_fdb.py` to do this; we mimic it via the saved
    # OOT predictions.  Build the OOT features by re-preprocessing sparkov_test.
    # Avoiding redoing preprocessing: use the saved predictions parquet as
    # the source of truth for (trans_num, y_true) and score naive on the
    # *processed* features matching those rows via order from
    # `data/processed/X_test.csv` is not available.  Simpler: re-run the
    # naive model on sparkov_test via the same preprocessing path used by
    # `src/benchmark_fdb.py`.  See _score_naive_via_benchmark below.
    raise NotImplementedError


def _build_oot_features(raw_path: Path = REPO_ROOT / "data" / "raw" / "sparkov_test.csv") -> pd.DataFrame:
    """Re-derive the 45-column feature matrix used at training time.

    Lifted from `src.benchmark_fdb` so we don't need to import it (avoids
    DVC/MLflow side effects at import time).  Produces exact same columns
    as `models/fraud_model_tuned_fixed.pkl`'s feature_columns.
    """
    raw = pd.read_csv(raw_path)
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"])
    out = pd.DataFrame()
    out["trans_num"] = raw["trans_num"]
    out["is_fraud"] = raw["is_fraud"].astype(int)
    out["amount"] = raw["amt"]
    out["hour_of_day"] = raw["trans_date_trans_time"].dt.hour
    out["day_of_month"] = raw["trans_date_trans_time"].dt.day
    out["balance_change_orig"] = 0.0
    out["balance_ratio"] = 0.0
    out["has_balance_info"] = 0
    out["tx_amount_log"] = np.log1p(raw["amt"])
    # Thresholds from the champion model (precomputed at train time).
    bundle = joblib.load(CHAMPION_PKL)
    th = bundle["feature_thresholds"]
    out["is_high_amount"] = (raw["amt"] > th["amount_mean"]).astype(int)
    out["is_p95_amount"] = (raw["amt"] > th["amount_p95"]).astype(int)
    out["is_p99_amount"] = (raw["amt"] > th["amount_p99"]).astype(int)
    out["amount_zscore"] = (raw["amt"] - th["amount_mean"]) / max(th["amount_std"], 1e-6)
    h = out["hour_of_day"]
    d = out["day_of_month"]
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["day_sin"] = np.sin(2 * np.pi * d / 31)
    out["day_cos"] = np.cos(2 * np.pi * d / 31)
    out["is_night"] = ((h < 6) | (h >= 22)).astype(int)
    out["is_weekend"] = raw["trans_date_trans_time"].dt.dayofweek.isin([5, 6]).astype(int)
    out["balance_change_abs"] = 0.0
    out["balance_ratio_clipped"] = 0.0
    out["balance_change_log"] = 0.0
    # One-hot transaction types: all paysim flags off (sparkov has no paysim type)
    for col in [
        "transaction_type_CASH_IN", "transaction_type_CASH_OUT", "transaction_type_DEBIT",
        "transaction_type_PAYMENT", "transaction_type_TRANSFER",
    ]:
        out[col] = 0
    # Sparkov category one-hot
    for cat in [
        "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
        "shopping_net", "shopping_pos", "travel",
    ]:
        out[f"transaction_type_{cat}"] = (raw["category"] == cat).astype(int)
    out["source_paysim"] = 0
    out["source_sparkov"] = 1
    out["amount_bin_high"] = ((raw["amt"] > th["amount_p95"]) & (raw["amt"] <= th["amount_p99"])).astype(int)
    out["amount_bin_low"] = (raw["amt"] <= th["amount_mean"]).astype(int)
    out["amount_bin_medium"] = ((raw["amt"] > th["amount_mean"]) & (raw["amt"] <= th["amount_p95"])).astype(int)
    out["amount_bin_very_high"] = (raw["amt"] > th["amount_p99"]).astype(int)
    return out


def run(out_dir: Path = RESULTS_DIR, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Loading LLM predictions from %s", out_dir / "llm_predictions.csv")
    llm_pred = pd.read_csv(out_dir / "llm_predictions.csv")
    sample_trans = llm_pred["trans_num"].tolist()

    # Champion XGBoost predictions for the same 200 rows.
    LOG.info("Loading Day-5 champion predictions from %s", DAY05_OOT)
    day5 = pd.read_parquet(DAY05_OOT, columns=["trans_num", "y_true", "y_prob", "y_pred"])
    champ = day5[day5["trans_num"].isin(sample_trans)].set_index("trans_num")
    LOG.info("Joined %d / %d champion predictions", len(champ), len(sample_trans))

    # Naive notebook baseline: re-derive OOT features and score.
    naive_model, feat_cols = _build_naive_baseline(seed=seed)
    LOG.info("Re-deriving OOT feature matrix for naive scoring...")
    oot_feat = _build_oot_features()
    oot_sub = oot_feat[oot_feat["trans_num"].isin(sample_trans)].set_index("trans_num")
    # Align column order — naive may have used a column list with txn_timestamp;
    # drop any feature the champion bundle doesn't list to keep the matrices compatible.
    bundle_cols = joblib.load(CHAMPION_PKL)["feature_columns"]
    naive_feats_used = [c for c in feat_cols if c in bundle_cols]
    Xn = oot_sub[naive_feats_used]
    t0 = time.perf_counter()
    naive_proba = naive_model.predict_proba(Xn)[:, 1]
    naive_lat = (time.perf_counter() - t0) / len(Xn)
    naive_pred = (naive_proba >= 0.5).astype(int)

    # Order everything by the LLM prediction row order
    llm_pred = llm_pred.set_index("trans_num").loc[sample_trans]
    champ = champ.loc[sample_trans]
    naive_proba_series = pd.Series(naive_proba, index=Xn.index).loc[sample_trans]
    naive_pred_series = pd.Series(naive_pred, index=Xn.index).loc[sample_trans]

    y_true = llm_pred["is_fraud"].to_numpy()

    llm_metrics = _metrics(
        y_true=y_true,
        y_score=llm_pred["suspicion_score"].to_numpy(),
        y_pred=llm_pred["is_suspicious"].to_numpy(),
    )
    champ_metrics = _metrics(
        y_true=y_true,
        y_score=champ["y_prob"].to_numpy(),
        y_pred=champ["y_pred"].to_numpy(),
    )
    naive_metrics = _metrics(
        y_true=y_true,
        y_score=naive_proba_series.to_numpy(),
        y_pred=naive_pred_series.to_numpy(),
    )

    # Latency: champion is XGBoost inference (microseconds/row); LLM is recorded.
    # Time champion locally for honest comparison.
    LOG.info("Timing champion XGB inference on the sample for latency...")
    champ_bundle = joblib.load(CHAMPION_PKL)
    feat_for_champ = oot_sub[champ_bundle["feature_columns"]]
    t0 = time.perf_counter()
    _ = champ_bundle["model"].predict_proba(feat_for_champ)
    champ_lat_per_row = (time.perf_counter() - t0) / len(feat_for_champ)

    rows = [
        {
            "strategy": "Sentinel champion (Day-5 XGBoost: temporal split + source-balanced + Optuna)",
            **champ_metrics,
            "latency_s_per_query": champ_lat_per_row,
            "cost_usd_per_query": 5e-9,  # ~5 ns of CPU, electricity-only
            "cost_usd_at_1k_qps_per_day": 5e-9 * 1000 * 86400,
        },
        {
            "strategy": "Naive notebook XGBoost (random split, defaults, no MLOps)",
            **naive_metrics,
            "latency_s_per_query": naive_lat,
            "cost_usd_per_query": 5e-9,
            "cost_usd_at_1k_qps_per_day": 5e-9 * 1000 * 86400,
        },
        {
            "strategy": "Claude Opus 4.6 LLM-judged (frontier model)",
            **llm_metrics,
            "latency_s_per_query": float(llm_pred["latency_s"].mean()),
            "cost_usd_per_query": float(((llm_pred["input_tokens"].sum() * 0.015 + llm_pred["output_tokens"].sum() * 0.075) / 1000) / len(llm_pred)),
            "cost_usd_at_1k_qps_per_day": None,
        },
    ]
    rows[2]["cost_usd_at_1k_qps_per_day"] = rows[2]["cost_usd_per_query"] * 1000 * 86400

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "frontier_comparison.csv", index=False)
    with open(out_dir / "frontier_comparison.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    LOG.info("Wrote %s", out_dir / "frontier_comparison.csv")
    return {"rows": rows, "n_sample": int(len(llm_pred)), "n_fraud": int(y_true.sum())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    s = run(out_dir=args.out_dir, seed=args.seed)
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
