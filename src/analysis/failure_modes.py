"""Day 5 Phase 4: Failure-mode breakdown on sparkov_test.csv (OOT).

Loads results/day05/oot_predictions.parquet (written by eval_best.py) and
slices precision/recall/AUC/AP by:
  - amount bucket (<$10, $10-100, $100-1000, $1000-10000, >$10000)
  - hour of day (0-5 night, 6-11 morning, 12-17 afternoon, 18-23 evening)
  - merchant category (sparkov 'category' field — 14 retail categories)
  - gender (sparkov demographic — sanity check for fairness)

The point: where does the tuned model systematically miss fraud? That
answer drives the Day-5 targeted fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results" / "day05"
PRED_PATH = RESULTS_DIR / "oot_predictions.parquet"


def load_predictions() -> pd.DataFrame:
    if PRED_PATH.exists():
        return pd.read_parquet(PRED_PATH)
    csv_path = RESULTS_DIR / "oot_predictions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No OOT predictions at {PRED_PATH}; run `python -m src.tuning.eval_best` first."
    )


def _slice_metrics(sub: pd.DataFrame) -> dict:
    """Per-slice metrics. Skip AUC/AP when only one class present."""
    yt = sub["y_true"].astype(int).to_numpy()
    yp = sub["y_pred"].astype(int).to_numpy()
    pr = sub["y_prob"].astype(float).to_numpy()

    n = int(len(sub))
    n_fraud = int(yt.sum())
    n_pred_fraud = int(yp.sum())
    n_caught = int(((yt == 1) & (yp == 1)).sum())
    n_false_pos = int(((yt == 0) & (yp == 1)).sum())

    has_both_classes = n_fraud > 0 and n_fraud < n
    auc = float(roc_auc_score(yt, pr)) if has_both_classes else None
    ap = float(average_precision_score(yt, pr)) if has_both_classes else None

    recall = float(n_caught / max(n_fraud, 1))
    precision = float(n_caught / max(n_pred_fraud, 1)) if n_pred_fraud else 0.0

    return {
        "n": n,
        "n_fraud": n_fraud,
        "fraud_rate": float(n_fraud / max(n, 1)),
        "n_pred_fraud": n_pred_fraud,
        "n_caught": n_caught,
        "n_false_pos": n_false_pos,
        "recall_at_0.5": recall,
        "precision_at_0.5": precision,
        "auc": auc,
        "average_precision": ap,
    }


def breakdown_by(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for value, sub in df.groupby(column, dropna=False):
        m = _slice_metrics(sub)
        m["slice_variable"] = label
        m["slice_value"] = str(value)
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = ["slice_variable", "slice_value", "n", "n_fraud", "fraud_rate",
            "n_pred_fraud", "n_caught", "n_false_pos",
            "recall_at_0.5", "precision_at_0.5", "auc", "average_precision"]
    out = out[cols].sort_values(["slice_variable", "slice_value"]).reset_index(drop=True)
    return out


def main() -> None:
    df = load_predictions()
    print(f"[Failure] Loaded {len(df):,} OOT predictions from sparkov_test.csv")

    # Derived slicing columns.
    amt = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)
    df["amount_bucket"] = pd.cut(
        amt,
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<$10", "$10-100", "$100-1000", "$1000-10000", ">$10000"],
    ).astype("string")

    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    hour = ts.dt.hour.fillna(0).astype(int)
    df["hour_bucket"] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["00-05_night", "06-11_morning", "12-17_afternoon", "18-23_evening"],
    ).astype("string")

    breakdowns = []
    breakdowns.append(breakdown_by(df, "amount_bucket", "amount_bucket"))
    breakdowns.append(breakdown_by(df, "hour_bucket", "hour_bucket"))
    breakdowns.append(breakdown_by(df, "category", "merchant_category"))
    if "gender" in df.columns:
        breakdowns.append(breakdown_by(df, "gender", "gender"))

    full = pd.concat(breakdowns, ignore_index=True)
    full_path = RESULTS_DIR / "failure_modes.csv"
    full.to_csv(full_path, index=False)
    print(f"[Failure] Saved breakdown -> {full_path}")

    # Headline summary: which slice has the worst recall when fraud exists?
    fraud_only = full[full["n_fraud"] >= 5].copy()
    fraud_only["recall_rank"] = fraud_only["recall_at_0.5"].rank(method="min")
    fraud_only = fraud_only.sort_values(["slice_variable", "recall_at_0.5"])

    summary = {}
    for sv in fraud_only["slice_variable"].unique():
        sub = fraud_only[fraud_only["slice_variable"] == sv].copy()
        # Worst-recall slice for this variable + best for contrast.
        worst = sub.iloc[0].to_dict()
        best = sub.iloc[-1].to_dict()
        summary[sv] = {
            "worst": {k: worst[k] for k in ("slice_value", "n", "n_fraud",
                                             "recall_at_0.5", "auc")},
            "best": {k: best[k] for k in ("slice_value", "n", "n_fraud",
                                           "recall_at_0.5", "auc")},
            "spread_recall": float(best["recall_at_0.5"] - worst["recall_at_0.5"]),
        }

    # Dominant failure mode: the slice variable with the LARGEST spread in
    # recall across its values is where the model is most uneven — that's
    # where a targeted fix can move the needle most.
    dominant_variable = max(summary, key=lambda v: summary[v]["spread_recall"])

    headline = {
        "n_total": int(len(df)),
        "n_fraud_total": int(df["y_true"].sum()),
        "n_caught_total": int(((df["y_true"] == 1) & (df["y_pred"] == 1)).sum()),
        "overall_recall_at_0.5": float(
            ((df["y_true"] == 1) & (df["y_pred"] == 1)).sum() / max(int(df["y_true"].sum()), 1)
        ),
        "by_slice": summary,
        "dominant_failure_variable": dominant_variable,
        "interpretation": (
            f"Variable '{dominant_variable}' shows the widest gap in recall@0.5 "
            f"across its values "
            f"(spread={summary[dominant_variable]['spread_recall']:.3f}). "
            f"That's the slice where targeted re-weighting or category-aware "
            f"features should yield the biggest lift."
        ),
    }
    summary_path = RESULTS_DIR / "failure_modes_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)
    print(f"[Failure] Saved summary -> {summary_path}")

    print("\n[Failure] Per-slice recall spreads:")
    for var, info in summary.items():
        print(
            f"  - {var}: worst={info['worst']['slice_value']} "
            f"(recall={info['worst']['recall_at_0.5']:.3f}) "
            f"best={info['best']['slice_value']} "
            f"(recall={info['best']['recall_at_0.5']:.3f}) "
            f"spread={info['spread_recall']:.3f}"
        )
    print(f"\n[Failure] DOMINANT FAILURE MODE: {dominant_variable}")
    print(f"[Failure] {headline['interpretation']}")


if __name__ == "__main__":
    main()
