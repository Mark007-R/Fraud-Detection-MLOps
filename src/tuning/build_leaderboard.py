"""Aggregate Day 5 leaderboard CSV from the various result JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results" / "day05"


def main() -> None:
    rows: list[dict] = []

    rows.append(
        {
            "rank": "",
            "strategy": "Day-1 honest baseline (random XGB defaults, temporal split fix)",
            "training_scope": "full (6.1M)",
            "sample_weight": "uniform",
            "threshold": 0.5,
            "in_dist_sparkov_auc": 0.996608,
            "oot_sparkov_test_auc": 0.794896,
            "oot_ap": None,
            "oot_recall": None,
            "oot_precision": None,
            "oot_f1": None,
            "delta_vs_autogluon": round(0.794896 - 0.952, 4),
            "notes": "honest baseline -pre-tuning",
        }
    )

    # Day 5 Optuna tuned, full retrain, threshold=0.5.
    with open(RESULTS_DIR / "tuned_eval.json", "r", encoding="utf-8") as f:
        tuned = json.load(f)
    oot = tuned["out_of_time_sparkov_test_csv"]
    rows.append(
        {
            "rank": "",
            "strategy": "Day-5 Optuna best (30 trials) -full retrain",
            "training_scope": "full (6.1M)",
            "sample_weight": "uniform",
            "threshold": 0.5,
            "in_dist_sparkov_auc": tuned["in_distribution_per_source"]["source_sparkov"]["auc"],
            "oot_sparkov_test_auc": oot["auc"],
            "oot_ap": oot["average_precision"],
            "oot_recall": oot["recall_at_0.5"],
            "oot_precision": oot["precision_at_0.5"],
            "oot_f1": oot["f1_at_0.5"],
            "delta_vs_autogluon": round(oot["auc"] - 0.952, 4),
            "notes": "tuning recovers ranking; recall still collapses at 0.5",
        }
    )

    # Day 5 targeted fix at threshold 0.5.
    with open(RESULTS_DIR / "targeted_fix_eval.json", "r", encoding="utf-8") as f:
        fix = json.load(f)
    oot_fix = fix["out_of_time_sparkov_test_csv"]
    fix_05 = oot_fix["at_threshold_0.5"]
    fix_tau = oot_fix["at_threshold_tau_star"]

    rows.append(
        {
            "rank": "",
            "strategy": "Day-5 Optuna + source-balanced weights -threshold=0.5",
            "training_scope": "full (6.1M, paysim 0.60x / sparkov 2.95x)",
            "sample_weight": "source_balanced",
            "threshold": 0.5,
            "in_dist_sparkov_auc": fix["in_distribution_per_source"]["source_sparkov"]["auc"],
            "oot_sparkov_test_auc": oot_fix["auc"],
            "oot_ap": oot_fix["average_precision"],
            "oot_recall": fix_05["recall"],
            "oot_precision": fix_05["precision"],
            "oot_f1": fix_05["f1"],
            "delta_vs_autogluon": round(oot_fix["auc"] - 0.952, 4),
            "notes": "AUC ties AutoGluon; recall jumps 6x",
        }
    )

    rows.append(
        {
            "rank": "",
            "strategy": "Day-5 Optuna + source-balanced weights -threshold=tau*",
            "training_scope": "full (6.1M, paysim 0.60x / sparkov 2.95x)",
            "sample_weight": "source_balanced",
            "threshold": fix["decision_threshold_tau_star"],
            "in_dist_sparkov_auc": fix["in_distribution_per_source"]["source_sparkov"]["auc"],
            "oot_sparkov_test_auc": oot_fix["auc"],
            "oot_ap": oot_fix["average_precision"],
            "oot_recall": fix_tau["recall"],
            "oot_precision": fix_tau["precision"],
            "oot_f1": fix_tau["f1"],
            "delta_vs_autogluon": round(oot_fix["auc"] - 0.952, 4),
            "notes": "F1 trade: higher precision (0.55), lower recall (0.15)",
        }
    )

    # AutoGluon reference (literature baseline)
    rows.append(
        {
            "rank": "",
            "strategy": "AutoGluon (FDB published baseline)",
            "training_scope": "FDB standard",
            "sample_weight": "n/a",
            "threshold": "n/a",
            "in_dist_sparkov_auc": None,
            "oot_sparkov_test_auc": 0.952,
            "oot_ap": None,
            "oot_recall": None,
            "oot_precision": None,
            "oot_f1": None,
            "delta_vs_autogluon": 0.0,
            "notes": "reference baseline",
        }
    )

    df = pd.DataFrame(rows)
    df = df.sort_values("oot_sparkov_test_auc", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    out = RESULTS_DIR / "day05_leaderboard.csv"
    df.to_csv(out, index=False)
    print(f"[Leaderboard] Saved {len(df)} rows -> {out}")
    print(df[["rank", "strategy", "oot_sparkov_test_auc", "delta_vs_autogluon",
              "oot_recall", "oot_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
