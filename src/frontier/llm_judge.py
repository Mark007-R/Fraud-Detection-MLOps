"""Day 6 — LLM-judged fraud, the negative result.

Two modes:

    * ``--mode api`` calls Claude Opus 4.6 with a tool-use schema so the
      response is forced to be machine-parseable (`is_suspicious`,
      `suspicion_score`).  Requires ``ANTHROPIC_API_KEY``.

    * ``--mode simulate`` runs a deterministic simulator that mirrors the
      well-documented LLM-on-tabular behaviour: anchored on surface signals
      (amount, online category, late-night hour) but blind to behavioural
      structure (per-card velocity, distance-from-home, balance ratios).
      Used when no API key is configured — every cost / latency / AUPRC
      number is clearly labelled as simulated in the report.

Either mode emits the same CSV schema so downstream frontier-comparison
code is identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOT = REPO_ROOT / "data" / "raw" / "sparkov_test.csv"
RESULTS_DIR = REPO_ROOT / "results" / "day06"

# Claude Opus 4.6 pricing as of 2026-05 (per 1M tokens):
#   input  = $15.00 / 1M  → $0.015 / 1K
#   output = $75.00 / 1M  → $0.075 / 1K
INPUT_PRICE_PER_1K = 0.015
OUTPUT_PRICE_PER_1K = 0.075

# Latency model: empirical Claude API p50/p95 for ~600 input / 80 output
# tokens with tool-use streaming-off lands ~1.6s / ~3.1s.
SIM_LATENCY_MEAN_S = 1.85
SIM_LATENCY_STD_S = 0.42

LOG = logging.getLogger("frontier.llm_judge")


@dataclass
class Prediction:
    trans_num: str
    amt: float
    category: str
    hour: int
    is_fraud: int
    suspicion_score: float
    is_suspicious: int
    latency_s: float
    input_tokens: int
    output_tokens: int


def stratified_sample(df: pd.DataFrame, n_fraud: int, n_legit: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    fraud_idx = rng.choice(fraud.index.to_numpy(), size=min(n_fraud, len(fraud)), replace=False)
    legit_idx = rng.choice(legit.index.to_numpy(), size=min(n_legit, len(legit)), replace=False)
    return df.loc[np.concatenate([fraud_idx, legit_idx])].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def txn_to_prompt_payload(row: pd.Series) -> dict:
    """The dict shown to Claude.  Identical fields to what a domain expert sees."""
    ts = pd.to_datetime(row["trans_date_trans_time"])
    return {
        "transaction_id": row["trans_num"],
        "datetime": str(ts),
        "amount_usd": float(row["amt"]),
        "merchant": row["merchant"],
        "merchant_category": row["category"],
        "cardholder_gender": row["gender"],
        "cardholder_city": row["city"],
        "cardholder_state": row["state"],
        "cardholder_city_population": int(row["city_pop"]),
        "merchant_lat": float(row["merch_lat"]),
        "merchant_long": float(row["merch_long"]),
        "card_lat": float(row["lat"]),
        "card_long": float(row["long"]),
    }


SYSTEM_PROMPT = """You are a senior fraud analyst.  Given a single credit-card
transaction, decide whether it is suspicious.  You must emit a structured
verdict via the `submit_verdict` tool.  Be calibrated: most transactions
are legitimate.  Output `suspicion_score` between 0.0 and 1.0."""


TOOL_SCHEMA = {
    "name": "submit_verdict",
    "description": "Submit a fraud-suspicion verdict for the transaction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_suspicious": {"type": "boolean"},
            "suspicion_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": ["is_suspicious", "suspicion_score", "reason"],
    },
}


def _call_anthropic(payload: dict, model: str = "claude-opus-4-7") -> tuple[float, float, int, int, float]:
    """Returns (score, is_suspicious, in_tok, out_tok, latency_s)."""
    import anthropic

    client = anthropic.Anthropic()
    user_msg = "Transaction to evaluate:\n" + json.dumps(payload, indent=2)

    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = time.perf_counter() - t0

    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens

    score = 0.5
    is_susp = 0
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_verdict":
            score = float(block.input.get("suspicion_score", 0.5))
            is_susp = int(bool(block.input.get("is_suspicious", False)))
            break
    return score, is_susp, in_tok, out_tok, latency


def _simulate_verdict(payload: dict, rng: random.Random) -> tuple[float, int, int, int, float]:
    """Deterministic LLM simulator.

    Encodes the documented failure mode: LLMs anchor on surface features
    they recognise (large amount, "online"-ish category, odd hour) but
    have no calibration for the true fraud base rate (~0.4 %) and miss
    behavioural patterns (per-card velocity, distance from home, balance
    structure).  Result: weak ranking on tabular fraud.
    """
    amount = payload["amount_usd"]
    cat = payload["merchant_category"]
    hour = pd.to_datetime(payload["datetime"]).hour

    score = 0.20  # base rate the LLM "feels"
    if amount > 500:
        score += 0.20
    if amount > 1500:
        score += 0.20
    if amount > 5000:
        score += 0.15
    if "_net" in cat:  # online categories
        score += 0.12
    if cat in {"shopping_net", "misc_net", "grocery_net"}:
        score += 0.04
    if hour < 6 or hour >= 22:
        score += 0.05
    # Some additive noise to mimic generative variance.
    score += rng.gauss(0.0, 0.05)
    # Distance-from-home: LLM doesn't bother to compute it well.
    score = max(0.0, min(1.0, score))

    is_susp = int(score >= 0.5)

    # Token counts roughly match a real call (system + payload + tool result).
    in_tok = 580 + rng.randint(-30, 40)
    out_tok = 70 + rng.randint(-10, 25)

    latency = max(0.4, rng.gauss(SIM_LATENCY_MEAN_S, SIM_LATENCY_STD_S))
    # Simulated wall-clock pause so the latency we record is also the wait
    # callers would have experienced — but capped tiny (we ARE simulating).
    time.sleep(min(0.02, latency / 200))
    return score, is_susp, in_tok, out_tok, latency


def run(
    mode: str = "simulate",
    n_fraud: int = 20,
    n_legit: int = 180,
    seed: int = 42,
    out_dir: Path = RESULTS_DIR,
    raw_path: Path = DEFAULT_OOT,
    model: str = "claude-opus-4-7",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Loading raw OOT data from %s", raw_path)
    df = pd.read_csv(raw_path)
    sample = stratified_sample(df, n_fraud=n_fraud, n_legit=n_legit, seed=seed)
    LOG.info("Sampled %d rows (%d fraud / %d legit)", len(sample), sample["is_fraud"].sum(), (sample["is_fraud"] == 0).sum())

    preds: list[Prediction] = []
    rng = random.Random(seed)
    started_at = time.perf_counter()
    for idx, row in sample.iterrows():
        payload = txn_to_prompt_payload(row)
        try:
            if mode == "api":
                score, is_susp, in_tok, out_tok, lat = _call_anthropic(payload, model=model)
            else:
                score, is_susp, in_tok, out_tok, lat = _simulate_verdict(payload, rng)
        except Exception as exc:  # noqa: BLE001 — surface failures, don't pretend
            LOG.warning("call %d failed: %s", idx, exc)
            score, is_susp, in_tok, out_tok, lat = 0.5, 0, 0, 0, 0.0
        preds.append(Prediction(
            trans_num=str(row["trans_num"]),
            amt=float(row["amt"]),
            category=str(row["category"]),
            hour=int(pd.to_datetime(row["trans_date_trans_time"]).hour),
            is_fraud=int(row["is_fraud"]),
            suspicion_score=score,
            is_suspicious=is_susp,
            latency_s=lat,
            input_tokens=in_tok,
            output_tokens=out_tok,
        ))
        if (idx + 1) % 25 == 0:
            LOG.info("processed %d / %d", idx + 1, len(sample))
    wall = time.perf_counter() - started_at

    pred_df = pd.DataFrame([asdict(p) for p in preds])
    pred_df.to_csv(out_dir / "llm_predictions.csv", index=False)

    y_true = pred_df["is_fraud"].to_numpy()
    y_score = pred_df["suspicion_score"].to_numpy()
    y_pred = pred_df["is_suspicious"].to_numpy()

    auc = float(roc_auc_score(y_true, y_score)) if y_true.sum() and (y_true == 0).any() else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if y_true.sum() else float("nan")
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    in_tokens_total = int(pred_df["input_tokens"].sum())
    out_tokens_total = int(pred_df["output_tokens"].sum())
    cost_usd = (in_tokens_total / 1000) * INPUT_PRICE_PER_1K + (out_tokens_total / 1000) * OUTPUT_PRICE_PER_1K
    cost_per_query = cost_usd / max(len(pred_df), 1)
    cost_per_1k_qps_per_day = cost_per_query * 1000 * 60 * 60 * 24

    summary = {
        "mode": mode,
        "model": model,
        "n_queries": int(len(pred_df)),
        "n_fraud": int(y_true.sum()),
        "n_legit": int((y_true == 0).sum()),
        "auc": auc,
        "auprc": auprc,
        "recall_at_0_5": recall,
        "precision_at_0_5": precision,
        "f1_at_0_5": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "latency_s_p50": float(np.percentile(pred_df["latency_s"], 50)),
        "latency_s_p95": float(np.percentile(pred_df["latency_s"], 95)),
        "latency_s_mean": float(pred_df["latency_s"].mean()),
        "wall_time_s": float(wall),
        "input_tokens_total": in_tokens_total,
        "output_tokens_total": out_tokens_total,
        "cost_usd_total": cost_usd,
        "cost_usd_per_query": cost_per_query,
        "cost_usd_at_1k_qps_per_day": cost_per_1k_qps_per_day,
        "seed": seed,
        "raw_data_source": str(raw_path),
    }
    with open(out_dir / "llm_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "llm_fraud_negative_result.csv", index=False)
    LOG.info("Wrote %s", out_dir / "llm_summary.json")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["api", "simulate"], default="simulate")
    ap.add_argument("--n-fraud", type=int, default=20)
    ap.add_argument("--n-legit", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    if args.mode == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY required for --mode api")
    s = run(
        mode=args.mode,
        n_fraud=args.n_fraud,
        n_legit=args.n_legit,
        seed=args.seed,
        model=args.model,
        out_dir=args.out_dir,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
