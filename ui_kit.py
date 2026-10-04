"""Shared building blocks for the SENTINEL Streamlit pages.

Paths, cached readers for the committed artifacts, the model scorer, the demo
batch, and the small HTML components that ui_theme.components_css() styles.
Every page in views/ imports from here so numbers and markup stay consistent.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METRICS_PATH = ROOT / "metrics" / "scores.json"
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
REPO_URL = "https://github.com/Mark007-R/Fraud-Detection-MLOps"

THRESHOLD = 0.5  # src.predict flags at probability >= 0.5

PAYSIM_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
SPARKOV_CATEGORIES = [
    "shopping_net", "shopping_pos", "grocery_pos", "grocery_net", "misc_net", "misc_pos",
    "gas_transport", "food_dining", "entertainment", "health_fitness", "home",
    "kids_pets", "personal_care", "travel",
]
RAW_COLUMNS = [
    "amount", "transaction_type", "hour_of_day", "day_of_month",
    "balance_change_orig", "balance_ratio", "has_balance_info", "source",
]
SOURCE_LABELS = {"paysim": "PaySim · mobile money", "sparkov": "Sparkov · card"}


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_json(relpath: str) -> dict:
    """Read a JSON artifact relative to the repo root; {} when missing or broken."""
    try:
        return json.loads((ROOT / relpath).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def read_csv(relpath: str) -> pd.DataFrame | None:
    """Read a CSV artifact relative to the repo root; None when missing or empty."""
    path = ROOT / relpath
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return None if df.empty else df


def headline() -> dict:
    """The measured numbers quoted across pages, read from results/ with README fallbacks."""
    out = {
        "oot_auc": 0.95196, "autogluon_auc": 0.952, "leaky_auc": 0.9210, "honest_auc": 0.7949,
        "lift": 0.4013, "l0_auc": 0.5467, "l3_auc": 0.9480,
        "rollback_ms": 3.9, "rollback_audited_ms": 11.9,
        "drift_lag": 0, "drift_precision": 1.0, "drift_recall": 1.0,
        "retrain_p50_s": 6.85, "promoted": 3, "retrains": 3,
    }
    board = read_csv("results/day05/day05_leaderboard.csv")
    if board is not None:
        ag = board[board["strategy"].str.contains("AutoGluon", na=False)]
        best = board[board["strategy"].str.contains("source-balanced", na=False)]
        if not ag.empty:
            out["autogluon_auc"] = float(ag["oot_sparkov_test_auc"].iloc[0])
        if not best.empty:
            out["oot_auc"] = float(best["oot_sparkov_test_auc"].iloc[0])
    base = read_json("results/baseline_metrics.json")
    if base.get("pre_fix", {}).get("reported_sparkov_benchmark_auc"):
        out["leaky_auc"] = float(base["pre_fix"]["reported_sparkov_benchmark_auc"])
    ablation = read_csv("results/day06/ablation_modelling.csv")
    if ablation is not None and len(ablation) >= 2:
        out["l0_auc"] = float(ablation["oot_auc"].iloc[0])
        out["l3_auc"] = float(ablation["oot_auc"].iloc[-1])
        out["lift"] = out["l3_auc"] - out["l0_auc"]
    rollback = read_csv("results/registry_rollback_times.csv")
    if rollback is not None:
        out["rollback_ms"] = float(rollback["alias_flip_seconds"].median() * 1000)
        out["rollback_audited_ms"] = float(rollback["total_seconds"].median() * 1000)
    drift = read_json("results/drift_replay_summary.json")
    if drift:
        out["drift_lag"] = int(drift.get("detection_lag_days", 0))
        out["drift_precision"] = float(drift.get("precision", 1.0))
        out["drift_recall"] = float(drift.get("recall", 1.0))
    retrain = read_csv("results/drift_retrain_events.csv")
    if retrain is not None:
        out["retrain_p50_s"] = float(retrain["seconds_end_to_end"].median())
        out["retrains"] = len(retrain)
        out["promoted"] = int(retrain["promote_decision"].astype(str).str.lower().eq("true").sum())
    return out


def strategy_label(name: str) -> str:
    """Short display names for the strategies in results/day06/frontier_comparison.csv."""
    low = name.lower()
    if "champion" in low:
        return "SENTINEL champion"
    if "naive" in low:
        return "Naive notebook XGBoost"
    if "llm" in low:
        return "Frontier LLM as judge"
    return name


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict | None:
    """The trained artifact dict from models/fraud_model.pkl, or None if absent or invalid."""
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib

        artifact = joblib.load(MODEL_PATH)
    except Exception:
        return None
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        return None
    return artifact


@st.cache_data(show_spinner=False, max_entries=512)
def score(frame: pd.DataFrame) -> pd.DataFrame:
    """fraud_probability + fraud_prediction per row, via src.predict.predict_dataframe."""
    from src.predict import predict_dataframe

    return predict_dataframe(frame.reset_index(drop=True), str(MODEL_PATH))


def engineered(frame: pd.DataFrame) -> pd.DataFrame:
    """The engineered feature frame the model sees (training-time thresholds applied)."""
    from src.preprocess import engineer_features_pandas

    artifact = load_artifact() or {}
    return engineer_features_pandas(frame.copy(), thresholds=artifact.get("feature_thresholds") or None)


def require_model() -> dict:
    """Stop the page with a styled notice when the model artifact is unavailable."""
    artifact = load_artifact()
    if artifact is None:
        callout(
            "<b>No model artifact.</b> Scoring needs <code>models/fraud_model.pkl</code>. "
            "Run <code>dvc repro train</code> (or <code>dvc pull</code>) and reload.",
            tone="warn",
        )
        st.stop()
    return artifact


def normalise_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an uploaded batch to the raw scoring schema; raises ValueError on a bad file."""
    cols = {c.strip().lower(): c for c in df.columns}
    if "amount" not in cols:
        raise ValueError("The file needs at least an `amount` column.")
    out = df.rename(columns={v: k for k, v in cols.items()}).copy()
    if "transaction_type" not in out and "category" in out:
        out["transaction_type"] = out["category"]
    if "source" not in out:
        out["source"] = np.where(out.get("transaction_type", pd.Series("", index=out.index))
                                 .astype(str).str.upper().isin(PAYSIM_TYPES), "paysim", "sparkov")
    defaults = {"transaction_type": "PAYMENT", "hour_of_day": 12, "day_of_month": 15,
                "balance_change_orig": np.nan, "balance_ratio": np.nan, "has_balance_info": 0}
    for col, value in defaults.items():
        if col not in out:
            out[col] = value
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[out["amount"].notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows with a numeric `amount`.")
    return out


# Example transactions for the Score page and the Overview teaser. PaySim rows
# carry the sender's balance before/after; Sparkov rows have no balance data.
PRESETS = {
    "Card purchase": dict(source="sparkov", transaction_type="grocery_pos", amount=64.20,
                          hour_of_day=18, day_of_month=12),
    "Late-night online": dict(source="sparkov", transaction_type="shopping_net", amount=949.00,
                              hour_of_day=23, day_of_month=3),
    "Account drain": dict(source="paysim", transaction_type="TRANSFER", amount=181_000.00,
                          before=181_000.00, after=0.00, hour_of_day=2, day_of_month=14),
    "Salary in": dict(source="paysim", transaction_type="CASH_IN", amount=42_000.00,
                      before=8_500.00, after=50_500.00, hour_of_day=10, day_of_month=1),
}


def to_row(source: str, transaction_type: str, amount: float, hour_of_day: int, day_of_month: int,
           before: float | None = None, after: float | None = None) -> pd.DataFrame:
    """One transaction in the raw scoring schema, mirroring src/combine_datasets.py."""
    if source == "paysim":
        before, after = float(before or 0.0), float(after or 0.0)
        change, ratio, has_info = before - after, after / (before + 1.0), 1
    else:
        change, ratio, has_info = np.nan, np.nan, 0
    return pd.DataFrame([{
        "amount": float(amount), "transaction_type": transaction_type,
        "hour_of_day": int(hour_of_day), "day_of_month": int(day_of_month),
        "balance_change_orig": change, "balance_ratio": ratio,
        "has_balance_info": has_info, "source": source,
    }])[RAW_COLUMNS]


def describe_row(row: pd.Series) -> str:
    """'Card · grocery in store · $64.20 · 18:00' style one-liner."""
    kind = "Card" if row["source"] == "sparkov" else "Mobile money"
    return (f"{kind} · {type_label(row['transaction_type'])} · ${row['amount']:,.2f} · "
            f"{int(row['hour_of_day']):02d}:00")


def demo_batch(n: int = 480, seed: int = 7) -> pd.DataFrame:
    """A seeded synthetic batch in the raw schema — PaySim-style and Sparkov-style rows.

    Mostly ordinary activity plus a small share of the patterns fraud tends to
    take in each source (an account emptied by a night-time transfer, a large
    late-night online card purchase). Synthetic, not sampled from the datasets.
    """
    rng = np.random.default_rng(seed)
    n_pay = int(n * 0.5)
    n_card = n - n_pay

    hours_day = np.r_[np.arange(7, 23)]
    hours_night = np.r_[np.arange(0, 6), 22, 23]

    # PaySim-style mobile money
    p_type = rng.choice(PAYSIM_TYPES, size=n_pay, p=[0.16, 0.30, 0.33, 0.18, 0.03])
    p_amount = np.round(rng.lognormal(9.0, 1.3, n_pay), 2)
    before = np.round(p_amount * rng.uniform(1.1, 8.0, n_pay), 2)
    incoming = p_type == "CASH_IN"
    after = np.where(incoming, before + p_amount, np.maximum(before - p_amount, 0.0))
    p_hour = rng.choice(hours_day, size=n_pay)
    drain = rng.random(n_pay) < 0.07
    p_type = np.where(drain, rng.choice(["TRANSFER", "CASH_OUT"], size=n_pay), p_type)
    p_amount = np.where(drain, np.round(rng.lognormal(12.0, 0.8, n_pay), 2), p_amount)
    before = np.where(drain, p_amount, before)
    after = np.where(drain, 0.0, after)
    p_hour = np.where(drain & (rng.random(n_pay) < 0.7), rng.choice(hours_night, size=n_pay), p_hour)
    paysim = pd.DataFrame({
        "amount": p_amount,
        "transaction_type": p_type,
        "hour_of_day": p_hour,
        "day_of_month": rng.integers(1, 32, n_pay),
        "balance_change_orig": np.round(before - after, 2),
        "balance_ratio": np.round(after / (before + 1.0), 4),
        "has_balance_info": 1,
        "source": "paysim",
    })

    # Sparkov-style card purchases
    c_type = rng.choice(SPARKOV_CATEGORIES, size=n_card)
    c_amount = np.round(rng.lognormal(3.9, 1.0, n_card), 2)
    c_hour = rng.choice(hours_day, size=n_card)
    odd = rng.random(n_card) < 0.06
    c_type = np.where(odd, rng.choice(["shopping_net", "misc_net", "grocery_pos"], size=n_card), c_type)
    c_amount = np.where(odd, np.round(rng.uniform(280, 1200, n_card), 2), c_amount)
    c_hour = np.where(odd, rng.choice(hours_night, size=n_card), c_hour)
    sparkov = pd.DataFrame({
        "amount": c_amount,
        "transaction_type": c_type,
        "hour_of_day": c_hour,
        "day_of_month": rng.integers(1, 32, n_card),
        "balance_change_orig": np.nan,
        "balance_ratio": np.nan,
        "has_balance_info": 0,
        "source": "sparkov",
    })

    batch = pd.concat([paysim, sparkov], ignore_index=True)
    return batch.sample(frac=1.0, random_state=seed).reset_index(drop=True)[RAW_COLUMNS]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_FEATURE_LABELS = {
    "amount": "Amount",
    "tx_amount_log": "Amount (log)",
    "amount_zscore": "Amount z-score",
    "is_high_amount": "Amount over 200k",
    "is_p95_amount": "Amount above p95",
    "is_p99_amount": "Amount above p99",
    "hour_of_day": "Hour of day",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "day_of_month": "Day of month",
    "day_sin": "Day of month (sin)",
    "day_cos": "Day of month (cos)",
    "is_night": "Night-time, 22:00–06:00",
    "is_weekend": "Weekend",
    "balance_change_orig": "Sender balance change",
    "balance_change_abs": "Balance change (abs)",
    "balance_change_log": "Balance change (log)",
    "balance_ratio": "Balance ratio",
    "balance_ratio_clipped": "Balance ratio (clipped)",
    "has_balance_info": "Balance info present",
}
_GROUPS = {"transaction_type_": "Transaction type / category", "source_": "Source", "amount_bin_": "Amount bin"}


def feature_label(name: str) -> str:
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    for prefix, group in _GROUPS.items():
        if name.startswith(prefix):
            return f"{group.split(' /')[0]} · {name[len(prefix):].replace('_', ' ')}"
    return name.replace("_", " ")


def feature_group(name: str) -> str:
    for prefix, group in _GROUPS.items():
        if name.startswith(prefix):
            return group
    return feature_label(name)


def type_label(value: str) -> str:
    """PaySim types stay upper-case; Sparkov categories read as words."""
    value = str(value)
    return value if value.isupper() else value.replace("_net", " · online").replace("_pos", " · in store").replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# HTML components (classes styled in ui_theme.components_css)
# ---------------------------------------------------------------------------


def pct(p: float) -> str:
    """A probability as a percent string that doesn't round to 0 or 100 at the extremes."""
    v = float(p) * 100
    return f"{v:.2f}" if v < 1 or v >= 99.9 else f"{v:.1f}"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def raw(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, lede: str = "") -> None:
    """Mono eyebrow, serif title (may contain <em>), one-line lede."""
    lede_html = f'<p class="ph-lede">{lede}</p>' if lede else ""
    raw(f'<header class="ph"><div class="eyebrow">{esc(eyebrow)}</div>'
          f'<h1 class="ph-title">{title}</h1>{lede_html}</header>')


def section(index: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="sec-sub">{sub}</p>' if sub else ""
    raw(f'<div class="sec"><div class="sec-index">{esc(index)}</div>'
          f'<h2 class="sec-title">{title}</h2>{sub_html}</div>')


def stat_grid(items: list[dict], cols: int = 4) -> None:
    """Row of stat tiles. Each item: label, value, unit?, foot?, tone? (accent|ok|bad|ink)."""
    tiles = []
    for it in items:
        unit = f'<span class="stat-unit">{esc(it["unit"])}</span>' if it.get("unit") else ""
        foot = f'<div class="stat-foot">{it["foot"]}</div>' if it.get("foot") else ""
        tone = it.get("tone", "accent")
        tiles.append(f'<div class="stat stat-{tone}"><div class="stat-label">{esc(it["label"])}</div>'
                     f'<div class="stat-value">{esc(it["value"])}{unit}</div>{foot}</div>')
    raw(f'<div class="stat-grid" style="--cols:{cols}">{"".join(tiles)}</div>')


def callout(body_html: str, tone: str = "info", title: str = "") -> None:
    head = f'<div class="callout-title">{esc(title)}</div>' if title else ""
    raw(f'<div class="callout callout-{tone}">{head}<div class="callout-body">{body_html}</div></div>')


def chips(items: list[str]) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{esc(i)}</span>' for i in items) + "</div>"


def footer(text: str) -> None:
    raw(f'<footer class="pg-foot"><span>{text}</span>'
          f'<a href="{REPO_URL}" target="_blank" rel="noopener">Source on GitHub ↗</a></footer>')


def card(key: str):
    """Bordered container styled as a card (ui_theme targets the st-key-card_* class)."""
    return st.container(border=True, key=f"card_{key}")


def card_title(label: str, title: str = "") -> None:
    """Small heading inside a bordered container."""
    title_html = f'<div class="card-title">{title}</div>' if title else ""
    raw(f'<div class="card-head"><div class="eyebrow">{esc(label)}</div>{title_html}</div>')
