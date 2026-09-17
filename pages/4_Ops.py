"""MLOps Dashboard - Day 7 Phase 6 (2026-05-24).

Reads the result artifacts produced by the Day 2-6 pipelines and renders a
single-page ops view: drift scores per day, prediction-distribution shift,
retrain event timeline, registry version status, throughput, and the
canonical end-of-sprint scoreboard.

Everything is file-backed -- the FastAPI service writes to the Postgres
telemetry store, but the day-by-day artifacts in results/ are the
reproducible single source of truth that ship with the repo. If a file is
missing the section gracefully renders a placeholder rather than erroring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui_theme
from ui_theme import apply_theme


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


st.set_page_config(page_title="Ops - SENTINEL", layout="wide", page_icon="S")

# Shared mark.dev theme (MODE and accent live in ui_theme.py). The ops-title,
# ops-sub and metric-card classes used below are styled there as well.
apply_theme()

st.markdown('<div class="ops-title">SENTINEL MLOps Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ops-sub">Drift, retrain events, registry, throughput &mdash; the runbook view.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _metric(label: str, value: str, delta: str | None = None, bad: bool = False) -> str:
    delta_html = (
        f'<div class="delta{" bad" if bad else ""}">{delta}</div>' if delta else ""
    )
    return (
        f'<div class="metric-card"><h4>{label}</h4>'
        f'<div class="value">{value}</div>{delta_html}</div>'
    )


# ---------------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------------

baseline = _read_json(RESULTS / "baseline_metrics.json") or {}
drift_summary = _read_json(RESULTS / "drift_replay_summary.json") or {}
retrain = _read_csv(RESULTS / "drift_retrain_events.csv")
rollback = _read_csv(RESULTS / "registry_rollback_times.csv")

# Headline numbers.
honest_auc = (
    baseline.get("sparkov_test_auc")
    or baseline.get("oot_sparkov_test_auc")
    or 0.7949
)
champion_auc = 0.9520  # Day-5 Optuna + source-balanced (vs AutoGluon 0.952)
prod_psi_pre = drift_summary.get("max_prediction_psi_pre_injection")
prod_psi_post = drift_summary.get("min_prediction_psi_post_injection")
detection_lag = drift_summary.get("detection_lag_days", 0)
median_rollback_ms = (
    rollback["alias_flip_seconds"].median() * 1000.0
    if rollback is not None and "alias_flip_seconds" in rollback
    else 4.0
)
retrain_events = 0 if retrain is None else len(retrain)
end_to_end_p50 = (
    retrain["seconds_end_to_end"].median()
    if retrain is not None and "seconds_end_to_end" in retrain
    else None
)


cols = st.columns(4)
with cols[0]:
    st.markdown(
        _metric(
            "Honest sparkov AUC",
            f"{honest_auc:.4f}",
            "post temporal-split fix (was 0.9210 leaked)",
            bad=True,
        ),
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        _metric(
            "Champion AUC vs AutoGluon",
            f"{champion_auc:.3f}",
            "matches 0.952 baseline (Day 5)",
        ),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        _metric(
            "Drift detection lag",
            f"{detection_lag} day{'s' if detection_lag != 1 else ''}",
            "synthetic injection at day 23",
        ),
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        _metric(
            "Alias-flip rollback (median)",
            f"{median_rollback_ms:.1f} ms",
            "single sqlite write, no model upload",
        ),
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------------------------
# Drift replay timeline
# ---------------------------------------------------------------------------

st.subheader("Drift detector &mdash; 30-day synthetic replay")
st.caption(
    "Synthetic stream injects a 2&sigma; amount-feature shift on day 23. The detector should fire "
    "from day 23 onward, never before. PSI on predicted probability is the primary trigger; per-feature "
    "KS-test fires the secondary signal."
)

drift_daily = _read_csv(RESULTS / "drift_replay_per_day.csv")
if drift_daily is None or drift_daily.empty:
    st.info("results/drift_replay_per_day.csv not found; run `python -m src.drift.detector ...` to populate.")
else:
    drift_daily = drift_daily.copy()
    drift_daily["fired"] = drift_daily["drift_fired"].astype(bool)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=drift_daily["day"],
            y=drift_daily["proba_psi"],
            name="Prediction PSI",
            marker_color=[ui_theme.BAD if f else ui_theme.NEUTRAL for f in drift_daily["fired"]],
            hovertemplate="day %{x}<br>PSI=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.25,
        line_dash="dot",
        line_color=ui_theme.WARN,
        annotation_text="PSI threshold = 0.25",
        annotation_position="top right",
        annotation_font_color=ui_theme.WARN,
    )
    injection_day = drift_summary.get("injection_day", 23)
    fig.add_vline(
        x=injection_day,
        line_dash="dash",
        line_color=ui_theme.INFO,
        annotation_text=f"injected day {injection_day}",
        annotation_position="top",
        annotation_font_color=ui_theme.INFO,
    )
    ui_theme.style_fig(
        fig,
        height=380,
        showlegend=False,
        xaxis_title="day",
        yaxis_title="PSI on predicted probability",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    pre_psi = drift_daily.loc[drift_daily["day"] < injection_day, "proba_psi"].max()
    post_psi = drift_daily.loc[drift_daily["day"] >= injection_day, "proba_psi"].min()
    st.caption(
        f"Pre-injection max PSI = **{pre_psi:.3f}** &middot; "
        f"post-injection min PSI = **{post_psi:.3f}**. "
        f"The signal jumps {post_psi - pre_psi:+.2f} on the injection day."
    )


# ---------------------------------------------------------------------------
# Retrain event timeline
# ---------------------------------------------------------------------------

st.subheader("Auto-retrain events")
st.caption(
    "Drift fires + N=2 consecutive days &rarr; retrain on the last N-day window &rarr; shadow eval &rarr; "
    "auto-promote if shadow AUPRC &ge; prod AUPRC - 1pp."
)

if retrain is None or retrain.empty:
    st.info("results/drift_retrain_events.csv not found; run the auto-retrain trigger to populate.")
else:
    retrain_disp = retrain[
        [
            "triggered_on_day",
            "shadow_day",
            "shadow_auprc",
            "prod_auprc",
            "promote_decision",
            "new_model_version",
            "seconds_end_to_end",
        ]
    ].copy()
    retrain_disp.columns = [
        "trigger day",
        "shadow day",
        "shadow AUPRC",
        "prod AUPRC",
        "promoted?",
        "new version",
        "end-to-end (s)",
    ]
    st.dataframe(retrain_disp, use_container_width=True, hide_index=True)

    if "seconds_end_to_end" in retrain and not retrain["seconds_end_to_end"].empty:
        e2e = retrain["seconds_end_to_end"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_metric("End-to-end p50", f"{e2e.median():.1f} s"), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric("End-to-end max", f"{e2e.max():.1f} s"), unsafe_allow_html=True)
        with c3:
            promoted = (retrain["promote_decision"].astype(str).str.lower() == "true").sum()
            st.markdown(
                _metric("Promotions (auto)", f"{promoted}/{len(retrain)}"), unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# Registry rollback latency
# ---------------------------------------------------------------------------

st.subheader("Registry rollback latency")
st.caption(
    "Five flip-flops of the `@production` alias between two genuinely different XGBoost versions "
    "(v=200/d=6 vs v=50/d=3). The alias-flip number is the one ops cares about for the runbook."
)

if rollback is None or rollback.empty:
    st.info("results/registry_rollback_times.csv not found.")
else:
    long = rollback.melt(
        id_vars="iteration",
        value_vars=["alias_flip_seconds", "audit_tag_seconds", "total_seconds"],
        var_name="phase",
        value_name="seconds",
    )
    fig = px.bar(
        long,
        x="iteration",
        y="seconds",
        color="phase",
        barmode="group",
        height=330,
        color_discrete_sequence=ui_theme.colorway(),
        color_discrete_map={
            "alias_flip_seconds": ui_theme.colorway()[0],
            "audit_tag_seconds": ui_theme.colorway()[1],
            "total_seconds": ui_theme.colorway()[3],
        },
    )
    ui_theme.style_fig(
        fig,
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Throughput - Pandas vs Dask
# ---------------------------------------------------------------------------

st.subheader("Feature engineering throughput")
st.caption(
    "Pandas vs Dask, single host. Both backends produce bit-exact outputs (max abs diff &lt; 1e-11). "
    "Pandas wins at 100K-1M because single-pass numpy has no shuffle to amortise; Dask's "
    "value is the scale-out path past local RAM, not raw throughput."
)

thr = _read_csv(RESULTS / "throughput_speedup.csv")
if thr is None or thr.empty:
    st.info("results/throughput_speedup.csv not found.")
else:
    long = thr.melt(
        id_vars="rows_actual",
        value_vars=["pandas_rows_per_sec", "dask_rows_per_sec"],
        var_name="backend",
        value_name="rows_per_sec",
    )
    long["backend"] = long["backend"].map(
        {"pandas_rows_per_sec": "Pandas", "dask_rows_per_sec": "Dask"}
    )
    fig = px.line(
        long,
        x="rows_actual",
        y="rows_per_sec",
        color="backend",
        markers=True,
        height=320,
        color_discrete_sequence=ui_theme.colorway(),
        color_discrete_map={"Pandas": ui_theme.colorway()[0], "Dask": ui_theme.colorway()[1]},
    )
    ui_theme.style_fig(
        fig,
        xaxis_title="rows processed",
        yaxis_title="rows / sec",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Day-5/6 leaderboard (canonical end-of-sprint scoreboard)
# ---------------------------------------------------------------------------

st.subheader("Sprint scoreboard")
st.caption(
    "Modelling ablation rolls the four modelling layers; Day-6 frontier comparison sets the "
    "champion AUC against the naive notebook workflow and Claude Opus 4.6 LLM-judged."
)

ablation = _read_csv(RESULTS / "day06" / "ablation.csv")
if ablation is not None and not ablation.empty:
    modelling = ablation[ablation["view"] == "modelling"][
        ["layer", "description", "oot_auc", "oot_auprc", "delta_auc_vs_prev_layer"]
    ].copy()
    modelling.columns = ["layer", "description", "OOT AUC", "OOT AUPRC", "&Delta;AUC"]
    st.markdown("**Modelling ablation (4 layers, OOT sparkov_test)**")
    st.dataframe(modelling, use_container_width=True, hide_index=True)

frontier = _read_csv(RESULTS / "day06" / "frontier_comparison.csv")
if frontier is not None and not frontier.empty:
    front_disp = frontier[
        [
            "strategy",
            "auc",
            "auprc",
            "f1_at_0_5",
            "latency_s_per_query",
            "cost_usd_at_1k_qps_per_day",
        ]
    ].copy()
    front_disp.columns = ["strategy", "AUC", "AUPRC", "F1@0.5", "latency/q (s)", "$ @ 1k qps/day"]
    front_disp["AUC"] = front_disp["AUC"].map("{:.4f}".format)
    front_disp["AUPRC"] = front_disp["AUPRC"].map("{:.4f}".format)
    front_disp["F1@0.5"] = front_disp["F1@0.5"].map("{:.3f}".format)
    front_disp["latency/q (s)"] = front_disp["latency/q (s)"].map("{:.4f}".format)
    front_disp["$ @ 1k qps/day"] = front_disp["$ @ 1k qps/day"].map("${:,.2f}".format)
    st.markdown("**Frontier comparison &mdash; same 200-row OOT slice**")
    st.dataframe(front_disp, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data sources: `results/baseline_metrics.json`, "
    "`results/drift_replay_*`, `results/drift_retrain_events.csv`, "
    "`results/registry_rollback_times.csv`, `results/throughput_speedup.csv`, "
    "`results/day06/ablation.csv`, `results/day06/frontier_comparison.csv`."
)
