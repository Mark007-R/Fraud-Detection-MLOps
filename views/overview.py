"""Overview — what SENTINEL is, the measured results, and how the loop runs."""

import plotly.graph_objects as go
import streamlit as st

import ui_kit as kit
import ui_theme

h = kit.headline()

# ---- hero ---------------------------------------------------------------------
left, right = st.columns([1.35, 1], gap="large", vertical_alignment="center")

with left:
    kit.raw(
        '<div class="hero"><div class="eyebrow"><span class="pulse"></span>Fraud detection · MLOps pipeline</div>'
        '<h1 class="hero-title">Fraud scoring that <em>keeps itself honest</em>.</h1>'
        '<p class="hero-lede">An XGBoost model trained on 6.1M PaySim and Sparkov transactions, served '
        'behind FastAPI and watched by a drift monitor. When the data moves it retrains, tests the '
        'candidate in shadow, and promotes it only if it wins. Any version rolls back in milliseconds.</p></div>'
    )
    with st.container(key="hero_cta", horizontal=True):
        st.page_link("views/score.py", label="Score a transaction", icon=":material/arrow_forward:")
        st.page_link("views/ops.py", label="See the ops loop", icon=":material/monitor_heart:")

with right:
    artifact = kit.load_artifact()
    rows = []
    if artifact is not None:
        for name in ("Card purchase", "Late-night online", "Account drain"):
            frame = kit.to_row(**kit.PRESETS[name])
            prob = float(kit.score(frame)["fraud_probability"].iloc[0])
            flagged = prob >= kit.THRESHOLD
            rows.append(
                f'<div class="live-row {"is-flag" if flagged else "is-clear"}"><div class="live-meta">'
                f'<div class="live-name">{kit.esc(name)}</div>'
                f'<div class="live-desc">{kit.esc(kit.describe_row(frame.iloc[0]))}</div></div>'
                f'<div class="live-score">{kit.pct(prob)}%'
                f'<span>{"flag" if flagged else "clear"}</span></div>'
                f'<div class="live-bar"><i style="width:{max(prob * 100, 1.5):.1f}%"></i></div></div>'
            )
        kit.raw(
            '<div class="live-card"><div class="live-head"><span class="eyebrow">Scored just now</span>'
            '<span class="live-tag">bundled model · threshold 0.50</span></div>'
            + "".join(rows) + "</div>"
        )
    else:
        kit.callout("The scoring teaser appears once <code>models/fraud_model.pkl</code> is present.")

# ---- headline numbers -----------------------------------------------------------
kit.stat_grid([
    {"label": "Out-of-time AUC", "value": f"{h['oot_auc']:.3f}",
     "foot": f"Level with AutoGluon ({h['autogluon_auc']:.3f}) on 555,719 later Sparkov transactions"},
    {"label": "Discipline lift", "value": f"+{h['lift']:.3f}", "unit": "AUC",
     "foot": f"Temporal split, source weights and tuning: {h['l0_auc']:.3f} → {h['l3_auc']:.3f}"},
    {"label": "Registry rollback", "value": f"{h['rollback_ms']:.1f}", "unit": "ms",
     "foot": f"Median alias flip · {h['rollback_audited_ms']:.1f} ms with the audit tag"},
    {"label": "Drift detection lag", "value": f"{h['drift_lag']}", "unit": "days",
     "foot": f"Precision {h['drift_precision']:.1f} · recall {h['drift_recall']:.1f} on a 2σ shift"},
])

# ---- the loop -------------------------------------------------------------------
kit.section("01", "How the loop runs",
            "Eight stages, each reproducible from a clean checkout. The last one feeds back into the fifth.")

steps = [
    ("Ingest", "PaySim and Sparkov pulled through a DVC pipeline."),
    ("Features", "Same code in pandas or Dask, bit-exact to 5.5e-12."),
    ("Split", "Per source, chronological, final 20% held out. A test fails on a random split."),
    ("Train", "XGBoost with source-balanced weights and a 30-trial Optuna sweep, logged to MLflow."),
    ("Register", "The winner takes the production alias. Rollback is the same move in reverse."),
    ("Serve", "FastAPI resolves the alias at startup and scores in about 60 µs."),
    ("Monitor", "KS and PSI over eight features, every day."),
    ("Retrain", "Two drift days trigger a retrain, a shadow eval, and promotion only if it wins."),
]
kit.raw(
    '<ol class="flow">'
    + "".join(f'<li class="flow-step{" is-loop" if i == 8 else ""}"><span class="flow-n">{i:02d}</span>'
              f'<span class="flow-t">{kit.esc(t)}</span><span class="flow-d">{kit.esc(d)}</span></li>'
              for i, (t, d) in enumerate(steps, start=1))
    + '</ol><div class="flow-note">↺ Retrain registers a new version and hands it back to stage 05</div>'
)

# ---- discipline ablation ----------------------------------------------------------
kit.section("02", "The discipline is the model",
            "Same XGBoost, same out-of-time Sparkov slice. Each layer of process adds measurable AUC.")

ablation = kit.read_csv("results/day06/ablation_modelling.csv")
text_col, chart_col = st.columns([1, 1.5], gap="large")
with text_col:
    kit.raw(
        '<div class="prose">'
        f'<p>The first version of this code used a random split on time-series data. It scored '
        f'<b>{h["leaky_auc"]:.3f}</b>, because future transactions leaked into training. Splitting each '
        f'source chronologically dropped the honest number to <b>{h["honest_auc"]:.3f}</b>.</p>'
        '<p>Source-balanced weights stop PaySim (83% of rows) from drowning Sparkov, and an Optuna sweep '
        f'does the rest. From a naive notebook to the champion is <b>+{h["lift"]:.3f} AUC</b>, and tuning '
        'only pays off once the split is honest.</p></div>'
    )
with chart_col:
    if ablation is not None:
        labels = ["Naive notebook", "+ temporal split", "+ source weights", "+ Optuna tuning", "Champion"]
        deltas = ablation["delta_auc_vs_prev_layer"].fillna(0).tolist()
        y = [float(ablation["oot_auc"].iloc[0])] + [float(d) for d in deltas[1:]] + [0]
        fig = go.Figure(go.Waterfall(
            x=labels, y=y, measure=["absolute", "relative", "relative", "relative", "total"],
            text=[f"{y[0]:.3f}"] + [f"+{d:.3f}" for d in y[1:4]] + [f"{float(ablation['oot_auc'].iloc[-1]):.3f}"],
            textposition="outside", textfont=dict(color=ui_theme.INK, size=13),
            increasing=dict(marker=dict(color=ui_theme.ACCENT)),
            totals=dict(marker=dict(color=ui_theme.ACCENT_SOFT)),
            connector=dict(line=dict(color=ui_theme.LINE_STRONG, width=1, dash="dot")),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))
        ui_theme.style_fig(fig, height=340, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis=dict(range=[0.4, 1.02], title="Out-of-time AUC",
                                      zeroline=False, tickformat=".2f"),
                           xaxis=dict(showgrid=False, linecolor=ui_theme.LINE_STRONG))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

# ---- specialist vs frontier ----------------------------------------------------------
kit.section("03", "A specialist beats a frontier LLM",
            "The same 200 out-of-time rows, scored three ways. Cost assumes 1,000 queries per second for a day.")

frontier = kit.read_csv("results/day06/frontier_comparison.csv")
if frontier is not None:
    def _latency(s: float) -> str:
        return f"{s * 1e6:.0f} µs" if s < 1e-3 else f"{s:.2f} s"

    rows = []
    for _, r in frontier.iterrows():
        name = kit.strategy_label(r["strategy"])
        champ = name == "SENTINEL champion"
        rows.append(
            f'<div class="vs-row{" is-champ" if champ else ""}"><div class="vs-name">{kit.esc(name)}</div>'
            f'<div class="vs-metric"><div class="vs-bar"><i style="width:{r["auc"] * 100:.1f}%"></i></div>'
            f'<span>AUC {r["auc"]:.3f}</span></div>'
            f'<div class="vs-num"><b>{_latency(r["latency_s_per_query"])}</b><span>per query</span></div>'
            f'<div class="vs-num"><b>${r["cost_usd_at_1k_qps_per_day"]:,.2f}</b><span>per day</span></div></div>'
        )
    kit.raw('<div class="vs">' + "".join(rows) + "</div>")
    kit.raw('<p class="fine">The LLM is about 30,000× slower and 2.9 million× more expensive at the same '
            'throughput, and it ranks worse. The naive XGBoost roughly ties it: the gap comes from the '
            'discipline layers, not the algorithm.</p>')

# ---- stack -------------------------------------------------------------------------------
kit.section("04", "Built with")
kit.raw(kit.chips(["DVC", "XGBoost", "Optuna", "MLflow registry", "pandas · Dask", "FastAPI",
                   "Streamlit", "Postgres", "Redis", "Docker Compose", "GitHub Actions"]))

kit.footer("Numbers on this page are read from the committed artifacts in results/")
