"""Model — evaluation of the production artifact: headline metrics, ROC, errors, features."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_kit as kit
import ui_theme

kit.page_header("Model", "How the model <em>performs</em>",
                "How models/fraud_model.pkl, the DVC train-stage model, scores on the combined temporal hold-out.")

m = kit.read_json("metrics/scores.json")
if not m:
    kit.callout("No metrics yet. Run <code>dvc repro evaluate</code> to write <code>metrics/scores.json</code>.",
                tone="warn")
    st.stop()

h = kit.headline()
tn, fp = int(m.get("true_negatives", 0)), int(m.get("false_positives", 0))
fn, tp = int(m.get("false_negatives", 0)), int(m.get("true_positives", 0))
neg, pos = tn + fp, fn + tp
total = neg + pos

kit.callout(
    f"Scored on <b>{total:,}</b> held-out transactions: the latest 20% of each source by time, fraud rate "
    f"<b>{pos / max(total, 1):.2%}</b>. PaySim is 83% of this set and its balance columns make fraud close to "
    f"deterministic, so the AUC here runs high. On Sparkov's later, never-seen window (Jun–Dec 2020) this same "
    f"artifact scores <b>{h['honest_auc']:.3f}</b>; the tuned champion reaches <b>{h['oot_auc']:.3f}</b> (Overview).",
    title="Read this first",
)

kit.stat_grid([
    {"label": "AUC-ROC", "value": f"{m.get('auc_roc', 0):.4f}", "foot": "Ranking quality across all thresholds"},
    {"label": "Average precision", "value": f"{m.get('average_precision', 0):.3f}",
     "foot": "Area under the precision-recall curve"},
    {"label": "Recall", "value": f"{m.get('recall', 0) * 100:.1f}", "unit": "%",
     "foot": f"{tp:,} of {pos:,} frauds caught at 0.50"},
    {"label": "Precision", "value": f"{m.get('precision', 0) * 100:.1f}", "unit": "%",
     "foot": f"{tp:,} of {tp + fp:,} flags are fraud · F1 {m.get('f1_score', 0):.3f}"},
], cols=4)

# ---- ROC + confusion matrix ----------------------------------------------------------------
roc_col, cm_col = st.columns([1.25, 1], gap="large")
with roc_col:
    with kit.card("roc"):
        kit.card_title("ROC curve", f"AUC {m.get('auc_roc', 0):.4f}")
        fpr, tpr = m.get("fpr", []), m.get("tpr", [])
        if fpr and tpr:
            log_x = st.toggle("Log-scale false-positive rate", value=True,
                              help="At a 0.38% fraud rate the useful part of the curve is the far left.")
            # the chance diagonal needs many points to stay y = x on a log x-axis
            chance = np.logspace(-6, 0, 80) if log_x else np.array([0.0, 1.0])
            fig = go.Figure([
                go.Scatter(x=[max(x, 1e-6) for x in fpr] if log_x else fpr, y=tpr, mode="lines", name="model",
                           line=dict(color=ui_theme.ACCENT, width=3), fill="tozeroy",
                           fillcolor=ui_theme._rgba(ui_theme.ACCENT, 0.10),
                           hovertemplate="FPR %{x:.4%}<br>TPR %{y:.3f}<extra></extra>"),
                go.Scatter(x=chance, y=chance, mode="lines",
                           name="random", line=dict(color=ui_theme.INK_3, width=1.5, dash="dash"),
                           hoverinfo="skip"),
            ])
            op_fpr = fp / max(neg, 1)
            fig.add_trace(go.Scatter(
                x=[max(op_fpr, 1e-6)], y=[m.get("recall", 0)], mode="markers", name="threshold 0.50",
                marker=dict(size=11, color=ui_theme.PAPER, line=dict(color=ui_theme.ACCENT, width=2.5)),
                hovertemplate="Operating point<br>FPR %{x:.3%} · recall %{y:.3f}<extra></extra>"))
            ui_theme.style_fig(
                fig, height=360, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.22, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=ui_theme.INK_2)),
                xaxis=dict(title="false-positive rate" + (" (log)" if log_x else ""), type="log" if log_x else "linear",
                           range=[-5, 0] if log_x else [0, 1], tickformat=None if log_x else ".0%",
                           tickvals=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] if log_x else None,
                           ticktext=["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"] if log_x else None),
                yaxis=dict(title="true-positive rate", range=[0, 1.02]),
            )
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
        else:
            kit.callout("No ROC points in the metrics file. Re-run evaluation to add them.")

with cm_col:
    with kit.card("errors"):
        kit.card_title("At threshold 0.50", "Where the errors land")
        cells = [
            ("Legit, cleared", tn, f"{tn / max(neg, 1):.2%} of legit", "tn"),
            ("Legit, flagged", fp, f"{fp / max(neg, 1):.3%} false alarms", "fp"),
            ("Fraud, missed", fn, f"{fn / max(pos, 1):.1%} of fraud", "fn"),
            ("Fraud, caught", tp, f"{tp / max(pos, 1):.1%} of fraud", "tp"),
        ]
        kit.raw(
            '<div class="cm"><span class="cm-axis cm-pred">predicted →</span>'
            '<div class="cm-head"></div><div class="cm-head">clear</div><div class="cm-head">flag</div>'
            '<div class="cm-side">legit</div>'
            + "".join(f'<div class="cm-cell cm-{k}"><b>{v:,}</b><span>{kit.esc(lbl)}</span><small>{kit.esc(s)}</small></div>'
                      for lbl, v, s, k in cells[:2])
            + '<div class="cm-side">fraud</div>'
            + "".join(f'<div class="cm-cell cm-{k}"><b>{v:,}</b><span>{kit.esc(lbl)}</span><small>{kit.esc(s)}</small></div>'
                      for lbl, v, s, k in cells[2:])
            + "</div>"
        )
        kit.raw(f'<p class="fine">About {fp / max(tp, 1):.1f} false alarms per fraud caught. Accuracy is '
                f'{m.get("accuracy", 0):.2%}, which says little when 99.6% of rows are legitimate.</p>')

# ---- feature importance ---------------------------------------------------------------------
kit.section("Features", "What the model leans on", "XGBoost gain importance from the trained artifact.")

importance = m.get("feature_importance", {})
if importance:
    grouped = st.toggle("Group one-hot columns", value=True,
                        help="Sum the per-category columns into one bar per original field.")
    imp = pd.Series(importance, dtype=float)
    if grouped:
        imp = imp.groupby(kit.feature_group).sum()
    else:
        imp.index = [kit.feature_label(i) for i in imp.index]
    total_gain = imp.sum() or 1.0
    imp = imp[imp > 0].sort_values().tail(15)
    share = imp / total_gain * 100
    fig = go.Figure(go.Bar(
        y=imp.index, x=share, orientation="h",
        marker=dict(color=share, colorscale=ui_theme.bar_scale()),
        text=[f"{v:.1f}%" for v in share], textposition="outside", textfont=dict(color=ui_theme.INK_2),
        cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% of total gain<extra></extra>",
    ))
    ui_theme.style_fig(fig, height=max(320, 30 * len(imp)), showlegend=False,
                       margin=dict(l=10, r=40, t=10, b=10),
                       xaxis=dict(title="share of total gain (top 15 shown)", ticksuffix="%"),
                       yaxis=dict(showgrid=False, tickfont=dict(color=ui_theme.INK)))
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
    kit.raw('<p class="fine">Sender balance change is the strongest single column, and it is the PaySim '
            'signal that inflates this page\'s AUC. Sparkov has no balance columns, so on card data the model '
            'leans on merchant category, amount and time of day.</p>')
else:
    kit.callout("No feature importance in the metrics file. Re-run evaluation to add it.")

st.download_button("Download metrics/scores.json", json.dumps(m, indent=2), file_name="scores.json",
                   mime="application/json", icon=":material/download:")

kit.footer("Metrics written by src/evaluate.py · DVC stage <code>evaluate</code>")
