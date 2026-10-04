"""Ops — the runbook view: drift replay, auto-retrain, registry rollback, throughput.

Everything is file-backed. The FastAPI service writes live telemetry to
Postgres, but the artifacts in results/ are the reproducible record that ships
with the repo. A missing file renders a notice instead of an error.
"""

import plotly.graph_objects as go
import streamlit as st

import ui_kit as kit
import ui_theme

kit.page_header("Ops", "The loop, <em>measured</em>",
                "Drift, retraining, the registry and throughput, read from the committed artifacts in results/.")

h = kit.headline()
PLOT_CFG = {"displayModeBar": False}

kit.stat_grid([
    {"label": "Sparkov out-of-time AUC", "value": f"{h['oot_auc']:.3f}",
     "foot": f"Tuned champion, up from an honest {h['honest_auc']:.3f}. The leaky split read {h['leaky_auc']:.3f}"},
    {"label": "Drift detection lag", "value": f"{h['drift_lag']}", "unit": "days",
     "foot": f"Precision {h['drift_precision']:.1f} · recall {h['drift_recall']:.1f}"},
    {"label": "Auto-retrain, median", "value": f"{h['retrain_p50_s']:.2f}", "unit": "s",
     "foot": f"Detect → train → shadow → promote · {h['promoted']}/{h['retrains']} promoted"},
    {"label": "Rollback", "value": f"{h['rollback_ms']:.1f}", "unit": "ms",
     "foot": f"Median alias flip · {h['rollback_audited_ms']:.1f} ms audited"},
])

# ---- 01 drift replay -----------------------------------------------------------------
summary = kit.read_json("results/drift_replay_summary.json")
daily = kit.read_csv("results/drift_replay_per_day.csv")
inj = int(summary.get("injection_day", 23))
kit.section("01", "Drift replay, 30 days",
            f"A synthetic stream shifts <code>amount</code> by {summary.get('injection_sigma', 2.0):g}σ from day {inj}. "
            "The detector should fire from that day on and never before.")

if daily is None:
    kit.callout("<code>results/drift_replay_per_day.csv</code> not found. "
                "Run <code>python -m tests.synthetic_drift</code> to regenerate it.")
else:
    fired = daily["drift_fired"].astype(str).str.lower().eq("true")
    injected = daily["injected"].astype(str).str.lower().eq("true")
    cells = []
    for d, f, i, psi in zip(daily["day"], fired, injected, daily["proba_psi"]):
        state = "hit" if f and i else "miss" if i else "false" if f else "quiet"
        cells.append(f'<div class="day day-{state}" title="day {d} · PSI {psi:.3f}"><span>{d}</span></div>')
    kit.raw(
        '<div class="days">' + "".join(cells) + "</div>"
        '<div class="days-legend"><span><i class="day-quiet"></i>quiet day</span>'
        '<span><i class="day-hit"></i>shift injected · detector fired</span>'
        '<span><i class="day-miss"></i>missed</span><span><i class="day-false"></i>false alarm</span></div>'
    )

    with kit.card("psi"):
        kit.card_title("Prediction PSI per day", "Population stability of the model's output (log scale)")
        fig = go.Figure(go.Bar(
            x=daily["day"], y=daily["proba_psi"],
            marker_color=[ui_theme.ACCENT if f else ui_theme.NEUTRAL for f in fired],
            customdata=daily["n_rows"],
            hovertemplate="day %{x}<br>PSI %{y:.3f} · %{customdata:,} rows<extra></extra>",
        ))
        threshold = summary.get("detector_params", {}).get("psi_threshold", 0.25)
        fig.add_hline(y=threshold, line_dash="dot", line_color=ui_theme.WARN,
                      annotation_text=f"retrain threshold {threshold}", annotation_position="top left",
                      annotation_font=dict(color=ui_theme.WARN, size=12))
        fig.add_vline(x=inj - 0.5, line_dash="dash", line_color=ui_theme.INK_3,
                      annotation_text=f"shift injected · day {inj}", annotation_position="top left",
                      annotation_font=dict(color=ui_theme.INK_2, size=12))
        ui_theme.style_fig(fig, height=330, showlegend=False, margin=dict(l=10, r=10, t=30, b=10), bargap=0.25,
                           xaxis=dict(title="day", dtick=2, showgrid=False),
                           yaxis=dict(title="PSI", type="log", dtick=1))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)
        pre = daily.loc[daily["day"] < inj, "proba_psi"].max()
        post = daily.loc[daily["day"] >= inj, "proba_psi"].min()
        kit.raw(f'<p class="fine">Before the shift PSI never passes <b>{pre:.3f}</b>; from day {inj} it never '
                f'drops below <b>{post:.3f}</b>, about {post / pre:.0f}× higher. Five amount-derived features '
                'also fail the KS test on every drift day.</p>')

# ---- 02 auto-retrain ------------------------------------------------------------------------
kit.section("02", "Auto-retrain events",
            "Two consecutive drift days trigger a retrain on that window, a shadow evaluation on the next day, "
            "and promotion only if shadow AUPRC is within 0.01 of production.")

retrain = kit.read_csv("results/drift_retrain_events.csv")
if retrain is None:
    kit.callout("<code>results/drift_retrain_events.csv</code> not found. Run the auto-retrain trigger to populate it.")
else:
    cards = []
    for _, r in retrain.iterrows():
        promoted = str(r["promote_decision"]).lower() == "true"
        top = max(r["shadow_auprc"], r["prod_auprc"], 1e-9)
        cards.append(
            f'<div class="event"><div class="event-top"><span class="eyebrow">Day {int(r["triggered_on_day"])}</span>'
            f'<span class="event-pill {"ok" if promoted else "bad"}">{"promoted" if promoted else "kept prod"} '
            f'→ v{int(r["new_model_version"])}</span></div>'
            f'<div class="event-row"><span>shadow</span><div class="event-bar is-new"><i style="width:{r["shadow_auprc"] / top * 100:.0f}%"></i></div>'
            f'<b>{r["shadow_auprc"]:.3f}</b></div>'
            f'<div class="event-row"><span>prod</span><div class="event-bar"><i style="width:{r["prod_auprc"] / top * 100:.0f}%"></i></div>'
            f'<b>{r["prod_auprc"]:.3f}</b></div>'
            f'<div class="event-foot"><span>AUPRC on day {int(r["shadow_day"])} · {int(r["train_rows"]):,} train rows</span>'
            f'<b>{r["seconds_end_to_end"]:.1f} s</b></div></div>'
        )
    kit.raw('<div class="events">' + "".join(cards) + "</div>")

    with kit.card("stages"):
        kit.card_title("Where the time goes", "Seconds per stage, each retrain event")
        stages = [("seconds_detect_to_retrain_start", "detect → start"), ("seconds_train", "train"),
                  ("seconds_shadow_eval", "shadow eval"), ("seconds_register_and_alias", "register + alias")]
        palette = [ui_theme.INK_3, ui_theme.ACCENT, ui_theme.ACCENT_SOFT, ui_theme.NEUTRAL]
        labels = [f"day {int(d)} → v{int(v)}" for d, v in zip(retrain["triggered_on_day"], retrain["new_model_version"])]
        fig = go.Figure([go.Bar(y=labels, x=retrain[col], name=name, orientation="h", marker_color=c,
                                hovertemplate=name + ": %{x:.3f} s<extra></extra>")
                         for (col, name), c in zip(stages, palette)])
        ui_theme.style_fig(fig, barmode="stack", height=250, margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, traceorder="normal",
                                       bgcolor="rgba(0,0,0,0)", font=dict(color=ui_theme.INK_2)),
                           xaxis=dict(title="seconds"),
                           yaxis=dict(autorange="reversed", showgrid=False))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)
        kit.raw('<p class="fine">The bars time the work itself, about a second per event. The end-to-end figure '
                'on each card also covers the MLflow run setup and model logging around the fit.</p>')

# ---- 03 registry + 04 throughput ------------------------------------------------------------------
reg_col, thr_col = st.columns(2, gap="large")

with reg_col:
    kit.section("03", "Registry rollback", "Five flips of the <code>@production</code> alias between two real versions.")
    rollback = kit.read_csv("results/registry_rollback_times.csv")
    if rollback is None:
        kit.callout("<code>results/registry_rollback_times.csv</code> not found.")
    else:
        with kit.card("rollback"):
            labels = [f"v{a}→v{b}" for a, b in zip(rollback["from_version"], rollback["to_version"])]
            labels = [f"{i}. {lab}" for i, lab in zip(rollback["iteration"], labels)]
            fig = go.Figure([
                go.Bar(x=labels, y=rollback["alias_flip_seconds"] * 1000, name="alias flip",
                       marker_color=ui_theme.ACCENT, hovertemplate="alias flip %{y:.2f} ms<extra></extra>"),
                go.Bar(x=labels, y=rollback["audit_tag_seconds"] * 1000, name="audit tag",
                       marker_color=ui_theme.NEUTRAL, hovertemplate="audit tag %{y:.2f} ms<extra></extra>"),
            ])
            ui_theme.style_fig(fig, barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                                           font=dict(color=ui_theme.INK_2)),
                               xaxis=dict(showgrid=False), yaxis=dict(title="ms"))
            st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)
            kit.raw('<p class="fine">One SQLite write moves the alias; no model files are copied.</p>')

with thr_col:
    kit.section("04", "Feature throughput", "pandas against Dask on one host, same output to 5.5e-12.")
    thr = kit.read_csv("results/throughput_speedup.csv")
    if thr is None:
        kit.callout("<code>results/throughput_speedup.csv</code> not found.")
    else:
        with kit.card("throughput"):
            sizes = [f"{int(r) // 1000:,}k" if r < 1e6 else f"{r / 1e6:g}M" for r in thr["rows_actual"]]
            fig = go.Figure([
                go.Bar(x=sizes, y=thr["pandas_rows_per_sec"] / 1e6, name="pandas", marker_color=ui_theme.ACCENT,
                       hovertemplate="pandas %{y:.2f}M rows/s<extra></extra>"),
                go.Bar(x=sizes, y=thr["dask_rows_per_sec"] / 1e6, name="Dask", marker_color=ui_theme.NEUTRAL,
                       hovertemplate="Dask %{y:.2f}M rows/s<extra></extra>"),
            ])
            ui_theme.style_fig(fig, barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
                                           font=dict(color=ui_theme.INK_2)),
                               xaxis=dict(title="rows processed", showgrid=False),
                               yaxis=dict(title="million rows / s"))
            st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)
            kit.raw('<p class="fine">Dask loses on one machine by design. It is the path past local RAM, and '
                    'matching pandas bit for bit is what makes the swap safe.</p>')

# ---- 05 scoreboard -----------------------------------------------------------------------------
kit.section("05", "Scoreboard", "The modelling ablation and the frontier comparison, as committed.")

ablation = kit.read_csv("results/day06/ablation_modelling.csv")
frontier = kit.read_csv("results/day06/frontier_comparison.csv")
a_col, f_col = st.columns(2, gap="large")
with a_col:
    if ablation is not None:
        def _delta(v) -> str:
            return "" if v != v else f"+{v:.3f}"  # first layer has no previous layer (NaN)

        rows = "".join(
            f'<tr><td><span class="layer">{kit.esc(r["layer"].split("_")[0])}</span>{kit.esc(r["description"])}</td>'
            f'<td class="num">{r["oot_auc"]:.4f}</td><td class="num">{r["oot_auprc"]:.3f}</td>'
            f'<td class="num delta">{_delta(r["delta_auc_vs_prev_layer"])}</td></tr>'
            for _, r in ablation.iterrows())
        kit.raw('<table class="board"><thead><tr><th>Modelling layer</th><th class="num">OOT AUC</th>'
                f'<th class="num">AUPRC</th><th class="num">Δ AUC</th></tr></thead><tbody>{rows}</tbody></table>')
with f_col:
    if frontier is not None:
        rows = "".join(
            f'<tr{" class=is-champ" if kit.strategy_label(r["strategy"]) == "SENTINEL champion" else ""}>'
            f'<td>{kit.esc(kit.strategy_label(r["strategy"]))}</td><td class="num">{r["auc"]:.4f}</td>'
            f'<td class="num">{r["auprc"]:.3f}</td><td class="num">${r["cost_usd_at_1k_qps_per_day"]:,.2f}</td></tr>'
            for _, r in frontier.iterrows())
        kit.raw('<table class="board"><thead><tr><th>Strategy, same 200 OOT rows</th><th class="num">AUC</th>'
                f'<th class="num">AUPRC</th><th class="num">$ / day at 1k qps</th></tr></thead><tbody>{rows}</tbody></table>')

kit.footer("Sources: baseline_metrics.json · drift_replay_* · drift_retrain_events.csv · "
           "registry_rollback_times.csv · throughput_speedup.csv · day06/*.csv")
