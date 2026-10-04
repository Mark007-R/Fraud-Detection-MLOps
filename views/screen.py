"""Screen — score a batch of transactions and see where the risk concentrates."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_kit as kit
import ui_theme

kit.page_header("Screen", "Screen a <em>batch</em>",
                "Score a file of transactions in one pass, then see where the risk concentrates.")

kit.require_model()

SAMPLE_ROWS = 50_000
CHART = dict(margin=dict(l=10, r=10, t=26, b=10), height=300)
PLOT_CFG = {"displayModeBar": False}


@st.cache_data(show_spinner=False)
def _training_sample() -> pd.DataFrame:
    """First rows of the DVC features file, mapped back to the raw scoring schema."""
    feats = pd.read_csv(kit.FEATURES_PATH, nrows=SAMPLE_ROWS)
    out = pd.DataFrame({c: feats[c] for c in kit.RAW_COLUMNS if c in feats})
    for prefix, col in (("transaction_type_", "transaction_type"), ("source_", "source")):
        onehot = [c for c in feats.columns if c.startswith(prefix)]
        if onehot and col not in out:
            out[col] = feats[onehot].idxmax(axis=1).str[len(prefix):]
    if "is_fraud" in feats:
        out["is_fraud"] = feats["is_fraud"].astype(int)
    return out


options = ["Demo batch", "Upload CSV"] + (["Training sample"] if kit.FEATURES_PATH.exists() else [])
choice = st.segmented_control("Data", options, default="Demo batch", key="scr_source", required=True)

frame = None
if choice == "Demo batch":
    frame = kit.demo_batch()
    kit.callout(
        f"<b>Synthetic demo batch.</b> {len(frame)} seeded rows shaped like PaySim and Sparkov traffic, with a "
        "few planted patterns that fraud tends to follow (accounts emptied at night, large late-night online "
        "card purchases). Not real transactions: the 6.1M-row datasets are DVC-tracked and not bundled here."
    )
elif choice == "Upload CSV":
    up_col, tpl_col = st.columns([2, 1], gap="large", vertical_alignment="center")
    with up_col:
        upload = st.file_uploader("Transactions CSV", type="csv",
                                  help="Columns: " + ", ".join(kit.RAW_COLUMNS) + " (only amount is required)")
    with tpl_col:
        kit.raw('<p class="fine">PaySim rows take a type (TRANSFER, CASH_OUT…) and balance columns; '
                'Sparkov rows take a merchant category and leave balances empty.</p>')
        st.download_button("Download a template", kit.demo_batch(12).to_csv(index=False),
                           file_name="sentinel_template.csv", mime="text/csv",
                           icon=":material/download:")
    if upload is not None:
        try:
            frame = kit.normalise_batch(pd.read_csv(upload))
        except Exception as exc:  # unreadable CSV or wrong columns
            kit.callout(f"<b>Couldn't use that file.</b> {kit.esc(exc)}", tone="bad")
    else:
        kit.callout("Drop a CSV above, or switch to the demo batch to see the page populated.")
else:
    frame = _training_sample()
    kit.callout(f"First {len(frame):,} rows of <code>data/processed/features.csv</code> with their labels. "
                "These rows are in the training window, so the scores are in-sample.")

if frame is None:
    kit.footer("Batch scoring runs through src/predict.py, the same path as the CLI")
    st.stop()

with st.spinner(f"Scoring {len(frame):,} transactions…"):
    scored = kit.score(frame[kit.RAW_COLUMNS])
df = frame.reset_index(drop=True).assign(
    fraud_probability=scored["fraud_probability"].to_numpy(),
    flagged=scored["fraud_prediction"].to_numpy().astype(bool),
)
labelled = "is_fraud" in df

# ---- summary -----------------------------------------------------------------------
n, n_flag = len(df), int(df["flagged"].sum())
tiles = [
    {"label": "Transactions", "value": f"{n:,}", "tone": "ink",
     "foot": f"{(df['source'] == 'paysim').sum():,} mobile money · {(df['source'] == 'sparkov').sum():,} card"},
    {"label": "Flagged", "value": f"{n_flag:,}", "foot": f"{n_flag / n:.1%} of the batch at threshold 0.50"},
    {"label": "Median risk", "value": f"{df['fraud_probability'].median() * 100:.2f}", "unit": "%", "tone": "ink",
     "foot": f"Mean {df['fraud_probability'].mean() * 100:.2f}%"},
]
if labelled:
    caught = int((df["flagged"] & (df["is_fraud"] == 1)).sum())
    total = int(df["is_fraud"].sum())
    tiles.append({"label": "Labelled fraud caught", "value": f"{caught}/{total}",
                  "foot": f"{caught / max(total, 1):.0%} recall · {caught / max(n_flag, 1):.0%} precision"})
else:
    top = df.loc[df["fraud_probability"].idxmax()]
    tiles.append({"label": "Highest risk", "value": kit.pct(top["fraud_probability"]), "unit": "%",
                  "foot": kit.esc(kit.describe_row(top))})
kit.stat_grid(tiles)

# ---- charts ----------------------------------------------------------------------------
c1, c2 = st.columns(2, gap="large")
with c1:
    with kit.card("dist"):
        kit.card_title("Distribution", "How scores spread across the batch")
        bins = np.linspace(0, 1, 41)
        counts, edges = np.histogram(df["fraud_probability"], bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        above = mids >= kit.THRESHOLD
        fig = go.Figure([
            go.Bar(x=mids[~above], y=counts[~above], width=0.024, marker_color=ui_theme.NEUTRAL, name="clear",
                   hovertemplate="p≈%{x:.2f}: %{y} rows<extra></extra>"),
            go.Bar(x=mids[above], y=counts[above], width=0.024, marker_color=ui_theme.ACCENT, name="flagged",
                   hovertemplate="p≈%{x:.2f}: %{y} rows<extra></extra>"),
        ])
        fig.add_vline(x=kit.THRESHOLD, line_dash="dot", line_color=ui_theme.INK_3)
        ui_theme.style_fig(fig, **CHART, showlegend=False, bargap=0.05,
                           xaxis=dict(title="fraud probability", range=[0, 1]),
                           yaxis=dict(title="rows (log)", type="log", dtick=1))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)

with c2:
    with kit.card("hour"):
        kit.card_title("Time of day", "Share of each hour's transactions flagged")
        by_hour = df.groupby("hour_of_day").agg(rows=("flagged", "size"), rate=("flagged", "mean"))
        by_hour = by_hour.reindex(range(24), fill_value=0)
        night = [(h < 6 or h >= 22) for h in by_hour.index]
        fig = go.Figure(go.Bar(
            x=[f"{h:02d}" for h in by_hour.index], y=by_hour["rate"] * 100,
            marker_color=[ui_theme.ACCENT if nt else ui_theme.NEUTRAL for nt in night],
            customdata=by_hour["rows"], hovertemplate="%{x}:00 · %{y:.1f}% flagged of %{customdata}<extra></extra>",
        ))
        ui_theme.style_fig(fig, **CHART, showlegend=False,
                           xaxis=dict(title="hour (night hours in accent)", showgrid=False),
                           yaxis=dict(title="% flagged", ticksuffix="%"))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)

c3, c4 = st.columns(2, gap="large")
with c3:
    with kit.card("amount"):
        kit.card_title("Amount", "Amount against score, by verdict")
        fig = go.Figure()
        for is_flag, colour, name, opacity in ((False, ui_theme.NEUTRAL, "clear", 0.45),
                                               (True, ui_theme.ACCENT, "flagged", 0.9)):
            part = df[df["flagged"] == is_flag]
            fig.add_trace(go.Scatter(
                x=part["amount"].clip(lower=0.01), y=part["fraud_probability"], mode="markers", name=name,
                marker=dict(color=colour, size=7, opacity=opacity, line=dict(width=0)),
                text=[kit.describe_row(r) for _, r in part.iterrows()],
                hovertemplate="%{text}<br>p=%{y:.3f}<extra></extra>",
            ))
        fig.add_hline(y=kit.THRESHOLD, line_dash="dot", line_color=ui_theme.INK_3)
        ui_theme.style_fig(fig, **CHART,
                           legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)",
                                       font=dict(color=ui_theme.INK_2)),
                           xaxis=dict(title="amount ($, log)", type="log"),
                           yaxis=dict(title="fraud probability", range=[-0.03, 1.03]))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)

with c4:
    with kit.card("type"):
        kit.card_title("Type", "Mean score by transaction type or merchant category")
        by_type = (df.groupby("transaction_type")
                   .agg(rows=("fraud_probability", "size"), mean=("fraud_probability", "mean"))
                   .sort_values("mean").tail(12))
        fig = go.Figure(go.Bar(
            y=[kit.type_label(t) for t in by_type.index], x=by_type["mean"] * 100, orientation="h",
            marker=dict(color=by_type["mean"], colorscale=ui_theme.bar_scale()),
            customdata=by_type["rows"], hovertemplate="%{y}: mean %{x:.1f}% over %{customdata} rows<extra></extra>",
        ))
        ui_theme.style_fig(fig, **CHART, showlegend=False,
                           xaxis=dict(title="mean fraud probability", ticksuffix="%"),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG)

# ---- highest-risk rows -----------------------------------------------------------------------
kit.section("Review queue", "Highest-risk transactions", "The 25 rows a reviewer would open first.")
queue = df.sort_values("fraud_probability", ascending=False).head(25).copy()
queue["network"] = queue["source"].map({"paysim": "Mobile money", "sparkov": "Card"}).fillna(queue["source"])
queue["type"] = queue["transaction_type"].map(kit.type_label)
queue["hour"] = queue["hour_of_day"].astype(int).map("{:02d}:00".format)
queue["balance"] = queue["balance_change_orig"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
show = ["fraud_probability", "network", "type", "amount", "hour", "day_of_month", "balance"]
if labelled:
    show.append("is_fraud")
st.dataframe(
    queue[show], hide_index=True, width="stretch",
    column_config={
        "fraud_probability": st.column_config.ProgressColumn("Risk", format="%.3f", min_value=0.0, max_value=1.0),
        "network": "Network", "type": "Type", "hour": "Hour",
        "amount": st.column_config.NumberColumn("Amount", format="dollar"),
        "day_of_month": st.column_config.NumberColumn("Day"),
        "balance": "Balance change",
        "is_fraud": st.column_config.CheckboxColumn("Labelled fraud"),
    },
)

st.download_button("Download every scored row (CSV)", df.to_csv(index=False), file_name="sentinel_scored.csv",
                   mime="text/csv", icon=":material/download:")

kit.footer("Batch scoring runs through src/predict.py, the same path as the CLI")
