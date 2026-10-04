"""Score — one transaction through the production model, re-scored on every change."""

import json

import numpy as np
import streamlit as st

import ui_kit as kit

kit.page_header("Score", "Score a <em>transaction</em>",
                "The model re-scores on every change. Start from a scenario or set the fields yourself.")

artifact = kit.require_model()

FIELDS = ("sc_source", "sc_type", "sc_amount", "sc_before", "sc_after", "sc_hour", "sc_day")


def _load_preset(name: str) -> None:
    p = kit.PRESETS[name]
    st.session_state.update({
        "sc_source": p["source"], "sc_type": p["transaction_type"], "sc_amount": float(p["amount"]),
        "sc_hour": int(p["hour_of_day"]), "sc_day": int(p["day_of_month"]),
        "sc_before": float(p.get("before", st.session_state.get("sc_before", 50_000.0))),
        "sc_after": float(p.get("after", st.session_state.get("sc_after", 49_500.0))),
    })


def _on_preset() -> None:
    if st.session_state.get("sc_preset"):
        _load_preset(st.session_state.sc_preset)


def _on_source() -> None:
    st.session_state.sc_type = (kit.PAYSIM_TYPES if st.session_state.sc_source == "paysim"
                                else kit.SPARKOV_CATEGORIES)[0]
    st.session_state.sc_preset = None


def _on_edit() -> None:
    st.session_state.sc_preset = None  # any manual edit leaves the scenario


if "sc_source" not in st.session_state:
    st.session_state.sc_preset = "Late-night online"
    _load_preset("Late-night online")

form_col, result_col = st.columns([1.1, 1], gap="large")

# ---- inputs ----------------------------------------------------------------------
with form_col:
    with kit.card("form"):
        kit.card_title("Transaction")
        st.pills("Scenario", list(kit.PRESETS), key="sc_preset", on_change=_on_preset)
        st.segmented_control("Network", ["sparkov", "paysim"], key="sc_source", required=True,
                             format_func=kit.SOURCE_LABELS.get, on_change=_on_source)
        source = st.session_state.sc_source
        paysim = source == "paysim"

        c1, c2 = st.columns(2)
        c1.selectbox("Type" if paysim else "Merchant category",
                     kit.PAYSIM_TYPES if paysim else kit.SPARKOV_CATEGORIES,
                     key="sc_type", format_func=kit.type_label, on_change=_on_edit)
        c2.number_input("Amount ($)", min_value=0.0, step=10.0, format="%.2f", key="sc_amount",
                        on_change=_on_edit)

        if paysim:
            b1, b2 = st.columns(2)
            b1.number_input("Sender balance before", min_value=0.0, step=100.0, format="%.2f",
                            key="sc_before", on_change=_on_edit)
            b2.number_input("Sender balance after", min_value=0.0, step=100.0, format="%.2f",
                            key="sc_after", on_change=_on_edit)
        else:
            kit.raw('<p class="fine">Card transactions in Sparkov carry no balance data, '
                    'so the model sees those features as missing.</p>')

        h1, h2 = st.columns(2)
        h1.slider("Hour of day", 0, 23, key="sc_hour", on_change=_on_edit, format="%02d:00")
        h2.slider("Day of month", 1, 31, key="sc_day", on_change=_on_edit)

    h = kit.headline()
    kit.callout(
        "This is <code>models/fraud_model.pkl</code>, the DVC train-stage model. It is close to perfect on "
        "PaySim mobile money and much weaker on card data from a later window (Sparkov out-of-time AUC "
        f"<b>{h['honest_auc']:.3f}</b>), so card scores can swing hard on small changes. The tuned "
        f"<b>{h['oot_auc']:.3f}</b> champion is written to <code>models/fraud_model_tuned.pkl</code> by "
        "<code>src/tuning/eval_best.py</code> and isn't bundled here.",
        title="About this model",
    )

row = kit.to_row(
    source=source, transaction_type=st.session_state.sc_type, amount=st.session_state.sc_amount,
    hour_of_day=st.session_state.sc_hour, day_of_month=st.session_state.sc_day,
    before=st.session_state.get("sc_before"), after=st.session_state.get("sc_after"),
)
prob = float(kit.score(row)["fraud_probability"].iloc[0])
flagged = prob >= kit.THRESHOLD
feats = kit.engineered(row).iloc[0]

# ---- verdict ---------------------------------------------------------------------------
with result_col:
    gap = abs(prob - kit.THRESHOLD) * 100
    note = (f"{gap:.1f} points above the 0.50 threshold. The service would return label 1 and route "
            "this to review." if flagged else
            f"{gap:.1f} points under the 0.50 threshold. The service would return label 0 and let it through.")
    kit.raw(
        f'<div class="verdict {"is-flag" if flagged else "is-clear"}">'
        f'<div class="v-top"><span class="eyebrow">Fraud probability</span>'
        f'<span class="v-pill">{"Flag for review" if flagged else "Clear"}</span></div>'
        f'<div class="v-prob">{kit.pct(prob)}<span>%</span></div>'
        f'<div class="meter"><i style="width:{max(prob * 100, 0.8):.2f}%"></i>'
        f'<b style="left:{kit.THRESHOLD * 100:.0f}%"></b></div>'
        '<div class="v-scale"><span>0</span><span>0.25</span><span>0.50 threshold</span><span>0.75</span><span>1</span></div>'
        f'<p class="v-note">{note}</p></div>'
    )

    def _yes(v) -> str:
        return "yes" if int(v) else "no"

    balance = ("—" if not paysim else f"{row['balance_change_orig'].iloc[0]:,.2f}")
    ratio = ("—" if not paysim else f"{row['balance_ratio'].iloc[0]:.4f}")
    band = next((b for b in ("low", "medium", "high", "very_high") if feats.get(f"amount_bin_{b}", 0) == 1), "—")
    signals = [
        ("Amount z-score", f"{feats['amount_zscore']:+.2f}", "vs the training mean"),
        ("Amount band", band.replace("_", " "), "<1k · <10k · <100k · above"),
        ("Above p95 / p99", f"{_yes(feats['is_p95_amount'])} / {_yes(feats['is_p99_amount'])}", "training percentiles"),
        ("Night-time", _yes(feats["is_night"]), "22:00 to 06:00"),
        ("Balance change", balance, "before minus after"),
        ("Balance ratio", ratio, "after ÷ (before + 1)"),
    ]
    kit.raw(
        '<div class="signals"><div class="eyebrow">What the model sees</div><dl>'
        + "".join(f'<div><dt>{kit.esc(k)}</dt><dd>{kit.esc(v)}</dd><small>{kit.esc(s)}</small></div>'
                  for k, v, s in signals)
        + f'</dl><p class="fine">{len(artifact["feature_columns"])} engineered features in all, '
          'built by <code>src/preprocess.py</code> with the training-time thresholds.</p></div>'
    )

    with st.expander("Send the same request to the API"):
        cols = artifact["feature_columns"]
        aligned = {c: (None if c not in feats or (isinstance(feats[c], float) and np.isnan(feats[c]))
                       else round(float(feats[c]), 6)) for c in cols}
        payload = {"request_id": "demo-1", "features": {k: v for k, v in aligned.items() if v is not None}}
        st.caption("`POST /predict` on the FastAPI service (`docker compose --profile serving up -d api`).")
        st.code(json.dumps(payload, indent=2), language="json")

kit.footer("Scored by models/fraud_model.pkl through src/predict.py, the same path the batch CLI uses")
