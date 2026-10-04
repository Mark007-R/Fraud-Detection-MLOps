"""SENTINEL fraud detection — Streamlit entry point.

Sets the page config and theme once, then routes to the pages in views/ via
top navigation:

    Overview    what the system is and the measured results
    Score       score one transaction with the production model
    Screen      score and explore a batch (upload, demo batch, or local data)
    Model       evaluation metrics, ROC, confusion matrix, feature importance
    Ops         drift replay, auto-retrain, registry rollback, throughput

Run with ``streamlit run app.py``.
"""

import streamlit as st

from ui_kit import ROOT
from ui_theme import apply_theme

st.set_page_config(
    page_title="SENTINEL · Fraud Detection",
    page_icon=str(ROOT / "assets" / "logo-mark.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Shared mark.dev theme — MODE and accent live in ui_theme.py (tokens, type, widgets, SENTINEL components)
apply_theme()

st.logo(str(ROOT / "assets" / "logo.svg"), size="large")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/shield:", default=True),
    st.Page("views/score.py", title="Score", icon=":material/bolt:", url_path="score"),
    st.Page("views/screen.py", title="Screen", icon=":material/table_rows:", url_path="screen"),
    st.Page("views/model.py", title="Model", icon=":material/insights:", url_path="model"),
    st.Page("views/ops.py", title="Ops", icon=":material/monitor_heart:", url_path="ops"),
]

st.navigation(pages, position="top").run()
