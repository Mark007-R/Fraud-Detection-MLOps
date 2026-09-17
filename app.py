"""SENTINEL Fraud Detection - Streamlit Web UI.

Multi-page Streamlit application for fraud prediction, model evaluation, and transaction analysis.
"""

import streamlit as st
from pathlib import Path
import json

from ui_theme import apply_theme

st.set_page_config(
    page_title="SENTINEL Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="S",
)

# Shared mark.dev theme — MODE and accent live in ui_theme.py (tokens, type, widgets, SENTINEL components)
apply_theme()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
        <div class="brand-mark">S</div>
        <h2>SENTINEL</h2>
        <p>FRAUD DETECTION SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="side-block">
        <div class="side-label">NAVIGATION</div>
        <p class="side-list">
        Home Dashboard<br>
        Predict Fraud<br>
        Performance Metrics<br>
        Transaction Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="side-block">
        <div class="side-label">POWERED BY</div>
        <p class="side-list">
        XGBoost Classifier<br>
        Pandas + scikit-learn<br>
        DVC Pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center;">'
        '<span class="status-badge status-active">System Online</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# Main page content
st.markdown("""
<div class="main-header">
    <h1>SENTINEL</h1>
    <p>Advanced fraud detection for digital payment transactions</p>
</div>
""", unsafe_allow_html=True)

# Load metrics once (resolve path relative to this file, not cwd)
PROJECT_ROOT = Path(__file__).resolve().parent
metrics_file = PROJECT_ROOT / "metrics" / "scores.json"
scores = {}
if metrics_file.exists():
    with open(metrics_file) as f:
        scores = json.load(f)

# KPI Row
if not scores:
    st.info("Model not yet evaluated. Run `dvc repro evaluate` to generate metrics.")

if scores:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{scores.get('accuracy', 0):.2%}</div>
            <div class="kpi-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{scores.get('precision', 0):.2%}</div>
            <div class="kpi-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{scores.get('recall', 0):.2%}</div>
            <div class="kpi-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{scores.get('f1_score', 0):.2%}</div>
            <div class="kpi-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{scores.get('auc_roc', 0):.4f}</div>
            <div class="kpi-label">AUC-ROC</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# System status row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Model Status", value="Active", delta="Ready")
with col2:
    st.metric(label="Algorithm", value="XGBoost", delta="v2.0.3")
with col3:
    st.metric(label="Processing", value="Pandas", delta="In-memory")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Features section
st.markdown("### Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>Real-time Prediction</h4>
        <p>Upload transactions and receive instant fraud probability scores with confidence levels.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>Performance Metrics</h4>
        <p>Comprehensive model evaluation with ROC curves, confusion matrices, and feature analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>Transaction Analysis</h4>
        <p>Visualize patterns, detect anomalies, and explore statistical distributions interactively.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <h4>Batch Processing</h4>
        <p>Upload CSV files for bulk fraud screening with exportable results and summaries.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Quick guide
st.markdown("### Quick Start Guide")

st.markdown("""
| Step | Action | Page |
|------|--------|------|
| 1 | Upload or enter transaction data for fraud scoring | **Predict Fraud** |
| 2 | Review model accuracy, ROC curves, and feature importance | **Performance** |
| 3 | Explore transaction distributions and correlations | **Transactions** |
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Model details
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Architecture")
    st.markdown("""
    | Component | Detail |
    |-----------|--------|
    | Algorithm | Gradient Boosting (XGBoost) |
    | Training Data | PaySim + Sparkov datasets |
    | Features | 20+ engineered fraud indicators |
    | Framework | Pandas + scikit-learn |
    | Pipeline | DVC version control |
    """)

with col2:
    st.markdown("### Model Performance")
    if scores:
        st.markdown(f"""
        | Metric | Score |
        |--------|-------|
        | Accuracy | {scores.get('accuracy', 0):.4f} |
        | Precision | {scores.get('precision', 0):.4f} |
        | Recall | {scores.get('recall', 0):.4f} |
        | F1 Score | {scores.get('f1_score', 0):.4f} |
        | AUC-ROC | {scores.get('auc_roc', 0):.4f} |
        | Avg Precision | {scores.get('average_precision', 0):.4f} |
        """)
    else:
        st.info("Run model evaluation to see metrics here.")

st.markdown("""
<div class="footer-text">
    SENTINEL Fraud Detection System v2.0 | XGBoost + Pandas + DVC Pipeline
</div>
""", unsafe_allow_html=True)
