"""SENTINEL Fraud Detection - Streamlit Web UI.

Multi-page Streamlit application for fraud prediction, model evaluation, and transaction analysis.
"""

import streamlit as st
from pathlib import Path
import json

st.set_page_config(
    page_title="SENTINEL Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="S",
)

# Shared CSS theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0b0f19 0%, #111827 40%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(147, 197, 253, 0.15);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f19 0%, #111827 100%);
        border-right: 1px solid rgba(147, 197, 253, 0.1);
    }

    .main-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem 0;
    }

    .main-header h1 {
        color: #e2e8f0;
        font-weight: 900;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #94a3b8 !important;
        font-size: 1.1rem;
        font-weight: 400;
    }

    .kpi-card {
        background: linear-gradient(135deg, rgba(147, 197, 253, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(147, 197, 253, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        border-color: rgba(147, 197, 253, 0.35);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }

    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #93c5fd;
        line-height: 1.2;
    }

    .kpi-card .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    .kpi-card .kpi-delta {
        font-size: 0.75rem;
        color: #34d399;
        margin-top: 0.25rem;
    }

    .feature-card {
        background: rgba(147, 197, 253, 0.05);
        border: 1px solid rgba(147, 197, 253, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }

    .feature-card:hover {
        border-color: rgba(147, 197, 253, 0.3);
    }

    .feature-card h4 {
        color: #93c5fd !important;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .feature-card p {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    h1, h2, h3 {
        color: #e2e8f0 !important;
    }

    p, label, .stMarkdown {
        color: #cbd5e1 !important;
    }

    .stTabs [role="tablist"] {
        background: rgba(147, 197, 253, 0.05);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [role="tablist"] button {
        color: #94a3b8 !important;
        border-radius: 6px;
        font-weight: 500;
    }

    .stTabs [role="tablist"] button[aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #93c5fd !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(147, 197, 253, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(147, 197, 253, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }

    .stMetric label {
        color: #94a3b8 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(147, 197, 253, 0.2), transparent);
        margin: 2rem 0;
    }

    .footer-text {
        text-align: center;
        color: #475569 !important;
        font-size: 0.8rem;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(147, 197, 253, 0.08);
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .status-active {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: #93c5fd;">S</div>
        <h2 style="color: #e2e8f0; margin: 0; font-weight: 900; font-size: 1.3rem; letter-spacing: 0.1em;">SENTINEL</h2>
        <p style="color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.8rem; letter-spacing: 0.05em;">FRAUD DETECTION SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem;">NAVIGATION</p>
        <p style="color: #64748b; font-size: 0.8rem; line-height: 2;">
        Home Dashboard<br>
        Predict Fraud<br>
        Performance Metrics<br>
        Transaction Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem;">POWERED BY</p>
        <p style="color: #64748b; font-size: 0.8rem; line-height: 1.8;">
        XGBoost Classifier<br>
        Dask Computing<br>
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

# Load metrics once
scores = {}
if Path("metrics/scores.json").exists():
    with open("metrics/scores.json") as f:
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
    st.metric(label="Processing", value="Dask", delta="Distributed")

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
    | Framework | Dask parallel computing |
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
    SENTINEL Fraud Detection System v2.0 | XGBoost + Dask + DVC Pipeline
</div>
""", unsafe_allow_html=True)
