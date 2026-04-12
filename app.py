"""SENTINEL Fraud Detection - Streamlit Web UI.

Multi-page Streamlit application for fraud prediction, model evaluation, and transaction analysis.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SENTINEL Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for light blue and black theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e6f4ff 0%, #bcdcff 100%);
    }
    
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #93c5fd 0%, #dbeafe 100%);
        border-bottom: 3px solid #0b0f19;
    }
    
    .main-header {
        text-align: center;
        color: #0b0f19;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    
    .main-header h1 {
        color: #0b0f19;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(14, 165, 233, 0.45);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #93c5fd 0%, #bfdbfe 100%);
        color: #0b0f19;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #0b0f19;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25);
    }
    
    .fraud-alert {
        background-color: #0b0f19;
        border-left: 6px solid #93c5fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #e6f4ff;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.35);
    }
    
    .safe-transaction {
        background-color: #bcdcff;
        border-left: 6px solid #0b0f19;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #0b0f19;
        font-weight: bold;
    }
    
    h1, h2, h3 {
        color: #0b0f19 !important;
    }
    
    .stTabs [role="tablist"] button[aria-selected="true"] {
        border-bottom: 3px solid #0b0f19 !important;
        color: #0b0f19 !important;
    }
    
    .stTabs [role="tablist"] button {
        color: #334155 !important;
    }
    
    label, p {
        color: #0b0f19 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #93c5fd 0%, #bfdbfe 100%); border-radius: 0.5rem; border: 2px solid #0b0f19;">
        <h2 style="color: #0b0f19; margin: 0; font-weight: 900;">SENTINEL</h2>
        <p style="color: #0b0f19; margin: 0.5rem 0; font-weight: bold;">Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="color: #334155;">

    Quick Navigation:
    - Home Dashboard
    - Predict Fraud
    - Performance Metrics
    - Transaction Analysis

    System Info

    Advanced fraud detection powered by XGBoost and Dask distributed computing.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #0b0f19; font-weight: bold;">Blue Mode Active</p>',
        unsafe_allow_html=True
    )

# Main page content
st.markdown("""
<div class="main-header">
    <h1>SENTINEL Fraud Detection Dashboard</h1>
    <p>Real-time fraud detection for digital payment transactions</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Predict Fraud", "Performance", "Transactions"])

with tab1:
    st.markdown("## Welcome to SENTINEL Fraud Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Model Status", value="Active", delta="Ready")

    with col2:
        st.metric(label="Model Type", value="XGBoost", delta="v2.0.3")

    with col3:
        st.metric(label="Data Source", value="Dask", delta="Scalable")
    
    st.markdown("""
    ### Features
    
    - **Real-time Fraud Prediction**: Upload transactions and get instant fraud probability scores
    - **Performance Dashboard**: View model metrics, ROC curves, confusion matrices
    - **Transaction Analysis**: Visualize transaction patterns and anomalies
    - **Batch Processing**: Handle multiple transactions at once
    
    ### How to Use
    
    1. Go to **Predict Fraud** tab to make predictions on new transactions
    2. Check **Performance** tab for model evaluation metrics
    3. Explore **Transactions** tab for data insights and patterns
    
    ### Model Details
    
    - **Algorithm**: Gradient Boosting (XGBoost)
    - **Training Data**: PaySim + Sparkov datasets
    - **Features**: 20+ engineered fraud indicators
    - **Framework**: Dask (parallel/distributed computing)
    - **Tracking**: DVC (Data Version Control)
    """)
    
    if Path("metrics/scores.json").exists():
        import json
        with open("metrics/scores.json") as f:
            scores = json.load(f)
        
        st.markdown("### Latest Model Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Accuracy", f"{scores.get('accuracy', 0):.4f}")
        with metric_col2:
            st.metric("Precision", f"{scores.get('precision', 0):.4f}")
        with metric_col3:
            st.metric("Recall", f"{scores.get('recall', 0):.4f}")
        with metric_col4:
            st.metric("F1 Score", f"{scores.get('f1_score', 0):.4f}")

with tab2:
    st.markdown("## Fraud Prediction")
    st.markdown("Make predictions on new transactions")

    st.info("Go to the Predict Fraud page from the sidebar for full prediction interface")

    st.markdown("### Quick Single Transaction Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount", value=100.0, min_value=0.0)

    with col2:
        merchant_type = st.selectbox("Merchant Type", ["online", "atm", "pos"])

    with col3:
        transaction_type = st.selectbox("Transaction Type", ["transfer", "cash_out", "payment"])

    if st.button("Predict Fraud Probability", use_container_width=True):
        st.info("Connect to full prediction page for complete functionality")

with tab3:
    st.markdown("## Model Performance")
    st.markdown("Model evaluation metrics and visualizations")

    st.info("Go to the Performance page from the sidebar for detailed metrics and visualizations")
    
    if Path("metrics/scores.json").exists():
        import json
        with open("metrics/scores.json") as f:
            scores = json.load(f)
        
        # Display key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Classification Metrics")
            metrics_text = f"""
            - **Accuracy**: {scores.get('accuracy', 0):.4f}
            - **Precision**: {scores.get('precision', 0):.4f}
            - **Recall**: {scores.get('recall', 0):.4f}
            - **F1 Score**: {scores.get('f1_score', 0):.4f}
            """
            st.markdown(metrics_text)
        
        with col2:
            st.markdown("### ROC Metrics")
            metrics_text = f"""
            - **AUC-ROC**: {scores.get('auc_roc', 0):.4f}
            - **Average Precision**: {scores.get('average_precision', 0):.4f}
            """
            st.markdown(metrics_text)

with tab4:
    st.markdown("## Transaction Visualization")
    st.markdown("Explore transaction patterns and anomalies")

    st.info("Go to the Transactions page from the sidebar for detailed analysis and visualizations")

    if Path("data/processed/features.csv").exists():
        import pandas as pd
        df = pd.read_csv("data/processed/features.csv", nrows=100)
        st.markdown(f"### Data Overview ({len(df)} sample rows)")
        st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.markdown("*SENTINEL Fraud Detection System | Powered by XGBoost & Dask*")

