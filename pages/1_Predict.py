"""Fraud Prediction Page - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import predict_dataframe

st.set_page_config(page_title="Predict Fraud - SENTINEL", layout="wide", page_icon="S")

# Dark theme CSS
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

    h1, h2, h3 {
        color: #e2e8f0 !important;
    }

    p, label, .stMarkdown {
        color: #cbd5e1 !important;
    }

    .page-header {
        padding: 1rem 0 2rem 0;
    }

    .page-header h1 {
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 2.2rem;
    }

    .page-header p {
        color: #94a3b8 !important;
    }

    .stTabs [role="tablist"] {
        background: rgba(147, 197, 253, 0.05);
        border-radius: 8px;
        padding: 4px;
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

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        border: 1px solid rgba(147, 197, 253, 0.3);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(147, 197, 253, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(147, 197, 253, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }

    .stMetric label { color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    .result-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .result-fraud h3 { color: #fca5a5 !important; margin: 0 0 0.5rem 0; }
    .result-fraud p { color: #fecaca !important; margin: 0; font-weight: 500; }

    .result-safe {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-left: 4px solid #34d399;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .result-safe h3 { color: #6ee7b7 !important; margin: 0 0 0.5rem 0; }
    .result-safe p { color: #a7f3d0 !important; margin: 0; font-weight: 500; }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(147, 197, 253, 0.2), transparent);
        margin: 2rem 0;
    }

    .stNumberInput label, .stSelectbox label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>Fraud Prediction</h1>
    <p>Analyze transactions for potential fraud using the trained XGBoost model</p>
</div>
""", unsafe_allow_html=True)

model_path = Path("models/fraud_model.pkl")
if not model_path.exists():
    st.error("Model not found at `models/fraud_model.pkl`. Please train the model first.")
    st.stop()

st.success("Model loaded and ready for predictions")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])

with tab1:
    st.markdown("### Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            value=500.0,
            min_value=0.0,
            step=10.0,
            help="Dollar amount of the transaction",
        )

    with col2:
        transaction_type = st.selectbox(
            "Transaction Type",
            options=["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
            help="Type of transaction being performed",
        )

    with col3:
        source = st.selectbox(
            "Data Source",
            options=["paysim", "sparkov"],
            help="Which payment network the transaction originates from",
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        hour_of_day = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour (0-23) when the transaction occurred",
        )

    with col5:
        day_of_month = st.slider(
            "Day of Month",
            min_value=1,
            max_value=31,
            value=15,
            help="Day of month when the transaction occurred",
        )

    with col6:
        has_balance_info = st.selectbox(
            "Balance Info Available",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether sender balance data is available",
        )

    col7, col8 = st.columns(2)

    with col7:
        balance_change_orig = st.number_input(
            "Balance Change (Sender)",
            value=0.0,
            step=100.0,
            help="Difference between old and new balance of the sender",
        )

    with col8:
        balance_ratio = st.number_input(
            "Balance Ratio",
            value=0.5,
            min_value=0.0,
            max_value=10.0,
            step=0.05,
            help="Ratio of new balance to old balance (newBal / (oldBal + 1))",
        )

    st.markdown("")

    if st.button("Analyze Transaction", use_container_width=True, type="primary"):
        try:
            input_data = pd.DataFrame({
                'amount': [amount],
                'transaction_type': [transaction_type],
                'hour_of_day': [hour_of_day],
                'day_of_month': [day_of_month],
                'balance_change_orig': [balance_change_orig],
                'balance_ratio': [balance_ratio],
                'has_balance_info': [has_balance_info],
                'source': [source],
            })

            with st.spinner("Analyzing transaction..."):
                results = predict_dataframe(input_data, str(model_path))

            fraud_prob = results['fraud_probability'].values[0]
            fraud_pred = results['fraud_prediction'].values[0]

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")

            with res_col2:
                prediction_label = "FRAUDULENT" if fraud_pred == 1 else "LEGITIMATE"
                st.metric("Prediction", prediction_label)

            with res_col3:
                confidence = max(fraud_prob, 1 - fraud_prob) * 100
                st.metric("Confidence", f"{confidence:.2f}%")

            if fraud_pred == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <h3>FRAUD ALERT</h3>
                    <p>This transaction has been flagged as potentially fraudulent
                    with a {fraud_prob * 100:.2f}% probability. Review recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <h3>LEGITIMATE TRANSACTION</h3>
                    <p>This transaction appears legitimate with a {(1 - fraud_prob) * 100:.2f}% confidence level.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Transaction Summary")
            summary_df = pd.DataFrame({
                "Field": ["Amount", "Type", "Source", "Hour", "Day",
                          "Balance Change", "Balance Ratio", "Balance Info"],
                "Value": [f"${amount:,.2f}", transaction_type, source,
                          str(hour_of_day), str(day_of_month),
                          f"${balance_change_orig:,.2f}", f"{balance_ratio:.4f}",
                          "Yes" if has_balance_info else "No"],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Ensure the model is trained and all required features are properly formatted.")

with tab2:
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file to analyze multiple transactions at once.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV file should contain transaction columns: amount, merchant_id, customer_id, type, merchant_type, etc.",
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.markdown(f"**Loaded {len(input_df)} transactions**")
            st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("Analyze All Transactions", use_container_width=True, type="primary"):
                try:
                    with st.spinner(f"Analyzing {len(input_df)} transactions..."):
                        results = predict_dataframe(input_df, str(model_path))

                    output_df = pd.concat([input_df, results], axis=1)

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### Batch Results")

                    fraudulent_count = (results['fraud_prediction'] == 1).sum()
                    legitimate_count = (results['fraud_prediction'] == 0).sum()
                    avg_fraud_prob = results['fraud_probability'].mean()

                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                    with stats_col1:
                        st.metric("Total Transactions", len(input_df))
                    with stats_col2:
                        st.metric("Flagged as Fraud", fraudulent_count)
                    with stats_col3:
                        st.metric("Legitimate", legitimate_count)
                    with stats_col4:
                        st.metric("Avg Fraud Probability", f"{avg_fraud_prob * 100:.2f}%")

                    st.markdown("### Distribution")
                    col1, col2 = st.columns(2)

                    with col1:
                        fraud_dist = results['fraud_prediction'].value_counts()
                        st.bar_chart(fraud_dist)

                    with col2:
                        fraud_labels = {0: "Legitimate", 1: "Fraudulent"}
                        pie_data = results['fraud_prediction'].value_counts().rename(fraud_labels)
                        st.write(pie_data)

                    st.markdown("### Detailed Predictions")

                    display_df = output_df.copy()
                    display_df['fraud_probability'] = display_df['fraud_probability'].apply(
                        lambda x: f"{x * 100:.2f}%"
                    )
                    display_df['fraud_prediction'] = display_df['fraud_prediction'].map(
                        {0: "Legitimate", 1: "Fraudulent"}
                    )

                    st.dataframe(display_df, use_container_width=True)

                    csv = output_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")

        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")

    else:
        st.info("Upload a CSV file to get started with batch predictions.")

        with st.expander("Example CSV Format"):
            example_data = {
                'amount': [500.0, 1000.0, 150.0],
                'merchant_id': [123, 456, 789],
                'customer_id': [1, 2, 3],
                'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'],
                'merchant_type': ['online', 'atm', 'pos'],
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)

            example_csv = example_df.to_csv(index=False)
            st.download_button(
                label="Download Example CSV",
                data=example_csv,
                file_name="example_transactions.csv",
                mime="text/csv",
            )

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1.5rem 0;
            border-top: 1px solid rgba(147, 197, 253, 0.08);">
    Predictions powered by XGBoost trained on PaySim and Sparkov datasets
</div>
""", unsafe_allow_html=True)
