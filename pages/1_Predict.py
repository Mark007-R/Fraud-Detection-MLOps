"""Fraud Prediction Page - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import predict_dataframe

st.set_page_config(page_title="Predict Fraud - SENTINEL", layout="wide")

# Custom CSS for red and white theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    h1, h2, h3, label {
        color: #FFFFFF !important;
    }
    
    .stTabs [role="tablist"] button[aria-selected="true"] {
        border-bottom: 3px solid #CC0000 !important;
        color: #CC0000 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #CC0000 0%, #990000 100%);
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFFFFF 0%, #CCCCCC 100%);
        color: #CC0000 !important;
    }
    
    .stNumberInput, .stSelectbox, .stTextInput {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Fraud Prediction")
st.markdown("Detect fraudulent transactions in real-time using our XGBoost model")

model_path = Path("models/fraud_model.pkl")
if not model_path.exists():
    st.error("Model not found at `models/fraud_model.pkl`. Please train the model first.")
    st.stop()

st.success("Model loaded successfully")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])

with tab1:
    st.markdown("## Single Transaction Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            value=500.0,
            min_value=0.0,
            step=10.0,
            help="Amount of the transaction"
        )
    
    with col2:
        merchant_id = st.number_input(
            "Merchant ID",
            value=1,
            min_value=1,
            help="Unique merchant identifier"
        )
    
    with col3:
        customer_id = st.number_input(
            "Customer ID",
            value=1,
            min_value=1,
            help="Unique customer identifier"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        transaction_type = st.selectbox(
            "Transaction Type",
            options=["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
            help="Type of transaction being performed"
        )
    
    with col5:
        merchant_type = st.selectbox(
            "Merchant Type",
            options=["online", "pos", "atm", "physical"],
            help="Category of the merchant"
        )
    
    with col6:
        time_of_day = st.selectbox(
            "Time of Day",
            options=["night", "morning", "afternoon", "evening"],
            help="Time period of the transaction"
        )
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        days_since_last = st.number_input(
            "Days Since Last Transaction",
            value=0,
            min_value=0,
            help="Number of days since customer's last transaction"
        )
    
    with col8:
        transactions_today = st.number_input(
            "Transactions Today",
            value=1,
            min_value=1,
            help="Number of transactions customer made today"
        )
    
    with col9:
        amount_change = st.number_input(
            "Amount Change % (vs avg)",
            value=0.0,
            min_value=-100.0,
            max_value=1000.0,
            help="Percentage change from average transaction amount"
        )
    
    # Prediction button
    if st.button("🔮 Predict Fraud Probability", use_container_width=True, type="primary"):
        try:
            # Create dataframe for single prediction
            input_data = pd.DataFrame({
                'amount': [amount],
                'merchant_id': [merchant_id],
                'customer_id': [customer_id],
                'type': [transaction_type],
                'merchant_type': [merchant_type],
                'time_of_day': [time_of_day],
                'days_since_last_transaction': [days_since_last],
                'transactions_today': [transactions_today],
                'amount_change_pct': [amount_change],
            })
            
            with st.spinner("Analyzing transaction..."):
                results = predict_dataframe(input_data, str(model_path))

            fraud_prob = results['fraud_probability'].values[0]
            fraud_pred = results['fraud_prediction'].values[0]

            st.markdown("---")
            st.markdown("## Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_prob*100:.2f}%",
                    delta=None
                )

            with res_col2:
                prediction_label = "FRAUDULENT" if fraud_pred == 1 else "LEGITIMATE"
                st.metric(
                    "Prediction",
                    prediction_label,
                    delta=None
                )

            with res_col3:
                confidence = (max(fraud_prob, 1-fraud_prob) * 100)
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                    delta=None
                )

            if fraud_pred == 1:
                st.markdown(f"""
                <div style="background-color: #CC0000; border-left: 6px solid #FFFFFF;
                            padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(204, 0, 0, 0.4);">
                    <h3 style="color: #FFFFFF; margin: 0;">FRAUD ALERT DETECTED</h3>
                    <p style="color: #FFFFFF; margin: 0.5rem 0 0 0; font-weight: bold; font-size: 16px;">
                        This transaction has been flagged as potentially fraudulent
                        with a {fraud_prob*100:.2f}% probability.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #1a1f3a; border-left: 6px solid #00FF00;
                            padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h3 style="color: #00FF00; margin: 0;">LEGITIMATE TRANSACTION DETECTED</h3>
                    <p style="color: #FFFFFF; margin: 0.5rem 0 0 0; font-weight: bold; font-size: 16px;">
                        This transaction appears legitimate with a {(1-fraud_prob)*100:.2f}% confidence level.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Transaction Summary")
            summary_data = {
                "Amount": f"${amount:,.2f}",
                "Type": transaction_type,
                "Merchant": merchant_type,
                "Time": time_of_day,
                "Customer Transactions (Today)": transactions_today,
                "Days Since Last Activity": days_since_last,
            }

            for key, value in summary_data.items():
                st.write(f"**{key}:** {value}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Make sure the model is trained and all required features are properly formatted.")

with tab2:
    st.markdown("## Batch Prediction")
    st.markdown("Upload a CSV file to make predictions on multiple transactions at once")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV file should contain transaction columns: amount, merchant_id, customer_id, type, merchant_type, etc."
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.markdown(f"### Loaded {len(input_df)} transactions")

            st.markdown("**Data Preview:**")
            st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("Predict Fraud (All Transactions)", use_container_width=True, type="primary"):
                try:
                    with st.spinner(f"Analyzing {len(input_df)} transactions..."):
                        results = predict_dataframe(input_df, str(model_path))

                    output_df = pd.concat([input_df, results], axis=1)

                    st.markdown("### Prediction Statistics")

                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                    fraudulent_count = (results['fraud_prediction'] == 1).sum()
                    legitimate_count = (results['fraud_prediction'] == 0).sum()
                    avg_fraud_prob = results['fraud_probability'].mean()

                    with stats_col1:
                        st.metric("Total Transactions", len(input_df))

                    with stats_col2:
                        st.metric("Fraudulent", fraudulent_count)

                    with stats_col3:
                        st.metric("Legitimate", legitimate_count)

                    with stats_col4:
                        st.metric("Avg Fraud Probability", f"{avg_fraud_prob*100:.2f}%")

                    st.markdown("### Fraud Distribution")
                    fraud_dist = results['fraud_prediction'].value_counts()
                    col1, col2 = st.columns(2)

                    with col1:
                        st.bar_chart(fraud_dist)

                    with col2:
                        fraud_labels = {0: "Legitimate", 1: "Fraudulent"}
                        pie_data = results['fraud_prediction'].value_counts().rename(fraud_labels)
                        st.write(pie_data)

                    st.markdown("### Detailed Predictions")

                    display_df = output_df.copy()
                    display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x*100:.2f}%")
                    display_df['fraud_prediction'] = display_df['fraud_prediction'].map({0: "Legitimate", 1: "Fraudulent"})

                    st.dataframe(display_df, use_container_width=True)

                    csv = output_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")

        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")

    else:
        st.info("Upload a CSV file to get started with batch predictions")

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
                mime="text/csv"
            )

st.markdown("---")
st.markdown("*Predictions powered by XGBoost trained on PaySim and Sparkov datasets*")
