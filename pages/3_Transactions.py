"""Transaction Analysis & Visualization - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Transactions - SENTINEL", layout="wide")

# Custom CSS for light blue and black theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e6f4ff 0%, #bcdcff 100%);
    }
    
    h1, h2, h3, label, p {
        color: #0b0f19 !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #93c5fd 0%, #bfdbfe 100%);
        border: 2px solid #0b0f19;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25);
    }
    
    .stTabs [role="tablist"] button[aria-selected="true"] {
        border-bottom: 3px solid #0b0f19 !important;
        color: #0b0f19 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Transaction Analysis & Visualization")
st.markdown("Explore transaction patterns, anomalies, and statistical insights")

features_file = Path("data/processed/features.csv")
combined_file = Path("data/processed/combined_transactions.csv")

data_loaded = False
df = None

if features_file.exists():
    try:
        df = pd.read_csv(features_file)
        data_loaded = True
        st.success("Features data loaded successfully")
    except Exception as e:
        st.warning(f"Could not load features data: {e}")

if not data_loaded:
    st.warning("No transaction data found. Please run the preprocessing pipeline first.")
    st.info("Run `dvc repro` to generate processed transaction data.")
    st.stop()

st.markdown("## Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    if 'is_fraud' in df.columns:
        fraud_count = (df['is_fraud'] == 1).sum()
        st.metric("Fraudulent Cases", fraud_count)
    else:
        st.metric("Columns", len(df.columns))

with col3:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    st.metric("Numeric Features", len(numeric_cols))

with col4:
    if 'amount' in df.columns:
        st.metric("Total Amount", f"${df['amount'].sum():,.0f}")

with st.expander("View Raw Data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("### Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Amount Analysis", "Pattern Detection", "Time-based Trends", "Feature Distributions", "Correlations"]
)

with tab1:
    st.markdown("## Transaction Amount Analysis")
    
    if 'amount' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='amount',
                nbins=50,
                title='Transaction Amount Distribution',
                labels={'amount': 'Amount ($)', 'count': 'Frequency'}
            )
            fig.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'is_fraud' in df.columns:
                fig = px.box(
                    df,
                    y='amount',
                    x='is_fraud',
                    title='Amount Distribution by Class',
                    labels={'is_fraud': 'Class', 'amount': 'Amount ($)'},
                    color='is_fraud',
                    color_discrete_map={0: '#60a5fa', 1: '#1e3a8a'}
                )
            else:
                fig = px.box(df, y='amount', title='Amount Distribution')
            
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Amount Statistics")
            amount_stats = {
                'Mean': f"${df['amount'].mean():,.2f}",
                'Median': f"${df['amount'].median():,.2f}",
                'Std Dev': f"${df['amount'].std():,.2f}",
                'Min': f"${df['amount'].min():,.2f}",
                'Max': f"${df['amount'].max():,.2f}",
            }
            for key, value in amount_stats.items():
                st.write(f"**{key}**: {value}")
        
        with col4:
            st.markdown("### Amount Brackets")
            if 'amount_bracket' in df.columns:
                bracket_dist = df['amount_bracket'].value_counts()
                fig = px.pie(
                    values=bracket_dist.values,
                    names=bracket_dist.index,
                    title='Transactions by Amount Bracket',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                bins = [0, 1000, 5000, 10000, 50000, float('inf')]
                labels = ['<$1K', '$1K-$5K', '$5K-$10K', '$10K-$50K', '>$50K']
                bracket_dist = pd.cut(df['amount'], bins=bins, labels=labels).value_counts()
                
                fig = px.bar(
                    x=bracket_dist.index,
                    y=bracket_dist.values,
                    title='Transactions by Amount Bracket',
                    labels={'x': 'Amount Range', 'y': 'Count'},
                )
                fig.update_traces(marker_color='#60a5fa')
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## Pattern & Anomaly Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fraud Distribution")
        
        if 'is_fraud' in df.columns:
            fraud_dist = df['is_fraud'].value_counts().rename({0: 'Legitimate', 1: 'Fraudulent'})
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=fraud_dist.index,
                    values=fraud_dist.values,
                    marker_colors=['#60a5fa', '#1e3a8a'],
                )
            ])
            fig.update_layout(title="Fraud vs Legitimate Transactions")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Class Statistics")
            fraud_rate = (df['is_fraud'] == 1).sum() / len(df) * 100
            st.write(f"Fraud Rate: **{fraud_rate:.2f}%**")
            st.write(f"Legitimate: **{(df['is_fraud'] == 0).sum()}** transactions")
            st.write(f"Fraudulent: **{(df['is_fraud'] == 1).sum()}** transactions")
    
    with col2:
        st.markdown("### High-Risk Patterns")
        
        if 'is_fraud' in df.columns and 'amount' in df.columns:
            high_amount_frauds = df[(df['is_fraud'] == 1) & (df['amount'] > df['amount'].quantile(0.75))]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("High-Value Frauds", len(high_amount_frauds))
            with col_b:
                avg_high_fraud = high_amount_frauds['amount'].mean()
                st.metric("Avg Amount", f"${avg_high_fraud:,.0f}")
            with col_c:
                pct_high = len(high_amount_frauds) / (df['is_fraud'] == 1).sum() * 100
                st.metric("% of All Frauds", f"{pct_high:.1f}%")
        
        st.markdown("#### Top Anomaly Features")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            anomaly_scores = []
            for col in numeric_cols:
                if col != 'is_fraud':
                    skew = df[col].skew()
                    anomaly_scores.append({'Feature': col, 'Skewness': abs(skew)})
            
            if anomaly_scores:
                anomaly_df = pd.DataFrame(anomaly_scores).sort_values('Skewness', ascending=False).head(10)
                st.dataframe(anomaly_df, use_container_width=True)

with tab3:
    st.markdown("## Time-based Trends")
    
    st.info("Time-based analysis depends on temporal features in your dataset")
    
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]
    
    if time_cols:
        selected_time_col = st.selectbox("Select time column", time_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_dist = df[selected_time_col].value_counts().sort_index()
            fig = px.line(
                x=time_dist.index,
                y=time_dist.values,
                title=f'Transactions Over {selected_time_col}',
                labels={'x': selected_time_col, 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'is_fraud' in df.columns:
                fraud_by_time = df.groupby(selected_time_col)['is_fraud'].sum()
                fig = px.bar(
                    x=fraud_by_time.index,
                    y=fraud_by_time.values,
                    title=f'Frauds Over {selected_time_col}',
                    labels={'x': selected_time_col, 'y': 'Fraud Count'}
                )
                fig.update_traces(marker_color='#1e3a8a')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        ### Time-based Analysis Not Available
        
        To enable time-based analysis, ensure your dataset includes:
        - Date/time columns
        - Hour of transaction
        - Day of week
        - Month/year information
        
        These features are typically engineered during preprocessing.
        """)

with tab4:
    st.markdown("## Feature Distributions")
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if 'is_fraud' not in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != 'is_fraud']
    
    if numeric_cols:
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:3]
        )
        
        if selected_features:
            cols_per_row = 2
            for i in range(0, len(selected_features), cols_per_row):
                col1, col2 = st.columns(cols_per_row)
                columns = [col1, col2]
                
                for j, feature in enumerate(selected_features[i:i+cols_per_row]):
                    with columns[j]:
                        fig = px.histogram(
                            df,
                            x=feature,
                            nbins=30,
                            title=f'Distribution of {feature}',
                            labels={feature: feature, 'count': 'Frequency'}
                        )
                        fig.update_traces(marker_color='#3b82f6')
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric features available for distribution analysis")

with tab5:
    st.markdown("## Feature Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=700,
            xaxis_title='Features',
            yaxis_title='Features'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Highly Correlated Features")
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': f"{corr_val:.4f}"
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False, key=abs)
            st.dataframe(high_corr_df, use_container_width=True)
        else:
            st.info("No highly correlated feature pairs found (threshold > 0.7)")
    else:
        st.info("Need at least 2 numeric features for correlation analysis")

st.markdown("---")
st.markdown("## Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Features (CSV)",
        data=csv,
        file_name="transaction_features.csv",
        mime="text/csv"
    )

with col2:
    st.info("Use CSV export for data analysis")

with col3:
    st.write("")

st.markdown("---")
st.markdown("*Transaction data processed using Dask | Features engineered in preprocessing stage*")

