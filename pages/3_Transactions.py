"""Transaction Analysis & Visualization - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Transactions - SENTINEL", layout="wide", page_icon="S")

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

    h1, h2, h3 { color: #e2e8f0 !important; }
    p, label, .stMarkdown { color: #cbd5e1 !important; }

    .page-header h1 {
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 2.2rem;
    }

    .page-header p { color: #94a3b8 !important; }

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

    .stMetric {
        background: linear-gradient(135deg, rgba(147, 197, 253, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(147, 197, 253, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }

    .stMetric label { color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(147, 197, 253, 0.2), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Plotly dark theme layout
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17, 24, 39, 0.5)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=16),
    xaxis=dict(gridcolor="rgba(147, 197, 253, 0.08)", zerolinecolor="rgba(147, 197, 253, 0.1)"),
    yaxis=dict(gridcolor="rgba(147, 197, 253, 0.08)", zerolinecolor="rgba(147, 197, 253, 0.1)"),
    margin=dict(l=40, r=40, t=50, b=40),
)

st.markdown("""
<div class="page-header">
    <h1>Transaction Analysis</h1>
    <p>Explore transaction patterns, anomalies, and statistical insights</p>
</div>
""", unsafe_allow_html=True)

features_file = Path("data/processed/features.csv")

SAMPLE_SIZE = 50_000

data_loaded = False
df = None

if features_file.exists():
    try:
        df = pd.read_csv(features_file, nrows=SAMPLE_SIZE)
        data_loaded = True
        st.success(f"Loaded {len(df):,} transactions (sample) with {len(df.columns)} features")
    except Exception as e:
        st.warning(f"Could not load features data: {e}")

if not data_loaded:
    st.warning("No transaction data found. Please run the preprocessing pipeline first.")
    st.info("Run `dvc repro` to generate processed transaction data.")
    st.stop()

# Overview KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    if 'is_fraud' in df.columns:
        fraud_count = int((df['is_fraud'] == 1).sum())
        st.metric("Fraudulent Cases", f"{fraud_count:,}")
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
    st.markdown("#### Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Amount Analysis", "Pattern Detection", "Time-based Trends", "Feature Distributions", "Correlations"]
)

with tab1:
    st.markdown("### Transaction Amount Analysis")

    if 'amount' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x='amount', nbins=50,
                title='Transaction Amount Distribution',
                labels={'amount': 'Amount ($)', 'count': 'Frequency'},
            )
            fig.update_traces(marker_color='#3b82f6')
            fig.update_layout(height=400, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'is_fraud' in df.columns:
                fig = px.box(
                    df, y='amount', x='is_fraud',
                    title='Amount Distribution by Class',
                    labels={'is_fraud': 'Class', 'amount': 'Amount ($)'},
                    color='is_fraud',
                    color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
                )
            else:
                fig = px.box(df, y='amount', title='Amount Distribution')

            fig.update_layout(height=400, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Amount Statistics")
            amount_stats = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
                "Value": [
                    f"${df['amount'].mean():,.2f}",
                    f"${df['amount'].median():,.2f}",
                    f"${df['amount'].std():,.2f}",
                    f"${df['amount'].min():,.2f}",
                    f"${df['amount'].max():,.2f}",
                ],
            })
            st.dataframe(amount_stats, use_container_width=True, hide_index=True)

        with col4:
            st.markdown("#### Amount Brackets")
            bins = [0, 1000, 5000, 10000, 50000, float('inf')]
            labels = ['<$1K', '$1K-$5K', '$5K-$10K', '$10K-$50K', '>$50K']
            bracket_dist = pd.cut(df['amount'], bins=bins, labels=labels).value_counts()

            fig = go.Figure(data=[
                go.Bar(
                    x=bracket_dist.index.astype(str),
                    y=bracket_dist.values,
                    marker=dict(
                        color=['#1e3a5f', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
                    ),
                    text=bracket_dist.values,
                    textposition='auto',
                    textfont=dict(color="#e2e8f0"),
                )
            ])
            fig.update_layout(
                title='Transactions by Amount Bracket',
                xaxis_title='Amount Range',
                yaxis_title='Count',
                height=400,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Pattern & Anomaly Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fraud Distribution")

        if 'is_fraud' in df.columns:
            fraud_dist = df['is_fraud'].value_counts().rename({0: 'Legitimate', 1: 'Fraudulent'})

            fig = go.Figure(data=[
                go.Pie(
                    labels=fraud_dist.index,
                    values=fraud_dist.values,
                    marker_colors=['#3b82f6', '#ef4444'],
                    hole=0.45,
                    textfont=dict(color="#e2e8f0"),
                )
            ])
            fig.update_layout(
                title="Fraud vs Legitimate Transactions",
                height=400,
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
            )
            st.plotly_chart(fig, use_container_width=True)

            fraud_rate = (df['is_fraud'] == 1).sum() / len(df) * 100
            fraud_stats = pd.DataFrame({
                "Metric": ["Fraud Rate", "Legitimate Count", "Fraudulent Count"],
                "Value": [f"{fraud_rate:.2f}%", f"{(df['is_fraud'] == 0).sum():,}", f"{(df['is_fraud'] == 1).sum():,}"],
            })
            st.dataframe(fraud_stats, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### High-Risk Patterns")

        if 'is_fraud' in df.columns and 'amount' in df.columns:
            high_amount_frauds = df[(df['is_fraud'] == 1) & (df['amount'] > df['amount'].quantile(0.75))]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("High-Value Frauds", len(high_amount_frauds))
            with col_b:
                avg_high_fraud = high_amount_frauds['amount'].mean() if len(high_amount_frauds) > 0 else 0
                st.metric("Avg Amount", f"${avg_high_fraud:,.0f}")
            with col_c:
                total_frauds = (df['is_fraud'] == 1).sum()
                pct_high = len(high_amount_frauds) / total_frauds * 100 if total_frauds > 0 else 0
                st.metric("% of All Frauds", f"{pct_high:.1f}%")

        st.markdown("#### Top Skewed Features")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            anomaly_scores = []
            for col in numeric_cols:
                if col != 'is_fraud':
                    skew = df[col].skew()
                    anomaly_scores.append({'Feature': col, 'Skewness': round(abs(skew), 4)})

            if anomaly_scores:
                anomaly_df = pd.DataFrame(anomaly_scores).sort_values('Skewness', ascending=False).head(10)
                st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Time-based Trends")

    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'hour' in col.lower()]

    if time_cols:
        selected_time_col = st.selectbox("Select time column", time_cols)

        col1, col2 = st.columns(2)

        with col1:
            time_dist = df[selected_time_col].value_counts().sort_index()
            fig = go.Figure(data=[
                go.Scatter(
                    x=time_dist.index,
                    y=time_dist.values,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(color='#60a5fa', size=6),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)',
                )
            ])
            fig.update_layout(
                title=f'Transactions Over {selected_time_col}',
                xaxis_title=selected_time_col,
                yaxis_title='Count',
                height=400,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'is_fraud' in df.columns:
                fraud_by_time = df.groupby(selected_time_col)['is_fraud'].sum()
                fig = go.Figure(data=[
                    go.Bar(
                        x=fraud_by_time.index,
                        y=fraud_by_time.values,
                        marker_color='#ef4444',
                        text=fraud_by_time.values,
                        textposition='auto',
                        textfont=dict(color="#e2e8f0"),
                    )
                ])
                fig.update_layout(
                    title=f'Frauds Over {selected_time_col}',
                    xaxis_title=selected_time_col,
                    yaxis_title='Fraud Count',
                    height=400,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("""
        **Time-based analysis not available.** Ensure your dataset includes temporal features
        such as date/time columns, hour of transaction, or day of week.
        """)

with tab4:
    st.markdown("### Feature Distributions")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [col for col in numeric_cols if col != 'is_fraud']

    if numeric_cols:
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
        )

        if selected_features:
            cols_per_row = 2
            for i in range(0, len(selected_features), cols_per_row):
                col1, col2 = st.columns(cols_per_row)
                columns = [col1, col2]

                for j, feature in enumerate(selected_features[i:i + cols_per_row]):
                    with columns[j]:
                        fig = px.histogram(
                            df, x=feature, nbins=30,
                            title=f'Distribution of {feature}',
                            labels={feature: feature, 'count': 'Frequency'},
                        )
                        fig.update_traces(marker_color='#3b82f6')
                        fig.update_layout(height=350, **PLOTLY_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric features available for distribution analysis")

with tab5:
    st.markdown("### Feature Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        # Limit to top features for readable heatmap
        max_features = st.slider("Max features to display", min_value=5, max_value=min(30, len(numeric_cols)), value=min(15, len(numeric_cols)))
        display_cols = numeric_cols[:max_features]

        correlation_matrix = df[display_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale=[[0, '#1e3a5f'], [0.25, '#2563eb'], [0.5, '#1e293b'], [0.75, '#dc2626'], [1, '#fca5a5']],
            zmid=0,
            zmin=-1,
            zmax=1,
            textfont=dict(color="#e2e8f0"),
        ))

        fig.update_layout(
            title='Feature Correlation Matrix',
            height=700,
            xaxis_title='Features',
            yaxis_title='Features',
            **PLOTLY_LAYOUT,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Highly Correlated Pairs (|r| > 0.7)")

        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': f"{corr_val:.4f}",
                    })

        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False, key=abs)
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
        else:
            st.info("No highly correlated feature pairs found (threshold > 0.7)")
    else:
        st.info("Need at least 2 numeric features for correlation analysis")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("### Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Features (CSV)",
        data=csv,
        file_name="transaction_features.csv",
        mime="text/csv",
    )

with col2:
    st.markdown(f"**{len(df):,}** rows, **{len(df.columns)}** columns available for export")

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1.5rem 0;
            border-top: 1px solid rgba(147, 197, 253, 0.08);">
    Transaction data processed using Dask | Features engineered in preprocessing stage
</div>
""", unsafe_allow_html=True)
