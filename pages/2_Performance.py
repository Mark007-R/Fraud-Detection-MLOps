"""Model Performance Dashboard - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Performance - SENTINEL", layout="wide", page_icon="S")

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
    <h1>Model Performance</h1>
    <p>Comprehensive evaluation metrics and visualizations for the fraud detection model</p>
</div>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
metrics_file = PROJECT_ROOT / "metrics" / "scores.json"

if not metrics_file.exists():
    st.warning("No metrics found. Please run model evaluation first.")
    st.info("Run `dvc repro` or `python src/evaluate.py` to generate metrics.")
    st.stop()

with open(metrics_file) as f:
    metrics = json.load(f)

# KPI Row
st.markdown("### Key Performance Indicators")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}", help="Overall correctness of predictions")
with col2:
    st.metric("Precision", f"{metrics.get('precision', 0):.4f}", help="Of predicted frauds, how many are actually fraudulent")
with col3:
    st.metric("Recall", f"{metrics.get('recall', 0):.4f}", help="Of actual frauds, how many we detected")
with col4:
    st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}", help="Harmonic mean of precision and recall")
with col5:
    st.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}", help="Area under the ROC curve")
with col6:
    st.metric("Avg Precision", f"{metrics.get('average_precision', 0):.4f}", help="Area under precision-recall curve")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Classification Metrics", "Confusion Matrix", "ROC Curve", "Feature Importance", "Detailed Report"]
)

with tab1:
    st.markdown("### Classification Metrics Overview")
    metrics_detail = {
        'Metric': [
            'Accuracy', 'Precision', 'Recall', 'F1 Score',
            'AUC-ROC', 'Average Precision', 'Support (Neg)', 'Support (Pos)',
        ],
        'Value': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1_score', 0):.4f}",
            f"{metrics.get('auc_roc', 0):.4f}",
            f"{metrics.get('average_precision', 0):.4f}",
            f"{metrics.get('support_neg', 0):.0f}",
            f"{metrics.get('support_pos', 0):.0f}",
        ],
        'Interpretation': [
            'Overall correctness',
            'Positive prediction accuracy',
            'Sensitivity / True Positive Rate',
            'Balance between precision & recall',
            'Model discrimination ability',
            'Area under PR curve',
            'Number of legitimate transactions',
            'Number of fraudulent transactions',
        ],
    }
    metrics_df = pd.DataFrame(metrics_detail)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        metrics_for_chart = {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1': metrics.get('f1_score', 0),
        }

        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics_for_chart.keys()),
                y=list(metrics_for_chart.values()),
                marker_color=['#3b82f6', '#60a5fa', '#06b6d4', '#8b5cf6'],
                text=[f"{v:.4f}" for v in metrics_for_chart.values()],
                textposition='auto',
                textfont=dict(color="#e2e8f0"),
            )
        ])
        fig.update_layout(
            title="Classification Metrics Comparison",
            yaxis_title="Score",
            xaxis_title="Metric",
            height=400,
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        support_data = {
            'Class': ['Legitimate (0)', 'Fraudulent (1)'],
            'Count': [
                metrics.get('support_neg', 0),
                metrics.get('support_pos', 0),
            ],
        }
        support_df = pd.DataFrame(support_data)

        fig = go.Figure(data=[
            go.Pie(
                labels=support_df['Class'],
                values=support_df['Count'],
                marker_colors=['#3b82f6', '#ef4444'],
                textfont=dict(color="#e2e8f0"),
                hole=0.4,
            )
        ])
        fig.update_layout(
            title="Class Distribution in Test Set",
            height=400,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Confusion Matrix")
    st.markdown("Distribution of correct and incorrect predictions")

    tn = metrics.get('true_negatives', 0)
    fp = metrics.get('false_positives', 0)
    fn = metrics.get('false_negatives', 0)
    tp = metrics.get('true_positives', 0)

    cm_data = np.array([[tn, fp], [fn, tp]])

    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=cm_data,
        texttemplate='%{text}',
        textfont={"size": 18, "color": "#e2e8f0"},
        colorscale=[[0, '#1e293b'], [0.5, '#3b82f6'], [1, '#93c5fd']],
        showscale=True,
        colorbar=dict(title="Count", tickfont=dict(color="#94a3b8")),
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        **PLOTLY_LAYOUT,
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matrix Elements")
        cm_df = pd.DataFrame({
            "Element": ["True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)"],
            "Count": [int(tn), int(fp), int(fn), int(tp)],
            "Meaning": [
                "Correctly identified legitimate",
                "Legitimate flagged as fraud",
                "Fraudulent transactions missed",
                "Correctly identified fraud",
            ],
        })
        st.dataframe(cm_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Derived Metrics")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        derived_df = pd.DataFrame({
            "Metric": ["Specificity", "False Positive Rate", "False Negative Rate"],
            "Value": [f"{specificity:.4f}", f"{fpr_val:.4f}", f"{fnr:.4f}"],
            "Meaning": ["True Negative Rate", "False Alarm Rate", "Miss Rate"],
        })
        st.dataframe(derived_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### ROC Curve")
    st.markdown("Receiver Operating Characteristic curve showing model discrimination ability")

    fpr_data = metrics.get('fpr', [])
    tpr_data = metrics.get('tpr', [])
    auc_score = metrics.get('auc_roc', 0)

    if fpr_data and tpr_data:
        fig = go.Figure(data=[
            go.Scatter(
                x=fpr_data,
                y=tpr_data,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_score:.4f})',
                line=dict(color='#3b82f6', width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
            ),
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='#475569', width=2, dash='dash'),
            ),
        ])

        fig.update_layout(
            title=f"ROC Curve (AUC = {auc_score:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            hovermode='closest',
            legend=dict(x=0.55, y=0.1, font=dict(color="#94a3b8")),
            **PLOTLY_LAYOUT,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC curve data not available in metrics file. Re-run evaluation to generate it.")

    quality = "excellent" if auc_score > 0.9 else "good" if auc_score > 0.8 else "reasonable"
    st.markdown(f"""
    **AUC-ROC Score:** {auc_score:.4f} -- The model achieves **{quality}** discrimination ability.
    A score of 0.5 = random, 1.0 = perfect classification.
    """)

with tab4:
    st.markdown("### Feature Importance")

    feature_importance = metrics.get('feature_importance', {})

    if feature_importance:
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance'],
        ).sort_values('Importance', ascending=False).head(20)

        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#93c5fd']],
                ),
                text=importance_df['Importance'].apply(lambda x: f"{x:.4f}"),
                textposition='auto',
                textfont=dict(color="#e2e8f0"),
            )
        ])

        fig.update_layout(
            title="Top 20 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed", gridcolor="rgba(147, 197, 253, 0.08)"),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'yaxis'},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### All Features")
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance data not available. Re-run evaluation to generate it.")

with tab5:
    st.markdown("### Detailed Evaluation Report")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Configuration")
        config_df = pd.DataFrame({
            "Setting": ["Algorithm", "Model Version", "Training Framework", "Data Processing"],
            "Value": ["XGBoost", "v2.0.3", "DVC", "Pandas"],
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Dataset Information")
        total_samples = int(metrics.get('support_neg', 0) + metrics.get('support_pos', 0))
        neg = int(metrics.get('support_neg', 0))
        pos = int(metrics.get('support_pos', 0))
        ratio = f"1:{pos / neg:.4f}" if neg > 0 else "N/A"

        dataset_df = pd.DataFrame({
            "Metric": ["Test Set Size", "Legitimate Samples", "Fraudulent Samples", "Class Balance Ratio"],
            "Value": [str(total_samples), str(neg), str(pos), ratio],
        })
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)

    st.markdown("#### Complete Metrics Summary")

    # Filter out large array fields for display
    display_metrics = {k: v for k, v in metrics.items() if k not in ('fpr', 'tpr', 'feature_importance')}
    metrics_comprehensive = pd.DataFrame({
        'Metric': list(display_metrics.keys()),
        'Value': [str(v) for v in display_metrics.values()],
    })

    st.dataframe(metrics_comprehensive, use_container_width=True, hide_index=True)

    metrics_json = json.dumps(metrics, indent=2)
    st.download_button(
        label="Download Metrics (JSON)",
        data=metrics_json,
        file_name="model_metrics.json",
        mime="application/json",
    )

st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1.5rem 0;
            border-top: 1px solid rgba(147, 197, 253, 0.08);">
    Model trained with DVC pipeline | Metrics generated by evaluate.py
</div>
""", unsafe_allow_html=True)
