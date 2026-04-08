"""Model Performance Dashboard - Streamlit App"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Performance - SENTINEL", layout="wide")

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

st.markdown("# Model Performance Dashboard")
st.markdown("Comprehensive evaluation metrics and visualizations for the fraud detection model")
metrics_file = Path("metrics/scores.json")

if not metrics_file.exists():
    st.warning("No metrics found. Please run model evaluation first.")
    st.info("Run `dvc repro` or `python src/evaluate.py` to generate metrics.")
    st.stop()

with open(metrics_file) as f:
    metrics = json.load(f)
st.markdown("## Key Performance Indicators")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0):.4f}",
        help="Overall correctness of predictions"
    )

with col2:
    st.metric(
        "Precision",
        f"{metrics.get('precision', 0):.4f}",
        help="Of predicted frauds, how many are actually fraudulent"
    )

with col3:
    st.metric(
        "Recall",
        f"{metrics.get('recall', 0):.4f}",
        help="Of actual frauds, how many we detected"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{metrics.get('f1', 0):.4f}",
        help="Harmonic mean of precision and recall"
    )

with col5:
    st.metric(
        "AUC-ROC",
        f"{metrics.get('auc_roc', 0):.4f}",
        help="Area under the ROC curve"
    )

with col6:
    st.metric(
        "Avg Precision",
        f"{metrics.get('avg_precision', 0):.4f}",
        help="Area under precision-recall curve"
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Classification Metrics", "Confusion Matrix", "ROC Curve", "Feature Importance", "Detailed Report"]
)

with tab1:
    st.markdown("## Classification Metrics Overview")
    metrics_detail = {
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC-ROC',
            'Average Precision',
            'Support (Neg)',
            'Support (Pos)'
        ],
        'Value': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('auc_roc', 0):.4f}",
            f"{metrics.get('avg_precision', 0):.4f}",
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
            'F1': metrics.get('f1', 0),
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics_for_chart.keys()),
                y=list(metrics_for_chart.values()),
                marker_color=['#3b82f6', '#60a5fa', '#f59e0b', '#1e3a8a'],
                text=[f"{v:.4f}" for v in metrics_for_chart.values()],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Classification Metrics Comparison",
            yaxis_title="Score",
            xaxis_title="Metric",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        support_data = {
            'Class': ['Legitimate (0)', 'Fraudulent (1)'],
            'Count': [
                metrics.get('support_neg', 0),
                metrics.get('support_pos', 0),
            ]
        }
        support_df = pd.DataFrame(support_data)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=support_df['Class'],
                values=support_df['Count'],
                marker_colors=['#60a5fa', '#1e3a8a'],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Class Distribution in Test Set",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## Confusion Matrix")
    st.markdown("Shows the distribution of correct and incorrect predictions")
    
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
        textfont={"size": 14},
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix Elements")
        st.markdown(f"""
        - **True Negatives (TN)**: {int(tn)} - Correctly identified legitimate transactions
        - **False Positives (FP)**: {int(fp)} - Legitimate transactions flagged as fraud
        - **False Negatives (FN)**: {int(fn)} - Fraudulent transactions missed
        - **True Positives (TP)**: {int(tp)} - Correctly identified fraudulent transactions
        """)
    
    with col2:
        st.markdown("### Derived Metrics")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        st.markdown(f"""
        - **Specificity**: {specificity:.4f} - True Negative Rate
        - **False Positive Rate**: {fpr:.4f} - False Alarm Rate
        - **False Negative Rate**: {fnr:.4f} - Miss Rate
        """)

with tab3:
    st.markdown("## ROC Curve")
    st.markdown("Receiver Operating Characteristic curve showing model's discriminative ability")
    
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
            ),
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='#334155', width=2, dash='dash'),
            )
        ])
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            hovermode='closest',
            legend=dict(x=0.6, y=0.1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC curve data not available in metrics file")
    
    st.markdown(f"""
    ### Interpretation
    
    - **AUC-ROC Score**: {auc_score:.4f}
    - The ROC curve plots True Positive Rate vs False Positive Rate
    - A curve closer to the top-left indicates better performance
    - AUC = 0.5 means random classifier, AUC = 1.0 means perfect classifier
    - Our model achieves an AUC of **{auc_score:.4f}**, indicating {"excellent" if auc_score > 0.9 else "good" if auc_score > 0.8 else "reasonable"} discrimination ability
    """)

with tab4:
    st.markdown("## Feature Importance")
    
    feature_importance = metrics.get('feature_importance', {})
    
    if feature_importance:
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(20)
        
        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color='#3b82f6',
                text=importance_df['Importance'].apply(lambda x: f"{x:.4f}"),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top 20 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed"),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### All Features")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance data not available")

with tab5:
    st.markdown("## Detailed Evaluation Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        st.markdown(f"""
        - **Algorithm**: XGBoost
        - **Model Version**: v2.0.3
        - **Training Framework**: DVC
        - **Data Processing**: Dask
        """)
    
    with col2:
        st.markdown("### Dataset Information")
        st.markdown(f"""
        - **Test Set Size**: {int(metrics.get('support_neg', 0) + metrics.get('support_pos', 0))}
        - **Legitimate Samples**: {int(metrics.get('support_neg', 0))}
        - **Fraudulent Samples**: {int(metrics.get('support_pos', 0))}
        - **Class Balance Ratio**: 1:{metrics.get('support_pos', 1)/metrics.get('support_neg', 1):.2f}
        """)
    
    st.markdown("### Complete Metrics Summary")
    
    metrics_comprehensive = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [str(v) for v in metrics.values()]
    })
    
    st.dataframe(metrics_comprehensive, use_container_width=True)
    
    metrics_json = json.dumps(metrics, indent=2)
    st.download_button(
        label="Download Metrics (JSON)",
        data=metrics_json,
        file_name="model_metrics.json",
        mime="application/json"
    )

st.markdown("---")
st.markdown("*Last Updated: Model trained with DVC pipeline | Metrics generated by evaluate.py*")

