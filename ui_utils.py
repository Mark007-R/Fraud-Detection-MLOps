"""Utility functions for Streamlit UI components and operations."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


def load_model_metrics() -> Dict[str, Any]:
    """Load model metrics from JSON file.
    
    Returns
    -------
    Dict[str, Any]
        Model metrics dictionary
    """
    metrics_file = Path("metrics/scores.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {}


def load_features_data(nrows: int = None) -> pd.DataFrame:
    """Load processed features data.
    
    Parameters
    ----------
    nrows : int, optional
        Number of rows to read
    
    Returns
    -------
    pd.DataFrame
        Processed features dataframe
    """
    features_file = Path("data/processed/features.csv")
    if features_file.exists():
        return pd.read_csv(features_file, nrows=nrows)
    return pd.DataFrame()


def load_raw_data(nrows: int = None) -> pd.DataFrame:
    """Load raw transaction data.
    
    Parameters
    ----------
    nrows : int, optional
        Number of rows to read
    
    Returns
    -------
    pd.DataFrame
        Raw transaction dataframe
    """
    combined_file = Path("data/processed/combined_transactions.csv")
    if combined_file.exists():
        return pd.read_csv(combined_file, nrows=nrows)
    return pd.DataFrame()


def format_metric(value: float, metric_type: str = "percentage") -> str:
    """Format numeric values for display.
    
    Parameters
    ----------
    value : float
        Value to format
    metric_type : str
        Type of metric: 'percentage', 'currency', 'decimal'
    
    Returns
    -------
    str
        Formatted string
    """
    if metric_type == "percentage":
        return f"{value*100:.2f}%"
    elif metric_type == "currency":
        return f"${value:,.2f}"
    else:
        return f"{value:.4f}"


def create_metric_cards(metrics: Dict[str, float]) -> None:
    """Create metric card rows in Streamlit.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    """
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(name, f"{value:.4f}")


def validate_transaction_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate transaction dataframe has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe to validate
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    required_columns = ['amount', 'merchant_id', 'customer_id', 'type']
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    return True, "Valid"


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistics for numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze
    
    Returns
    -------
    pd.DataFrame
        Statistics dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    return stats


def detect_outliers(df: pd.DataFrame, column: str, method: str = "iqr") -> pd.DataFrame:
    """Detect outliers in a column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing the column
    column : str
        Column name to analyze
    method : str
        Detection method: 'iqr' or 'zscore'
    
    Returns
    -------
    pd.DataFrame
        Rows identified as outliers
    """
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method == "zscore":
        from scipy import stats
        clean = df[column].dropna()
        if len(clean) < 2 or clean.std() == 0:
            return pd.DataFrame()
        mask = np.abs(stats.zscore(df[column].fillna(df[column].median()))) > 3
        return df[mask]
    
    return pd.DataFrame()


def calculate_fraud_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate fraud-related statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'is_fraud' column
    
    Returns
    -------
    Dict[str, Any]
        Fraud statistics
    """
    if 'is_fraud' not in df.columns:
        return {}
    
    total = len(df)
    frauds = (df['is_fraud'] == 1).sum()
    legitimate = total - frauds
    
    stats = {
        'total_transactions': total,
        'fraudulent_count': frauds,
        'legitimate_count': legitimate,
        'fraud_rate': frauds / total if total > 0 else 0,
        'fraudulent_avg_amount': df[df['is_fraud'] == 1]['amount'].mean() if 'amount' in df.columns else 0,
        'legitimate_avg_amount': df[df['is_fraud'] == 0]['amount'].mean() if 'amount' in df.columns else 0,
    }
    
    return stats


def get_top_fraud_features(df: pd.DataFrame, n_features: int = 10) -> pd.DataFrame:
    """Identify features most predictive of fraud.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'is_fraud' column
    n_features : int
        Number of features to return
    
    Returns
    -------
    pd.DataFrame
        Top features and their correlation with fraud
    """
    if 'is_fraud' not in df.columns:
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = []
    
    for col in numeric_cols:
        if col != 'is_fraud':
            corr = df[col].corr(df['is_fraud'])
            correlations.append({
                'Feature': col,
                'Fraud Correlation': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Fraud Correlation', ascending=False)
    return corr_df.head(n_features)


@st.cache_data
def get_cached_metrics() -> Dict[str, Any]:
    """Get cached model metrics.
    
    Returns
    -------
    Dict[str, Any]
        Model metrics dictionary
    """
    return load_model_metrics()


@st.cache_data
def get_cached_features(nrows: int = 1000) -> pd.DataFrame:
    """Get cached features data.
    
    Parameters
    ----------
    nrows : int
        Number of rows to cache
    
    Returns
    -------
    pd.DataFrame
        Features dataframe
    """
    return load_features_data(nrows=nrows)


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for better UI display.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to format
    
    Returns
    -------
    pd.DataFrame
        Formatted dataframe
    """
    formatted = df.copy()
    
    # Format float columns to 2 decimal places
    float_cols = formatted.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        if 'probability' in col.lower() or 'rate' in col.lower():
            formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        elif 'amount' in col.lower():
            formatted[col] = formatted[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        else:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    return formatted


def create_download_button(df: pd.DataFrame, filename: str, file_format: str = "csv") -> None:
    """Create a file download button for a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to download
    filename : str
        Output filename
    file_format : str
        File format: 'csv' or 'json'
    """
    if file_format == "csv":
        data = df.to_csv(index=False)
        mime = "text/csv"
    elif file_format == "json":
        data = df.to_json(orient='records')
        mime = "application/json"
    else:
        return
    
    st.download_button(
        label=f"📥 Download {file_format.upper()}",
        data=data,
        file_name=filename,
        mime=mime
    )
