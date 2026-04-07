"""Stage 3: Feature engineering for fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Load YAML parameters.

    Parameters
    ----------
    path : str
        Path to params file.

    Returns
    -------
    dict
        Parsed params.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_amount_bin_partition(series: pd.Series) -> pd.Series:
    """Create amount bins in a pandas partition.

    Parameters
    ----------
    series : pd.Series
        Amount series.

    Returns
    -------
    pd.Series
        Categorical bin labels.
    """
    bins = [-np.inf, 1000, 10000, 100000, np.inf]
    labels = ["low", "medium", "high", "very_high"]
    return pd.cut(series, bins=bins, labels=labels).astype("string")


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering using pandas operations.

    Parameters
    ----------
    df : pd.DataFrame
        Combined standardized transaction dataset.

    Returns
    -------
    pd.DataFrame
        Engineered feature dataframe with encoded categoricals.
    """
    required = {
        "amount",
        "is_fraud",
        "transaction_type",
        "hour_of_day",
        "day_of_month",
        "balance_change_orig",
        "balance_ratio",
        "has_balance_info",
        "source",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[Preprocess] Missing required columns: {sorted(missing)}")

    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).astype("int64")
    day = pd.to_numeric(df["day_of_month"], errors="coerce").fillna(1).astype("int64")

    bal_change = pd.to_numeric(df["balance_change_orig"], errors="coerce")
    bal_ratio = pd.to_numeric(df["balance_ratio"], errors="coerce")

    bal_change_abs = bal_change.abs()
    median_change = float(bal_change_abs.quantile(0.5))
    bal_change_abs = bal_change_abs.fillna(median_change)

    bal_ratio_clipped = bal_ratio.fillna(0.0).clip(-10, 10)

    out = df.assign(
        amount=amount,
        is_fraud=pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype("int64"),
        tx_amount_log=np.log1p(amount),
        is_high_amount=(amount > 200000).astype("int64"),
        hour_sin=np.sin(2.0 * np.pi * hour / 24.0),
        hour_cos=np.cos(2.0 * np.pi * hour / 24.0),
        day_sin=np.sin(2.0 * np.pi * (day % 31) / 31.0),
        day_cos=np.cos(2.0 * np.pi * (day % 31) / 31.0),
        balance_change_abs=bal_change_abs,
        balance_ratio_clipped=bal_ratio_clipped,
    )

    out["amount_bin"] = _add_amount_bin_partition(out["amount"])

    out = pd.get_dummies(out, columns=["transaction_type", "source", "amount_bin"], dtype="int64")
    return out


def engineer_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering logic in pandas for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input standardized dataframe.

    Returns
    -------
    pd.DataFrame
        Engineered dataframe.
    """
    amount = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    hour = pd.to_numeric(df.get("hour_of_day", 0), errors="coerce").fillna(0).astype(int)
    day = pd.to_numeric(df.get("day_of_month", 1), errors="coerce").fillna(1).astype(int)

    bal_change = pd.to_numeric(df.get("balance_change_orig", 0.0), errors="coerce")
    bal_ratio = pd.to_numeric(df.get("balance_ratio", 0.0), errors="coerce")

    bal_change_abs = bal_change.abs()
    med = float(bal_change_abs.dropna().median()) if bal_change_abs.notna().any() else 0.0

    out = df.copy()
    out["amount"] = amount
    out["tx_amount_log"] = np.log1p(amount)
    out["is_high_amount"] = (amount > 200000).astype(int)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["day_sin"] = np.sin(2.0 * np.pi * (day % 31) / 31.0)
    out["day_cos"] = np.cos(2.0 * np.pi * (day % 31) / 31.0)
    out["balance_change_abs"] = bal_change_abs.fillna(med)
    out["balance_ratio_clipped"] = bal_ratio.fillna(0.0).clip(-10, 10)
    out["amount_bin"] = pd.cut(
        out["amount"],
        bins=[-np.inf, 1000, 10000, 100000, np.inf],
        labels=["low", "medium", "high", "very_high"],
    ).astype("string")

    for col in ("transaction_type", "source"):
        if col not in out.columns:
            out[col] = "unknown"

    out = pd.get_dummies(out, columns=["transaction_type", "source", "amount_bin"], dtype=int)
    return out


def align_feature_columns(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Align a dataframe to model feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    feature_columns : Iterable[str]
        Expected feature column order.

    Returns
    -------
    pd.DataFrame
        Aligned dataframe.
    """
    aligned = df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    extra = [c for c in aligned.columns if c not in feature_columns]
    if extra:
        aligned = aligned.drop(columns=extra)
    return aligned[list(feature_columns)]


def main() -> None:
    """Run Stage 3 pipeline step."""
    params = load_params()
    cfg = params.get("preprocess", {})

    input_path = cfg.get("input", "data/processed/combined_transactions.csv")
    output_path = cfg.get("output", "data/processed/features.csv")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"[Preprocess] Input file not found: {input_path}")

    print(f"[Preprocess] Loading {input_path}")
    df = pd.read_csv(input_path)

    features_df = engineer_features_df(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    feature_names = list(features_df.columns)
    shape_rows = int(features_df.shape[0])
    null_counts = features_df.isnull().sum().to_dict()
    fraud_dist = features_df["is_fraud"].value_counts().to_dict()

    print(f"[Preprocess] Saved features to {output_path}")
    print(f"[Preprocess] Shape: ({shape_rows}, {len(feature_names)})")
    print(f"[Preprocess] Feature names: {feature_names}")
    print(f"[Preprocess] Null counts: {null_counts}")
    print(f"[Preprocess] Fraud distribution: {fraud_dist}")


if __name__ == "__main__":
    main()
