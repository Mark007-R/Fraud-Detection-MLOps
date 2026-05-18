"""Stage 3: Feature engineering for fraud detection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_params


def compute_feature_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """Compute training-time thresholds used by percentile / z-score features.

    These values MUST be persisted and reused at inference time — computing
    them from a single inference row produces degenerate features (e.g.,
    `is_p95_amount` becomes 0 for every row since `amount.quantile(0.95)`
    equals the row's own amount).

    Parameters
    ----------
    df : pd.DataFrame
        Standardized transaction dataframe containing an `amount` column.

    Returns
    -------
    dict[str, float]
        Thresholds: amount_p95, amount_p99, amount_mean, amount_std.
    """
    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return {
        "amount_p95": float(amount.quantile(0.95)) if len(amount) > 0 else 200000.0,
        "amount_p99": float(amount.quantile(0.99)) if len(amount) > 0 else 500000.0,
        "amount_mean": float(amount.mean()) if len(amount) > 0 else 0.0,
        "amount_std": float(amount.std()) if len(amount) > 1 else 1.0,
    }


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
        "txn_timestamp",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[Preprocess] Missing required columns: {sorted(missing)}")

    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).astype("int64")
    day = pd.to_numeric(df["day_of_month"], errors="coerce").fillna(1).astype("int64")
    ts = pd.to_numeric(df["txn_timestamp"], errors="coerce").fillna(0).astype("int64")

    bal_change = pd.to_numeric(df["balance_change_orig"], errors="coerce")
    bal_ratio = pd.to_numeric(df["balance_ratio"], errors="coerce")

    bal_change_abs = bal_change.abs()
    median_change = float(bal_change_abs.quantile(0.5))
    bal_change_abs = bal_change_abs.fillna(median_change)

    bal_ratio_clipped = bal_ratio.fillna(0.0).clip(-10, 10)

    # Percentile-based amount flags
    amount_p95 = float(amount.quantile(0.95)) if len(amount) > 0 else 200000
    amount_p99 = float(amount.quantile(0.99)) if len(amount) > 0 else 500000

    out = df.assign(
        amount=amount,
        is_fraud=pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype("int64"),
        txn_timestamp=ts,
        tx_amount_log=np.log1p(amount),
        is_high_amount=(amount > 200000).astype("int64"),
        is_p95_amount=(amount > amount_p95).astype("int64"),
        is_p99_amount=(amount > amount_p99).astype("int64"),
        amount_zscore=((amount - amount.mean()) / (amount.std() + 1e-8)),
        hour_sin=np.sin(2.0 * np.pi * hour / 24.0),
        hour_cos=np.cos(2.0 * np.pi * hour / 24.0),
        day_sin=np.sin(2.0 * np.pi * (day % 31) / 31.0),
        day_cos=np.cos(2.0 * np.pi * (day % 31) / 31.0),
        is_night=(((hour >= 0) & (hour < 6)) | (hour >= 22)).astype("int64"),
        is_weekend=((day % 7).isin([5, 6])).astype("int64"),
        balance_change_abs=bal_change_abs,
        balance_ratio_clipped=bal_ratio_clipped,
        balance_change_log=np.log1p(bal_change_abs),
    )

    out["amount_bin"] = _add_amount_bin_partition(out["amount"])

    out = pd.get_dummies(out, columns=["transaction_type", "source", "amount_bin"], dtype="int64")
    return out


def engineer_features_pandas(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply the same feature engineering logic in pandas for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input standardized dataframe.
    thresholds : dict[str, float] | None
        Training-time thresholds (amount_p95, amount_p99, amount_mean,
        amount_std). When provided, these are used instead of recomputing
        from `df` — which is essential for single-row or small-batch
        inference where inference-data statistics are meaningless.

    Returns
    -------
    pd.DataFrame
        Engineered dataframe.
    """
    def _series_or_default(column: str, default: float | int) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index)

    amount = _series_or_default("amount", 0.0).fillna(0.0)
    hour = _series_or_default("hour_of_day", 0).fillna(0).astype(int)
    day = _series_or_default("day_of_month", 1).fillna(1).astype(int)

    bal_change = _series_or_default("balance_change_orig", 0.0)
    bal_ratio = _series_or_default("balance_ratio", 0.0)

    bal_change_abs = bal_change.abs()
    med = float(bal_change_abs.dropna().median()) if bal_change_abs.notna().any() else 0.0

    if thresholds:
        # Prefer training-time thresholds to avoid leakage from inference data.
        amount_p95 = float(thresholds.get("amount_p95", 200000.0))
        amount_p99 = float(thresholds.get("amount_p99", 500000.0))
        amount_mean = float(thresholds.get("amount_mean", 0.0))
        amount_std = float(thresholds.get("amount_std", 1.0))
    else:
        # Fallback: recompute from input (degenerate for single-row inference).
        amount_p95 = float(amount.quantile(0.95)) if len(amount) > 1 else 200000.0
        amount_p99 = float(amount.quantile(0.99)) if len(amount) > 1 else 500000.0
        amount_mean = float(amount.mean()) if len(amount) > 0 else 0.0
        amount_std = float(amount.std()) if len(amount) > 1 else 1.0

    out = df.copy()
    out["amount"] = amount
    out["tx_amount_log"] = np.log1p(amount)
    out["is_high_amount"] = (amount > 200000).astype(int)
    out["is_p95_amount"] = (amount > amount_p95).astype(int)
    out["is_p99_amount"] = (amount > amount_p99).astype(int)
    out["amount_zscore"] = (amount - amount_mean) / (amount_std + 1e-8)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["day_sin"] = np.sin(2.0 * np.pi * (day % 31) / 31.0)
    out["day_cos"] = np.cos(2.0 * np.pi * (day % 31) / 31.0)
    out["is_night"] = (((hour >= 0) & (hour < 6)) | (hour >= 22)).astype(int)
    out["is_weekend"] = ((day % 7).isin([5, 6])).astype(int)
    out["balance_change_abs"] = bal_change_abs.fillna(med)
    out["balance_ratio_clipped"] = bal_ratio.fillna(0.0).clip(-10, 10)
    out["balance_change_log"] = np.log1p(bal_change_abs.fillna(med))
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

    thresholds = compute_feature_thresholds(df)
    features_df = engineer_features_df(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    thresholds_path = Path(output_path).parent / "feature_thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    feature_names = list(features_df.columns)
    shape_rows = int(features_df.shape[0])
    null_counts = features_df.isnull().sum().to_dict()
    fraud_dist = features_df["is_fraud"].value_counts().to_dict()

    print(f"[Preprocess] Saved features to {output_path}")
    print(f"[Preprocess] Saved thresholds to {thresholds_path}: {thresholds}")
    print(f"[Preprocess] Shape: ({shape_rows}, {len(feature_names)})")
    print(f"[Preprocess] Feature names: {feature_names}")
    print(f"[Preprocess] Null counts: {null_counts}")
    print(f"[Preprocess] Fraud distribution: {fraud_dist}")


if __name__ == "__main__":
    main()
