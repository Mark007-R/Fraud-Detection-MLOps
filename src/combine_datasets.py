"""Stage 2: Align PaySim + Sparkov schemas and combine datasets."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_params


def _normalize_sparkov(df: pd.DataFrame) -> pd.DataFrame:
    """Map Sparkov (Kaggle/FDB variants) into common schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Sparkov dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized Sparkov dataframe.
    """
    cols = set(df.columns)

    amount_col = "amt" if "amt" in cols else "amount"

    if "is_fraud" in cols:
        fraud_col = "is_fraud"
    elif "EVENT_LABEL" in cols:
        fraud_col = "EVENT_LABEL"
    elif "isFraud" in cols:
        fraud_col = "isFraud"
    else:
        raise KeyError("[Combine] Could not detect Sparkov fraud label column.")

    tx_type_col = "category" if "category" in cols else "transaction_type"

    if "trans_date_trans_time" in cols:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        hour = dt.dt.hour.fillna(0).astype("int64")
        day = dt.dt.day.fillna(1).astype("int64")
        ts = (dt.astype("int64") // 10**9).fillna(0).astype("int64")
    elif "TX_TIMESTAMP" in cols:
        dt = pd.to_datetime(df["TX_TIMESTAMP"], errors="coerce")
        hour = dt.dt.hour.fillna(0).astype("int64")
        day = dt.dt.day.fillna(1).astype("int64")
        ts = (dt.astype("int64") // 10**9).fillna(0).astype("int64")
    elif "unix_time" in cols:
        hour = (pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) * 0).astype("int64")
        day = ((pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) * 0) + 1).astype("int64")
        ts = pd.to_numeric(df["unix_time"], errors="coerce").fillna(0).astype("int64")
    else:
        hour = (pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) * 0).astype("int64")
        day = ((pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) * 0) + 1).astype("int64")
        ts = (pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) * 0).astype("int64")

    out = df.assign(
        amount=pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0),
        is_fraud=pd.to_numeric(df[fraud_col], errors="coerce").fillna(0).astype("int64"),
        transaction_type=df[tx_type_col].astype("string"),
        hour_of_day=hour,
        day_of_month=day,
        balance_change_orig=float("nan"),
        balance_ratio=float("nan"),
        has_balance_info=0,
        source="sparkov",
        txn_timestamp=ts,
    )

    return out[
        [
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
        ]
    ]


def _normalize_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """Map PaySim into common schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized PaySim dataframe.
    """
    required = {"amount", "isFraud", "type", "step", "oldbalanceOrg", "newbalanceOrig"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[Combine] Missing PaySim columns: {sorted(missing)}")

    step_numeric = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype("int64")
    old_bal = pd.to_numeric(df["oldbalanceOrg"], errors="coerce").fillna(0.0)
    new_bal = pd.to_numeric(df["newbalanceOrig"], errors="coerce").fillna(0.0)

    out = df.assign(
        amount=pd.to_numeric(df["amount"], errors="coerce").fillna(0.0),
        is_fraud=pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype("int64"),
        transaction_type=df["type"].astype("string"),
        hour_of_day=(step_numeric % 24).astype("int64"),
        day_of_month=(step_numeric // 24).astype("int64"),
        balance_change_orig=(old_bal - new_bal),
        balance_ratio=(new_bal / (old_bal + 1.0)),
        has_balance_info=1,
        source="paysim",
        txn_timestamp=(step_numeric * 3600).astype("int64"),
    )

    return out[["amount", "is_fraud", "transaction_type", "hour_of_day", "day_of_month", "balance_change_orig", "balance_ratio", "has_balance_info", "source", "txn_timestamp"]]


def _validate_combined(df: pd.DataFrame) -> None:
    """Validate the combined dataframe schema and data integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataframe to validate.

    Raises
    ------
    ValueError
        If validation checks fail.
    """
    required = {"amount", "is_fraud", "transaction_type", "hour_of_day",
                "day_of_month", "source", "txn_timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Combine] Missing columns after merge: {sorted(missing)}")

    if df["amount"].isnull().all():
        raise ValueError("[Combine] All amounts are null after merge")

    fraud_values = set(df["is_fraud"].dropna().unique())
    if not fraud_values.issubset({0, 1}):
        raise ValueError(f"[Combine] Unexpected fraud label values: {fraud_values}")

    null_pct = df.isnull().mean()
    high_null_cols = null_pct[null_pct > 0.5].index.tolist()
    if high_null_cols:
        print(f"[Combine] WARNING: High null rate (>50%) in columns: {high_null_cols}")


def main() -> None:
    """Run Stage 2 to combine PaySim and Sparkov."""
    params = load_params()
    combine_cfg = params.get("combine", {})

    paysim_path = combine_cfg.get("paysim_path", "data/raw/paysim.csv")
    sparkov_path = combine_cfg.get("sparkov_path", "data/raw/sparkov.csv")
    output_path = combine_cfg.get("output_path", "data/processed/combined_transactions.csv")

    print(f"[Combine] Loading PaySim from {paysim_path}")
    if not Path(paysim_path).exists():
        raise FileNotFoundError(f"[Combine] PaySim file not found: {paysim_path}")

    print(f"[Combine] Loading Sparkov from {sparkov_path}")
    if not Path(sparkov_path).exists():
        raise FileNotFoundError(f"[Combine] Sparkov file not found: {sparkov_path}")

    paysim_df = pd.read_csv(paysim_path)
    sparkov_df = pd.read_csv(sparkov_path)

    print(f"[Combine] PaySim shape: {paysim_df.shape}, columns: {list(paysim_df.columns)}")
    print(f"[Combine] Sparkov shape: {sparkov_df.shape}, columns: {list(sparkov_df.columns)}")

    paysim_std = _normalize_paysim(paysim_df)
    sparkov_std = _normalize_sparkov(sparkov_df)

    print(f"[Combine] PaySim standardized: {paysim_std.shape}")
    print(f"[Combine] Sparkov standardized: {sparkov_std.shape}")

    combined = pd.concat([paysim_std, sparkov_std], axis=0, ignore_index=True)
    # Sort by (source, txn_timestamp) to preserve per-source chronology.
    # Day 1 fix: previously this was a random shuffle, which destroyed time
    # ordering and made temporal splits impossible downstream in train.py.
    combined = combined.sort_values(["source", "txn_timestamp"], kind="mergesort").reset_index(drop=True)

    _validate_combined(combined)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    total_rows = int(combined.shape[0])
    fraud_ratio = float(combined["is_fraud"].mean())
    source_counts = combined["source"].value_counts().to_dict()
    type_counts = combined["transaction_type"].value_counts().to_dict()

    print(f"[Combine] Saved combined file to {output_path}")
    print(f"[Combine] Total rows: {total_rows}")
    print(f"[Combine] Fraud ratio: {fraud_ratio:.6f}")
    print(f"[Combine] Source counts: {source_counts}")
    print(f"[Combine] Transaction types: {type_counts}")


if __name__ == "__main__":
    main()
