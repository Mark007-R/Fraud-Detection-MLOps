"""Stage 1: Load Sparkov dataset via Amazon Fraud Dataset Benchmark (FDB)."""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd

from src.config import load_params


def _try_load_fdb_train_split(fdb_key: str = "sparkov") -> pd.DataFrame:
    """Load Sparkov train split from FDB using best-effort API compatibility.

    Parameters
    ----------
    fdb_key : str
        FDB dataset key.

    Returns
    -------
    pd.DataFrame
        Loaded train split dataframe.
    """
    try:
        from fraud_dataset_benchmark import FraudDatasetBenchmark  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FDB package is not installed. Install with: "
            "pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git"
        ) from exc

    benchmark = FraudDatasetBenchmark()

    candidate_calls: list[tuple[str, dict[str, Any]]] = [
        (
            "load_data",
            {
                "dataset_key": fdb_key,
                "split": "train",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
        (
            "load_data",
            {
                "key": fdb_key,
                "split": "train",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
        (
            "get_dataset",
            {
                "key": fdb_key,
                "split": "train",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
    ]

    last_error: Exception | None = None
    for method_name, kwargs in candidate_calls:
        method = getattr(benchmark, method_name, None)
        if method is None:
            continue
        try:
            result = method(**kwargs)
            if isinstance(result, pd.DataFrame):
                return result
            if isinstance(result, dict):
                for key in ("train", "data", "df"):
                    if key in result and isinstance(result[key], pd.DataFrame):
                        return result[key]
        except Exception as exc:  # pragma: no cover - API compatibility fallback
            last_error = exc
            continue

    raise RuntimeError(f"Could not load FDB data for key '{fdb_key}'. Last error: {last_error}")


def main() -> None:
    """Run Stage 1 pipeline step and persist Sparkov train split CSV."""
    params = load_params()
    fdb_cfg = params.get("fdb", {})

    output_path = pathlib.Path("data/raw/sparkov.csv")
    fallback_train_path = pathlib.Path("data/raw/sparkov_train.csv")
    fdb_key = str(fdb_cfg.get("dataset_key", "sparkov"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LoadFDB] Loading Sparkov train split from FDB key='{fdb_key}'...")
    try:
        sparkov_df = _try_load_fdb_train_split(fdb_key)
    except ImportError as exc:
        print(f"[LoadFDB] ImportError: {exc}")
        print("[LoadFDB] Falling back to local Sparkov train split if available.")
        if fallback_train_path.exists():
            sparkov_df = pd.read_csv(fallback_train_path)
            sparkov_df.to_csv(output_path, index=False)
            fraud_col = "is_fraud" if "is_fraud" in sparkov_df.columns else "EVENT_LABEL"
            ratio = float(sparkov_df[fraud_col].mean()) if fraud_col in sparkov_df.columns else float("nan")
            print(f"[LoadFDB] Loaded fallback train split from: {fallback_train_path}")
            print(f"[LoadFDB] Saved fallback output to: {output_path}")
            print(f"[LoadFDB] Shape: {sparkov_df.shape}")
            print(f"[LoadFDB] Columns: {list(sparkov_df.columns)}")
            print(f"[LoadFDB] Fraud ratio: {ratio:.6f}")
            return
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            fraud_col = "is_fraud" if "is_fraud" in existing_df.columns else "EVENT_LABEL"
            ratio = float(existing_df[fraud_col].mean()) if fraud_col in existing_df.columns else float("nan")
            print(f"[LoadFDB] Existing shape: {existing_df.shape}")
            print(f"[LoadFDB] Existing columns: {list(existing_df.columns)}")
            print(f"[LoadFDB] Existing fraud ratio: {ratio:.6f}")
            return
        raise

    sparkov_df.to_csv(output_path, index=False)

    fraud_col = "is_fraud" if "is_fraud" in sparkov_df.columns else "EVENT_LABEL"
    fraud_ratio = float(sparkov_df[fraud_col].mean()) if fraud_col in sparkov_df.columns else float("nan")

    print(f"[LoadFDB] Saved train split to: {output_path}")
    print(f"[LoadFDB] Shape: {sparkov_df.shape}")
    print(f"[LoadFDB] Columns: {list(sparkov_df.columns)}")
    print(f"[LoadFDB] Fraud ratio: {fraud_ratio:.6f}")


if __name__ == "__main__":
    main()
