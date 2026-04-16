"""Stage 6: Benchmark SENTINEL model against FDB published baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_params
from src.preprocess import align_feature_columns, engineer_features_pandas


def _load_fdb_test_split(fdb_key: str) -> pd.DataFrame:
    """Load FDB standardized test split with API compatibility fallbacks.

    Parameters
    ----------
    fdb_key : str
        FDB dataset key.

    Returns
    -------
    pd.DataFrame
        Test split dataframe.
    """
    from fraud_dataset_benchmark import FraudDatasetBenchmark  # type: ignore

    benchmark = FraudDatasetBenchmark()
    candidate_calls: list[tuple[str, dict[str, Any]]] = [
        (
            "load_data",
            {
                "dataset_key": fdb_key,
                "split": "test",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
        (
            "load_data",
            {
                "key": fdb_key,
                "split": "test",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
        (
            "get_dataset",
            {
                "key": fdb_key,
                "split": "test",
                "load_pre_downloaded": False,
                "delete_downloaded": False,
            },
        ),
    ]

    for method_name, kwargs in candidate_calls:
        method = getattr(benchmark, method_name, None)
        if method is None:
            continue
        try:
            result = method(**kwargs)
            if isinstance(result, pd.DataFrame):
                return result
            if isinstance(result, dict):
                for key in ("test", "data", "df"):
                    if key in result and isinstance(result[key], pd.DataFrame):
                        return result[key]
        except Exception:
            continue

    raise RuntimeError("Could not load FDB test split with available API methods.")


def _sparkov_to_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map Sparkov/FDB dataframe to Stage-2 common schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input Sparkov-style dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized schema dataframe.
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
        raise KeyError("[Benchmark] Could not detect fraud label column in FDB split.")

    tx_type_col = "category" if "category" in cols else "transaction_type"

    if "trans_date_trans_time" in cols:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    elif "TX_TIMESTAMP" in cols:
        dt = pd.to_datetime(df["TX_TIMESTAMP"], errors="coerce")
    else:
        dt = pd.Series(pd.Timestamp("1970-01-01"), index=df.index)

    out = pd.DataFrame(
        {
            "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0),
            "is_fraud": pd.to_numeric(df[fraud_col], errors="coerce").fillna(0).astype(int),
            "transaction_type": df[tx_type_col].astype(str),
            "hour_of_day": dt.dt.hour.fillna(0).astype(int),
            "day_of_month": dt.dt.day.fillna(1).astype(int),
            "balance_change_orig": 0.0,
            "balance_ratio": 0.0,
            "has_balance_info": 0,
            "source": "sparkov",
        }
    )
    return out


def main() -> None:
    """Run Stage 6 benchmark flow and save comparison JSON."""
    params = load_params()
    fdb_cfg = params.get("fdb", {})
    cfg = params.get("benchmark", {})

    model_path = Path(cfg.get("model_path", "models/fraud_model.pkl"))
    fdb_key = str(fdb_cfg.get("dataset_key", "sparkov"))
    output_path = Path(cfg.get("output_path", "metrics/fdb_benchmark.json"))

    if not model_path.exists():
        raise FileNotFoundError(f"[Benchmark] Missing model file: {model_path}")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    fdb_available = True
    try:
        print(f"[Benchmark] Loading FDB test split for key={fdb_key}")
        fdb_test = _load_fdb_test_split(fdb_key)
    except ImportError:
        fdb_available = False
        print(
            "[Benchmark] FDB package not installed. Install with: "
            "pip install git+https://github.com/amazon-science/fraud-dataset-benchmark.git"
        )
        sparkov_local = Path("data/raw/sparkov.csv")
        if not sparkov_local.exists():
            raise FileNotFoundError("[Benchmark] No FDB package and no local sparkov.csv fallback.")
        fdb_test = pd.read_csv(sparkov_local)
    except Exception as exc:
        fdb_available = False
        print(f"[Benchmark] FDB standardized load failed: {exc}. Falling back to local sparkov.csv")
        sparkov_local = Path("data/raw/sparkov.csv")
        if not sparkov_local.exists():
            raise FileNotFoundError("[Benchmark] No FDB split and no local sparkov.csv fallback.")
        fdb_test = pd.read_csv(sparkov_local)

    common = _sparkov_to_common_schema(fdb_test)
    y_true = common["is_fraud"].astype(int)

    engineered = engineer_features_pandas(common)
    if "is_fraud" in engineered.columns:
        engineered = engineered.drop(columns=["is_fraud"])
    X = align_feature_columns(engineered, feature_columns)

    y_prob = model.predict_proba(X)[:, 1]
    auc_roc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else 0.0

    baselines_cfg = cfg.get("baselines")
    if baselines_cfg:
        baselines = {str(name): float(score) for name, score in baselines_cfg.items()}
    else:
        # Fallback defaults if params.yaml omits benchmark.baselines.
        baselines = {
            "AutoGluon": 0.952,
            "H2O AutoML": 0.947,
            "AutoSklearn": 0.931,
        }

    comparisons = {}
    for name, score in baselines.items():
        comparisons[name] = {
            "baseline_auc": score,
            "our_auc": auc_roc,
            "status": "BEAT" if auc_roc >= score else "BELOW",
            "delta": round(auc_roc - score, 6),
        }

    output = {
        "dataset_key": fdb_key,
        "fdb_standardized": fdb_available,
        "our_model_auc": auc_roc,
        "baselines": comparisons,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("[Benchmark] Baseline comparison:")
    for name, info in comparisons.items():
        print(
            f"  - {name}: our_auc={info['our_auc']:.6f} | baseline={info['baseline_auc']:.3f} "
            f"| status={info['status']} | delta={info['delta']:+.6f}"
        )
    print(f"[Benchmark] Results saved to {output_path}")


if __name__ == "__main__":
    main()
