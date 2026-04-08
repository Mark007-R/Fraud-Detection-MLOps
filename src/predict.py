"""Inference utility for single or batch fraud prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import align_feature_columns, engineer_features_pandas


def _format_predictions(df: pd.DataFrame) -> None:
    """Print prediction output in a readable format.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction dataframe.
    """
    print("[Predict] Prediction results:")
    for idx, row in df.iterrows():
        print(
            f"  - row={idx} | fraud_probability={row['fraud_probability']:.6f} "
            f"| fraud_prediction={int(row['fraud_prediction'])}"
        )


def predict_dataframe(input_df: pd.DataFrame, model_path: str = "models/fraud_model.pkl") -> pd.DataFrame:
    """Run model inference on a dataframe.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input transaction dataframe.
    model_path : str
        Saved model path.

    Returns
    -------
    pd.DataFrame
        Fraud probability and class predictions.
    """
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    engineered = engineer_features_pandas(input_df.copy())
    if "is_fraud" in engineered.columns:
        engineered = engineered.drop(columns=["is_fraud"])
    X = align_feature_columns(engineered, feature_columns)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return pd.DataFrame({"fraud_probability": prob, "fraud_prediction": pred})


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="SENTINEL fraud prediction utility.")
    parser.add_argument("--input-csv", type=str, default="", help="Path to input CSV file")
    parser.add_argument("--input-json", type=str, default="", help="JSON object or list of objects")
    parser.add_argument("--model-path", type=str, default="models/fraud_model.pkl", help="Model path")
    parser.add_argument("--output-csv", type=str, default="", help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    """Run prediction CLI flow with demo fallback."""
    args = parse_args()

    if args.input_csv:
        if not Path(args.input_csv).exists():
            raise FileNotFoundError(f"[Predict] Input CSV not found: {args.input_csv}")
        input_df = pd.read_csv(args.input_csv)
        print(f"[Predict] Loaded batch input from {args.input_csv} ({len(input_df)} rows)")
    elif args.input_json:
        parsed: Any = json.loads(args.input_json)
        if isinstance(parsed, dict):
            input_df = pd.DataFrame([parsed])
            print("[Predict] Loaded single transaction from JSON object")
        elif isinstance(parsed, list):
            input_df = pd.DataFrame(parsed)
            print(f"[Predict] Loaded batch input from JSON list ({len(input_df)} rows)")
        else:
            raise ValueError("[Predict] --input-json must be a JSON object or list of objects")
    else:
        print("[Predict] No input provided. Running built-in demo transactions.")
        input_df = pd.DataFrame(
            [
                {
                    "amount": 250000,
                    "transaction_type": "TRANSFER",
                    "hour_of_day": 2,
                    "day_of_month": 14,
                    "balance_change_orig": 240000,
                    "balance_ratio": 0.05,
                    "has_balance_info": 1,
                    "source": "paysim",
                },
                {
                    "amount": 120,
                    "transaction_type": "PAYMENT",
                    "hour_of_day": 13,
                    "day_of_month": 9,
                    "balance_change_orig": 120,
                    "balance_ratio": 0.98,
                    "has_balance_info": 1,
                    "source": "sparkov",
                },
            ]
        )

    pred_df = predict_dataframe(input_df=input_df, model_path=args.model_path)
    _format_predictions(pred_df)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_csv, index=False)
        print(f"[Predict] Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
