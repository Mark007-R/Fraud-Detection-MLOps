"""Stage 4: Train XGBoost fraud classifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_params


def main() -> None:
    """Run Stage 4 training flow."""
    params = load_params()
    data_cfg = params.get("data", {})
    cfg = params.get("train", {})

    features_path = Path(data_cfg.get("features_path", "data/processed/features.csv"))
    model_path = Path(data_cfg.get("model_path", "models/fraud_model.pkl"))
    x_test_path = Path(data_cfg.get("x_test_path", "data/processed/X_test.csv"))
    y_test_path = Path(data_cfg.get("y_test_path", "data/processed/y_test.csv"))

    if not features_path.exists():
        raise FileNotFoundError(f"[Train] Missing features file: {features_path}")

    print(f"[Train] Loading features from {features_path}")
    pdf = pd.read_csv(features_path)

    if "is_fraud" not in pdf.columns:
        raise KeyError("[Train] Target column 'is_fraud' not found in features dataframe.")

    y = pd.to_numeric(pdf["is_fraud"], errors="coerce").fillna(0).astype(int)
    X = pdf.drop(columns=["is_fraud"])

    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    if bool(cfg.get("use_smote", False)):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "[Train] SMOTE requested via use_smote=true but imbalanced-learn is not installed. "
                "Install with `pip install imbalanced-learn`, or set use_smote=false in params.yaml."
            ) from exc

        print("[Train] Applying SMOTE to training split")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    early_stopping = int(cfg.get("early_stopping_rounds", 0))

    model = XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=int(cfg.get("max_depth", 6)),
        learning_rate=float(cfg.get("learning_rate", 0.1)),
        scale_pos_weight=float(cfg.get("scale_pos_weight", 50)),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=early_stopping if early_stopping > 0 else None,
    )

    print(f"[Train] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[Train] Test set: {X_test.shape[0]} samples")
    print(f"[Train] Fraud rate (train): {y_train.mean():.4f}")
    print(f"[Train] Fraud rate (test): {y_test.mean():.4f}")

    print("[Train] Fitting XGBoost model")
    fit_params = {}
    if early_stopping > 0:
        fit_params["eval_set"] = [(X_test, y_test)]
        fit_params["verbose"] = False
        print(f"[Train] Early stopping enabled: {early_stopping} rounds")

    model.fit(X_train, y_train, **fit_params)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load feature thresholds sidecar from preprocess stage so inference
    # reuses training-time percentiles/z-score stats (prevents leakage).
    thresholds: dict[str, float] = {}
    thresholds_path = features_path.parent / "feature_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        print(f"[Train] Loaded feature thresholds from {thresholds_path}")
    else:
        print(f"[Train] No thresholds sidecar at {thresholds_path}; artifact will omit them.")

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "target_column": "is_fraud",
        "feature_thresholds": thresholds,
    }
    joblib.dump(artifact, model_path)

    x_test_path.parent.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(x_test_path, index=False)
    pd.DataFrame({"is_fraud": y_test}).to_csv(y_test_path, index=False)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    print(f"[Train] Saved model to {model_path}")
    print(f"[Train] Saved X_test to {x_test_path}")
    print(f"[Train] Saved y_test to {y_test_path}")
    print(f"[Train] Accuracy: {acc:.6f}")
    print("[Train] Top 10 feature importances:")
    for feature, score in importances.items():
        print(f"  - {feature}: {float(score):.6f}")


if __name__ == "__main__":
    main()
