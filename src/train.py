"""Stage 4: Train XGBoost fraud classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_params(path: str = "params.yaml") -> dict:
    """Load YAML params.

    Parameters
    ----------
    path : str
        Params file path.

    Returns
    -------
    dict
        Parsed params.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

            print("[Train] Applying SMOTE to training split")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except ImportError:
            print("[Train] SMOTE requested but imbalanced-learn is not available. Continuing without SMOTE.")

    model = XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=int(cfg.get("max_depth", 6)),
        learning_rate=float(cfg.get("learning_rate", 0.1)),
        scale_pos_weight=float(cfg.get("scale_pos_weight", 50)),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    print("[Train] Fitting XGBoost model")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "target_column": "is_fraud",
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
