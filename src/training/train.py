"""Training module — Day 4 layout wrapper around the canonical src/train.py.

Day 4 Phase 3 (2026-05-21). The Day-1 training script lives at
``src/train.py`` and is wired into ``dvc.yaml`` by that path. To satisfy the
Day-4 module layout (``src/training/train.py``) without breaking the DVC
graph or the Day-1 temporal-split fix, this wrapper:

1. Re-exports ``temporal_split_per_source`` and ``main`` from ``src/train.py``
   so callers can ``from src.training.train import main``;
2. Adds a Pydantic ``TrainConfig`` (validates the ``train:`` block in
   params.yaml at startup);
3. Exposes a thin ``train_xgboost(X, y, cfg)`` function callable from the
   Day 5 Optuna sweep without going through the DVC stage.

Tomorrow's Optuna sweep imports ``train_xgboost``; today the inference path
and the auto-retrain trigger continue to call into ``src.train`` /
``src.drift.trigger`` as before.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import load_params
from src.train import main as _legacy_main
from src.train import temporal_split_per_source


class TrainConfig(BaseModel):
    """Validated view of the ``train:`` block in params.yaml."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str = "xgboost"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    n_estimators: int = Field(default=200, ge=1, le=5000)
    max_depth: int = Field(default=6, ge=1, le=32)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    scale_pos_weight: float = Field(default=50.0, gt=0.0)
    use_smote: bool = False
    early_stopping_rounds: int = Field(default=0, ge=0)

    @field_validator("model_type")
    @classmethod
    def _model_known(cls, v: str) -> str:
        if v.lower() != "xgboost":
            raise ValueError(f"only model_type='xgboost' is supported, got '{v}'")
        return v

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> "TrainConfig":
        params = params if params is not None else load_params()
        return cls(**{k: v for k, v in params.get("train", {}).items() if k in cls.model_fields})


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: TrainConfig,
    *,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
    mlflow_run_name: str | None = None,
) -> tuple[XGBClassifier, dict[str, float]]:
    """Fit XGBoost with the given config; return (model, eval_metrics).

    Used by the Day 5 Optuna sweep. Wraps the run in an MLflow run when a
    tracking URI is already set; otherwise it just trains.
    """
    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        scale_pos_weight=cfg.scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=-1,
        early_stopping_rounds=cfg.early_stopping_rounds if cfg.early_stopping_rounds > 0 else None,
    )

    fit_params: dict[str, Any] = {}
    if cfg.early_stopping_rounds > 0 and X_eval is not None and y_eval is not None:
        fit_params["eval_set"] = [(X_eval, y_eval)]
        fit_params["verbose"] = False

    active = mlflow.active_run()
    if active is None and mlflow_run_name is not None:
        ctx = mlflow.start_run(run_name=mlflow_run_name)
    else:
        ctx = _Nullctx()
    with ctx:
        if active is not None or mlflow_run_name is not None:
            mlflow.log_params(cfg.model_dump())
        model.fit(X_train, y_train, **fit_params)
        metrics: dict[str, float] = {}
        if X_eval is not None and y_eval is not None and y_eval.nunique() > 1:
            proba = model.predict_proba(X_eval)[:, 1]
            metrics["eval_auc"] = float(roc_auc_score(y_eval, proba))
            metrics["eval_auprc"] = float(average_precision_score(y_eval, proba))
            if active is not None or mlflow_run_name is not None:
                mlflow.log_metrics(metrics)

    return model, metrics


class _Nullctx:
    """No-op context manager so ``with`` can be unconditional above."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> None:
    """Run the canonical training pipeline (DVC stage 4)."""
    _legacy_main()


__all__ = [
    "TrainConfig",
    "main",
    "temporal_split_per_source",
    "train_xgboost",
]
