"""Shadow deployment — fires the latest staging model async on every prod call.

Day 4 Phase 3 (2026-05-21). The Day-3 trigger module registers every
auto-retrained candidate as a new MLflow version. Most of the time the
``@production`` alias does not flip immediately — the gate is shadow AUPRC
on the LATEST labelled day, but in serving we don't have labels. So we need
a *live* shadow-vs-prod comparison: every prediction the prod model serves
is also scored by the latest candidate; the responses are written to the
``predictions`` table tagged with role='shadow'; downstream the comparator
in ``recent_predictions`` joins on ``request_id`` to compute disagreement
rate and probability delta.

Key invariants:

1. **Shadow never blocks the response.** The prod prediction returns first;
   the shadow call runs in a thread pool and is invisible to the user.
2. **Stale shadow handle.** The shadow model is loaded once from
   MLflow's ``@staging`` alias (falling back to the most recent version
   that isn't the prod alias). If staging is absent, shadow is disabled and
   the API still serves prod cleanly.
3. **Per-prediction agreement.** When prob_prod and prob_shadow are both
   logged with the same request_id, the comparator can report agreement
   without re-running either model.

This module is intentionally small — the heavy MLflow lookup logic lives
in ``src/registry/promote.py``; the storage logic lives in
``src/telemetry/logger.py``; this file is just the glue + the background
executor.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.config import PROJECT_ROOT
from src.telemetry import TelemetryLogger


DEFAULT_MODEL_NAME = "sentinel-fraud-xgboost"
DEFAULT_SHADOW_ALIAS = "staging"


class ShadowConfig(BaseModel):
    """Validated config for the shadow evaluator."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = DEFAULT_MODEL_NAME
    shadow_alias: str = DEFAULT_SHADOW_ALIAS
    tracking_uri: str | None = None
    max_workers: int = Field(default=2, ge=1, le=16)
    enabled: bool = True

    def resolved_tracking_uri(self) -> str:
        return self.tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
        )


@dataclass
class _LoadedShadow:
    """In-memory handle on the current shadow model."""

    model: Any
    version: str
    source: str  # 'mlflow:@staging' | 'mlflow:vN' | 'disabled'


class ShadowEvaluator:
    """Fire shadow predictions async; log to telemetry tagged with the request_id."""

    def __init__(
        self,
        config: ShadowConfig,
        telemetry: TelemetryLogger,
        *,
        feature_columns: list[str],
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.feature_columns = list(feature_columns)
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="shadow")
        self._lock = threading.Lock()
        self._handle: _LoadedShadow | None = None
        if config.enabled:
            try:
                self._refresh()
            except Exception as exc:
                print(f"[Shadow] init refresh failed (non-fatal): {exc}")
                self._handle = _LoadedShadow(model=None, version="", source="disabled")

    # ----- Lifecycle --------------------------------------------------------

    def _refresh(self) -> None:
        """Re-resolve the shadow handle from MLflow. Safe to call repeatedly."""
        mlflow.set_tracking_uri(self.config.resolved_tracking_uri())
        client = MlflowClient()

        version: str | None = None
        try:
            mv = client.get_model_version_by_alias(
                self.config.model_name, self.config.shadow_alias
            )
            version = mv.version
            source_tag = f"mlflow:@{self.config.shadow_alias}"
        except mlflow.exceptions.MlflowException:
            # Fall back to the latest version that isn't the prod alias.
            try:
                prod = client.get_model_version_by_alias(self.config.model_name, "production")
                prod_version = prod.version
            except mlflow.exceptions.MlflowException:
                prod_version = None
            versions = sorted(
                client.search_model_versions(f"name='{self.config.model_name}'"),
                key=lambda v: int(v.version),
                reverse=True,
            )
            for mv in versions:
                if prod_version is None or str(mv.version) != str(prod_version):
                    version = mv.version
                    source_tag = f"mlflow:v{version}"
                    break

        if version is None:
            with self._lock:
                self._handle = _LoadedShadow(model=None, version="", source="disabled")
            return

        # Load the model from the MLflow run artifact. The Day-1 train logs
        # both a sklearn-flavored xgboost model AND a joblib pickle; either
        # is acceptable but xgboost flavor preserves predict_proba.
        try:
            mv = client.get_model_version(self.config.model_name, version)
            model_uri = f"models:/{self.config.model_name}/{version}"
            try:
                import mlflow.xgboost as mxgb
                shadow_model = mxgb.load_model(model_uri)
            except Exception:
                shadow_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            print(f"[Shadow] load v{version} failed: {exc}; disabling shadow")
            with self._lock:
                self._handle = _LoadedShadow(model=None, version="", source="disabled")
            return

        with self._lock:
            self._handle = _LoadedShadow(model=shadow_model, version=str(version), source=source_tag)
        print(f"[Shadow] loaded v{version} from {source_tag}")

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._handle is None or self._handle.model is None:
                return {"enabled": False, "version": None, "source": "disabled"}
            return {
                "enabled": True,
                "version": self._handle.version,
                "source": self._handle.source,
            }

    # ----- Scoring ----------------------------------------------------------

    def fire(
        self,
        *,
        request_id: str,
        features: pd.DataFrame,
        log_features: dict[str, Any] | None = None,
    ) -> Future | None:
        """Submit a shadow prediction. Returns the Future or None when disabled."""
        with self._lock:
            handle = self._handle
        if handle is None or handle.model is None:
            return None

        def _run() -> dict[str, Any] | None:
            t0 = time.perf_counter()
            try:
                X = features[self.feature_columns]
                if hasattr(handle.model, "predict_proba"):
                    proba = float(handle.model.predict_proba(X)[:, 1][0])
                else:
                    out = handle.model.predict(X)
                    arr = np.asarray(out).reshape(-1)
                    proba = float(arr[0])
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.telemetry.log_prediction(
                    request_id=request_id,
                    model_name=self.config.model_name,
                    model_version=handle.version,
                    role="shadow",
                    probability=proba,
                    latency_ms=latency_ms,
                    features=log_features,
                    extra={"source": handle.source},
                )
                return {"probability": proba, "latency_ms": latency_ms, "version": handle.version}
            except Exception as exc:
                print(f"[Shadow] scoring failed (non-fatal): {exc}")
                return None

        return self._executor.submit(_run)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def load_shadow_from_disk_fallback(
    pkl_path: Path, feature_columns: list[str]
) -> Callable[[pd.DataFrame], float] | None:
    """Local pickle fallback used by tests that don't have an MLflow registry.

    Returns a callable that takes a 1-row feature frame and returns a single
    probability — or None if the pickle is missing.
    """
    if not pkl_path.exists():
        return None
    artifact = joblib.load(pkl_path)
    model = artifact["model"]

    def _score(features: pd.DataFrame) -> float:
        X = features[feature_columns]
        return float(model.predict_proba(X)[:, 1][0])

    return _score


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_SHADOW_ALIAS",
    "ShadowConfig",
    "ShadowEvaluator",
    "load_shadow_from_disk_fallback",
]
