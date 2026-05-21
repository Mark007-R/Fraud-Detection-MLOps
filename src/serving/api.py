"""FastAPI serving layer — /predict + /metrics/* read endpoints.

Day 4 Phase 3 (2026-05-21). Brings together the Day-1 trained model, the
Day-2 registry, the Day-3 drift detector, and the new Day-4 telemetry
store into a single async service.

Endpoints:

* ``POST /predict`` — score one transaction. Body is a JSON dict matching
  the trained model's feature columns. The prod prediction is computed
  synchronously and returned; a shadow prediction is fired async if the
  shadow evaluator has a version loaded. Both rows land in the
  ``predictions`` telemetry table with the same ``request_id`` so the
  Day-7 dashboard can join on it.
* ``GET /healthz`` — readiness probe. Reports prod model version and shadow
  status.
* ``GET /metrics/predictions?hours=N`` — aggregate prediction stats by role
  over the last N hours.
* ``GET /metrics/drift?limit=N`` — last N drift-detector rows.
* ``GET /metrics/registry?limit=N`` — last N registry alias-flip rows.
* ``GET /metrics/retrain_events?limit=N`` — last N auto-retrain rows.
* ``GET /metrics/shadow_agreement?hours=N`` — joins prod + shadow on
  request_id and reports disagreement rate + mean |proba_delta|.

All input/output models use Pydantic v2. The app is created via
``create_app(...)`` so tests can pass injected configs / loaders.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.config import PROJECT_ROOT
from src.data.loader import LoaderConfig, SentinelDataLoader
from src.serving.shadow import ShadowConfig, ShadowEvaluator
from src.telemetry import TelemetryConfig, TelemetryLogger


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class APIConfig(BaseModel):
    """Validated startup config for the FastAPI service."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: Path | None = None  # falls back to the data loader's resolved path
    model_name: str = "sentinel-fraud-xgboost"
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_shadow: bool = True
    store_features: bool = False  # opt-in to PII-sensitive feature persistence
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            decision_threshold=float(os.environ.get("SENTINEL_DECISION_THRESHOLD", 0.5)),
            enable_shadow=os.environ.get("SENTINEL_DISABLE_SHADOW", "").lower() not in {"1", "true", "yes"},
            store_features=os.environ.get("SENTINEL_STORE_FEATURES", "").lower() in {"1", "true", "yes"},
        )


class PredictRequest(BaseModel):
    """A single transaction's engineered features.

    The shape is intentionally permissive — Day-1's pipeline produces 45
    columns and any drift-corrected retrain may rename or add features. The
    server reindexes against ``feature_columns`` on the loaded artifact, so
    missing features become NaN (XGBoost handles missing natively) and
    extras are dropped.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    request_id: str | None = None
    features: dict[str, float]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_version: str
    probability: float
    label: int
    threshold: float
    latency_ms: float
    shadow: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    prod_version: str
    prod_loaded_at: str
    shadow: dict[str, Any]
    telemetry_url: str


class PredictionStatsResponse(BaseModel):
    window_hours: float
    by_role: dict[str, dict[str, Any]]


class ShadowAgreementResponse(BaseModel):
    window_hours: float
    paired_predictions: int
    label_disagreements: int
    label_disagreement_rate: float
    mean_abs_proba_delta: float
    mean_signed_proba_delta: float


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Loaded model + sidecar metadata."""

    model: Any
    feature_columns: list[str]
    version: str  # local pkl gets 'local-pkl'; registry-resolved gets its registry version
    source: str
    loaded_at: str


def load_bundle_from_disk(pkl_path: Path, version_label: str = "local-pkl") -> ModelBundle:
    """Load the Day-1 ``models/fraud_model.pkl`` artifact."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"[API] Model artifact missing: {pkl_path}")
    artifact = joblib.load(pkl_path)
    if "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(f"[API] Unexpected artifact shape at {pkl_path}: keys={list(artifact)}")
    return ModelBundle(
        model=artifact["model"],
        feature_columns=list(artifact["feature_columns"]),
        version=version_label,
        source=str(pkl_path),
        loaded_at=pd.Timestamp.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    config: APIConfig | None = None,
    *,
    bundle: ModelBundle | None = None,
    telemetry: TelemetryLogger | None = None,
    shadow_config: ShadowConfig | None = None,
) -> FastAPI:
    """Build a FastAPI app. All collaborators can be injected for tests."""
    cfg = config or APIConfig.from_env()
    if bundle is None:
        loader_cfg = LoaderConfig.from_params()
        loader = SentinelDataLoader(loader_cfg)
        model_path = cfg.model_path or loader.model_path()
        bundle = load_bundle_from_disk(model_path)
    telemetry = telemetry or TelemetryLogger(TelemetryConfig.from_env())

    shadow_cfg = shadow_config or ShadowConfig(
        model_name=cfg.model_name, enabled=cfg.enable_shadow
    )
    shadow_eval: ShadowEvaluator | None = None
    if cfg.enable_shadow:
        try:
            shadow_eval = ShadowEvaluator(
                shadow_cfg, telemetry, feature_columns=bundle.feature_columns
            )
        except Exception as exc:
            print(f"[API] Shadow init failed (non-fatal): {exc}")

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        yield
        if shadow_eval is not None:
            shadow_eval.shutdown(wait=False)

    app = FastAPI(
        title="Sentinel Fraud Detection",
        version="day04-phase3",
        description="Serving layer with prod prediction + async shadow + telemetry",
        lifespan=_lifespan,
    )

    # ----- /healthz ----------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            prod_version=bundle.version,
            prod_loaded_at=bundle.loaded_at,
            shadow=shadow_eval.status() if shadow_eval else {"enabled": False, "version": None, "source": "disabled"},
            telemetry_url=str(telemetry.engine.url),
        )

    # ----- POST /predict -----------------------------------------------------

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not req.features:
            raise HTTPException(status_code=422, detail="`features` must be a non-empty dict")
        request_id = req.request_id or uuid.uuid4().hex
        features = pd.DataFrame([req.features]).reindex(columns=bundle.feature_columns)
        t0 = time.perf_counter()
        proba = float(bundle.model.predict_proba(features)[:, 1][0])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        label = int(proba >= cfg.decision_threshold)

        stored_features = req.features if cfg.store_features else None
        telemetry.log_prediction(
            request_id=request_id,
            model_name=cfg.model_name,
            model_version=bundle.version,
            role="production",
            probability=proba,
            threshold=cfg.decision_threshold,
            latency_ms=latency_ms,
            features=stored_features,
        )

        shadow_status = None
        if shadow_eval is not None:
            fut = shadow_eval.fire(
                request_id=request_id,
                features=features,
                log_features=stored_features,
            )
            if fut is not None:
                shadow_status = {"fired": True, "version": shadow_eval.status().get("version")}
            else:
                shadow_status = {"fired": False, "reason": "shadow_disabled"}

        return PredictResponse(
            request_id=request_id,
            model_version=bundle.version,
            probability=proba,
            label=label,
            threshold=cfg.decision_threshold,
            latency_ms=latency_ms,
            shadow=shadow_status,
        )

    # ----- /metrics/* -------------------------------------------------------

    @app.get("/metrics/predictions", response_model=PredictionStatsResponse)
    def metrics_predictions(hours: float = Query(24.0, gt=0.0, le=720.0)) -> PredictionStatsResponse:
        summary = telemetry.prediction_summary(hours=hours)
        return PredictionStatsResponse(**summary)

    @app.get("/metrics/drift")
    def metrics_drift(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return {"rows": telemetry.recent_drift(limit=limit)}

    @app.get("/metrics/registry")
    def metrics_registry(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return {"rows": telemetry.recent_registry_log(limit=limit)}

    @app.get("/metrics/retrain_events")
    def metrics_retrain_events(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        return {"rows": telemetry.recent_retrain_events(limit=limit)}

    @app.get("/metrics/shadow_agreement", response_model=ShadowAgreementResponse)
    def metrics_shadow_agreement(
        hours: float = Query(24.0, gt=0.0, le=720.0),
    ) -> ShadowAgreementResponse:
        rows = telemetry.recent_predictions(limit=10_000)
        cutoff_hours = hours
        now = pd.Timestamp.utcnow().tz_localize(None)
        groups: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rows:
            ts = pd.Timestamp(r["created_at"])
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            if (now - ts).total_seconds() > cutoff_hours * 3600.0:
                continue
            groups[r["request_id"]][r["role"]] = float(r["probability"])
            groups[r["request_id"]][f"{r['role']}_label"] = int(r["label"])

        paired = [g for g in groups.values() if "production" in g and "shadow" in g]
        n = len(paired)
        if n == 0:
            return ShadowAgreementResponse(
                window_hours=hours,
                paired_predictions=0,
                label_disagreements=0,
                label_disagreement_rate=0.0,
                mean_abs_proba_delta=0.0,
                mean_signed_proba_delta=0.0,
            )
        label_disagree = sum(int(g["production_label"] != g["shadow_label"]) for g in paired)
        abs_delta = float(np.mean([abs(g["production"] - g["shadow"]) for g in paired]))
        signed_delta = float(np.mean([g["shadow"] - g["production"] for g in paired]))
        return ShadowAgreementResponse(
            window_hours=hours,
            paired_predictions=n,
            label_disagreements=label_disagree,
            label_disagreement_rate=label_disagree / n,
            mean_abs_proba_delta=abs_delta,
            mean_signed_proba_delta=signed_delta,
        )

    # Stash collaborators for tests / shutdown.
    app.state.bundle = bundle
    app.state.telemetry = telemetry
    app.state.shadow = shadow_eval
    app.state.config = cfg
    return app


# ---------------------------------------------------------------------------
# Module-level app for `uvicorn src.serving.api:app`.
# ---------------------------------------------------------------------------


def _build_default_app() -> FastAPI:
    return create_app()


app = _build_default_app() if os.environ.get("SENTINEL_BUILD_APP_AT_IMPORT", "1") == "1" else None


__all__ = [
    "APIConfig",
    "ModelBundle",
    "PredictRequest",
    "PredictResponse",
    "create_app",
    "load_bundle_from_disk",
]
