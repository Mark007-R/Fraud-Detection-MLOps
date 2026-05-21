"""Telemetry backing store — predictions, drift, retrain, registry audit.

Day 4 Phase 3 (2026-05-21). The auto-retrain trigger from Day 3 emitted
RetrainEvent dataclasses to CSV; the drift detector wrote a DriftReport to
JSON; the registry CLI logged to stdout. Day 4 unifies all four signals
behind a single SQLAlchemy-backed telemetry store so the FastAPI serving
layer can answer "what's the live drift?" / "what was promoted last week?"
without scraping log files.

Schema (4 tables):

* ``predictions`` — every served prediction with model_version, latency,
  probability, request_id, plus an opt-in copy of the input features for
  shadow-eval replay. Index on (created_at, model_version).
* ``drift_scores`` — one row per scored window, per-feature KS stats plus
  overall PSI and the fire flag. Index on (window_label, created_at).
* ``retrain_events`` — one row per trigger-fired retrain attempt, capturing
  shadow_auprc/prod_auprc + promote decision + end-to-end latency.
* ``model_registry_log`` — one row per promote / rollback flip, capturing
  the from/to version and the per-step latency.

Postgres is the production backing store (docker-compose.yml ships one).
When DATABASE_URL points at sqlite or is omitted, the logger silently
degrades to a local sqlite file under ``data/telemetry.sqlite`` so the test
suite and the Day-7 demo run end-to-end without Docker.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class PredictionRow(_Base):
    """One row per served prediction (prod + shadow share this table)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    created_at = Column(DateTime, default=_dt.datetime.utcnow, nullable=False, index=True)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    role = Column(String(16), nullable=False)  # 'production' | 'shadow'
    probability = Column(Float, nullable=False)
    label = Column(Integer, nullable=False)  # threshold-classified prediction
    threshold = Column(Float, nullable=False, default=0.5)
    latency_ms = Column(Float, nullable=False)
    features = Column(JSON, nullable=True)  # opt-in stored input
    extra = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_predictions_created_model", "created_at", "model_version"),
    )


class DriftScoreRow(_Base):
    """One row per scored monitoring window."""

    __tablename__ = "drift_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    window_label = Column(String(64), index=True, nullable=False)
    created_at = Column(DateTime, default=_dt.datetime.utcnow, nullable=False, index=True)
    n_rows = Column(Integer, nullable=False)
    proba_psi = Column(Float, nullable=False)
    flagged_features = Column(JSON, nullable=False)  # list[str]
    feature_ks_stats = Column(JSON, nullable=False)  # dict[str, float]
    drift_fired = Column(Boolean, nullable=False)
    fire_reason = Column(String(256), nullable=False, default="")

    __table_args__ = (
        Index("ix_drift_window_created", "window_label", "created_at"),
    )


class RetrainEventRow(_Base):
    """One row per auto-retrain attempt."""

    __tablename__ = "retrain_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_dt.datetime.utcnow, nullable=False, index=True)
    triggered_on_day = Column(Integer, nullable=False)
    consecutive_days = Column(Integer, nullable=False)
    train_rows = Column(Integer, nullable=False)
    shadow_rows = Column(Integer, nullable=False)
    shadow_auprc = Column(Float, nullable=False)
    shadow_auc = Column(Float, nullable=False)
    prod_auprc = Column(Float, nullable=False)
    promote_decision = Column(Boolean, nullable=False)
    promote_reason = Column(String(256), nullable=False, default="")
    new_model_version = Column(String(32), nullable=True)
    run_id = Column(String(64), nullable=True)
    seconds_end_to_end = Column(Float, nullable=False, default=0.0)


class ModelRegistryLogRow(_Base):
    """One row per promote / rollback alias flip."""

    __tablename__ = "model_registry_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_dt.datetime.utcnow, nullable=False, index=True)
    action = Column(String(32), nullable=False)  # 'promote' | 'rollback'
    model_name = Column(String(128), nullable=False)
    alias = Column(String(64), nullable=False)
    from_version = Column(String(32), nullable=True)
    to_version = Column(String(32), nullable=False)
    flip_seconds = Column(Float, nullable=False)
    actor = Column(String(64), nullable=False, default="auto")
    reason = Column(String(256), nullable=False, default="")


# ---------------------------------------------------------------------------
# Config + logger
# ---------------------------------------------------------------------------


class TelemetryConfig(BaseModel):
    """Validated connection config. Resolves DATABASE_URL from the environment."""

    model_config = ConfigDict(protected_namespaces=())

    database_url: str | None = None
    sqlite_fallback_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "telemetry.sqlite"
    )
    echo_sql: bool = False

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        return cls(
            database_url=os.environ.get("SENTINEL_DATABASE_URL"),
            echo_sql=os.environ.get("SENTINEL_TELEMETRY_ECHO", "").lower() in {"1", "true", "yes"},
        )

    def resolve_url(self) -> str:
        """Return a usable SQLAlchemy URL, falling back to a local sqlite file."""
        if self.database_url:
            return self.database_url
        self.sqlite_fallback_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.sqlite_fallback_path.as_posix()}"


class TelemetryLogger:
    """Thread-safe wrapper over a SQLAlchemy session factory."""

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig.from_env()
        url = self.config.resolve_url()
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            # sqlite default disallows cross-thread sharing; the shadow path uses
            # a thread pool, so we relax the check and serialise writes ourselves.
            connect_args = {"check_same_thread": False}
        self.engine = create_engine(url, echo=self.config.echo_sql, connect_args=connect_args, future=True)
        _Base.metadata.create_all(self.engine)
        self._SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self._write_lock = threading.Lock()  # only used on sqlite to serialise INSERTs

    # ----- Session helpers --------------------------------------------------

    @contextlib.contextmanager
    def session(self) -> Iterable[Session]:
        s = self._SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ----- Writes -----------------------------------------------------------

    def log_prediction(
        self,
        *,
        request_id: str | None = None,
        model_name: str,
        model_version: str,
        role: str,
        probability: float,
        threshold: float = 0.5,
        latency_ms: float,
        features: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        if role not in {"production", "shadow"}:
            raise ValueError(f"role must be 'production' or 'shadow', got '{role}'")
        row = PredictionRow(
            request_id=request_id or uuid.uuid4().hex,
            model_name=model_name,
            model_version=model_version,
            role=role,
            probability=float(probability),
            label=int(probability >= threshold),
            threshold=float(threshold),
            latency_ms=float(latency_ms),
            features=features,
            extra=extra,
        )
        with self._write_lock, self.session() as s:
            s.add(row)
            s.flush()
            return int(row.id)

    def log_drift(
        self,
        *,
        window_label: str,
        n_rows: int,
        proba_psi: float,
        flagged_features: list[str],
        feature_ks_stats: dict[str, float],
        drift_fired: bool,
        fire_reason: str = "",
    ) -> int:
        row = DriftScoreRow(
            window_label=window_label,
            n_rows=int(n_rows),
            proba_psi=float(proba_psi),
            flagged_features=list(flagged_features),
            feature_ks_stats=dict(feature_ks_stats),
            drift_fired=bool(drift_fired),
            fire_reason=fire_reason,
        )
        with self._write_lock, self.session() as s:
            s.add(row)
            s.flush()
            return int(row.id)

    def log_retrain_event(self, event: dict[str, Any]) -> int:
        row = RetrainEventRow(
            triggered_on_day=int(event["triggered_on_day"]),
            consecutive_days=int(event["consecutive_days"]),
            train_rows=int(event["train_rows"]),
            shadow_rows=int(event["shadow_rows"]),
            shadow_auprc=float(event["shadow_auprc"]),
            shadow_auc=float(event["shadow_auc"]),
            prod_auprc=float(event["prod_auprc"]),
            promote_decision=bool(event["promote_decision"]),
            promote_reason=str(event.get("promote_reason", "")),
            new_model_version=str(event.get("new_model_version") or ""),
            run_id=str(event.get("run_id") or ""),
            seconds_end_to_end=float(event.get("seconds_end_to_end", 0.0)),
        )
        with self._write_lock, self.session() as s:
            s.add(row)
            s.flush()
            return int(row.id)

    def log_registry_flip(
        self,
        *,
        action: str,
        model_name: str,
        alias: str,
        from_version: str | None,
        to_version: str,
        flip_seconds: float,
        actor: str = "auto",
        reason: str = "",
    ) -> int:
        if action not in {"promote", "rollback"}:
            raise ValueError("action must be 'promote' or 'rollback'")
        row = ModelRegistryLogRow(
            action=action,
            model_name=model_name,
            alias=alias,
            from_version=from_version,
            to_version=to_version,
            flip_seconds=float(flip_seconds),
            actor=actor,
            reason=reason,
        )
        with self._write_lock, self.session() as s:
            s.add(row)
            s.flush()
            return int(row.id)

    # ----- Reads ------------------------------------------------------------

    def recent_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.query(PredictionRow)
                .order_by(PredictionRow.created_at.desc(), PredictionRow.id.desc())
                .limit(limit)
                .all()
            )
            return [self._prediction_to_dict(r) for r in rows]

    def prediction_summary(self, hours: float = 24.0) -> dict[str, Any]:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(hours=hours)
        with self.session() as s:
            rows = s.query(PredictionRow).filter(PredictionRow.created_at >= cutoff).all()
            by_role: dict[str, dict[str, Any]] = {}
            for r in rows:
                bucket = by_role.setdefault(r.role, {"count": 0, "fraud_predicted": 0, "latencies": [], "probas": []})
                bucket["count"] += 1
                bucket["fraud_predicted"] += int(r.label)
                bucket["latencies"].append(float(r.latency_ms))
                bucket["probas"].append(float(r.probability))
            summary: dict[str, Any] = {"window_hours": hours, "by_role": {}}
            for role, bucket in by_role.items():
                lat = bucket["latencies"]
                probas = bucket["probas"]
                summary["by_role"][role] = {
                    "count": bucket["count"],
                    "fraud_predicted": bucket["fraud_predicted"],
                    "fraud_rate": bucket["fraud_predicted"] / bucket["count"] if bucket["count"] else 0.0,
                    "latency_ms_mean": sum(lat) / len(lat) if lat else 0.0,
                    "latency_ms_p95": _percentile(lat, 95.0) if lat else 0.0,
                    "proba_mean": sum(probas) / len(probas) if probas else 0.0,
                }
            return summary

    def recent_drift(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.query(DriftScoreRow)
                .order_by(DriftScoreRow.created_at.desc(), DriftScoreRow.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "window_label": r.window_label,
                    "n_rows": r.n_rows,
                    "proba_psi": r.proba_psi,
                    "flagged_features": r.flagged_features,
                    "feature_ks_stats": r.feature_ks_stats,
                    "drift_fired": bool(r.drift_fired),
                    "fire_reason": r.fire_reason,
                }
                for r in rows
            ]

    def recent_retrain_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.query(RetrainEventRow)
                .order_by(RetrainEventRow.created_at.desc(), RetrainEventRow.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "triggered_on_day": r.triggered_on_day,
                    "consecutive_days": r.consecutive_days,
                    "train_rows": r.train_rows,
                    "shadow_rows": r.shadow_rows,
                    "shadow_auprc": r.shadow_auprc,
                    "shadow_auc": r.shadow_auc,
                    "prod_auprc": r.prod_auprc,
                    "promote_decision": bool(r.promote_decision),
                    "promote_reason": r.promote_reason,
                    "new_model_version": r.new_model_version,
                    "run_id": r.run_id,
                    "seconds_end_to_end": r.seconds_end_to_end,
                }
                for r in rows
            ]

    def recent_registry_log(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = (
                s.query(ModelRegistryLogRow)
                .order_by(ModelRegistryLogRow.created_at.desc(), ModelRegistryLogRow.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "action": r.action,
                    "model_name": r.model_name,
                    "alias": r.alias,
                    "from_version": r.from_version,
                    "to_version": r.to_version,
                    "flip_seconds": r.flip_seconds,
                    "actor": r.actor,
                    "reason": r.reason,
                }
                for r in rows
            ]

    @staticmethod
    def _prediction_to_dict(r: PredictionRow) -> dict[str, Any]:
        return {
            "id": r.id,
            "request_id": r.request_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "role": r.role,
            "probability": r.probability,
            "label": int(r.label),
            "threshold": r.threshold,
            "latency_ms": r.latency_ms,
            "features": r.features,
            "extra": r.extra,
        }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


__all__ = [
    "DriftScoreRow",
    "ModelRegistryLogRow",
    "PredictionRow",
    "RetrainEventRow",
    "TelemetryConfig",
    "TelemetryLogger",
]
