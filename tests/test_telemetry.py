"""Telemetry writer + reader basics — Day 4 Phase 3."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.telemetry import TelemetryConfig, TelemetryLogger


@pytest.fixture()
def logger(tmp_path: Path) -> TelemetryLogger:
    cfg = TelemetryConfig(database_url=f"sqlite:///{(tmp_path / 'tel.sqlite').as_posix()}")
    return TelemetryLogger(cfg)


def test_log_and_read_predictions(logger: TelemetryLogger) -> None:
    logger.log_prediction(
        model_name="m", model_version="v1", role="production",
        probability=0.9, latency_ms=10.0,
    )
    logger.log_prediction(
        model_name="m", model_version="v2", role="shadow",
        probability=0.7, latency_ms=12.0,
    )
    summary = logger.prediction_summary(hours=24.0)
    assert summary["by_role"]["production"]["count"] == 1
    assert summary["by_role"]["shadow"]["count"] == 1
    assert summary["by_role"]["production"]["fraud_predicted"] == 1


def test_log_drift_round_trip(logger: TelemetryLogger) -> None:
    logger.log_drift(
        window_label="2026-05-21",
        n_rows=100,
        proba_psi=0.05,
        flagged_features=["amount"],
        feature_ks_stats={"amount": 0.2},
        drift_fired=True,
        fire_reason="psi 0.05",
    )
    rows = logger.recent_drift(limit=5)
    assert len(rows) == 1
    assert rows[0]["drift_fired"] is True
    assert rows[0]["flagged_features"] == ["amount"]


def test_log_registry_round_trip(logger: TelemetryLogger) -> None:
    logger.log_registry_flip(
        action="promote", model_name="m", alias="production",
        from_version=None, to_version="1", flip_seconds=0.004,
    )
    logger.log_registry_flip(
        action="rollback", model_name="m", alias="production",
        from_version="1", to_version="2", flip_seconds=0.003,
    )
    rows = logger.recent_registry_log(limit=5)
    assert len(rows) == 2
    assert {r["action"] for r in rows} == {"promote", "rollback"}


def test_role_validation(logger: TelemetryLogger) -> None:
    with pytest.raises(ValueError):
        logger.log_prediction(
            model_name="m", model_version="v1", role="not-a-role",
            probability=0.5, latency_ms=1.0,
        )
