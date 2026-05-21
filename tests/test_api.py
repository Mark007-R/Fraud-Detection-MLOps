"""FastAPI serving layer basics — Day 4 Phase 3.

Exercises /healthz, /predict, /metrics/predictions against an isolated
sqlite telemetry store. Shadow is disabled because the test env doesn't
require an MLflow registry to be seeded.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp_dir = tmp_path_factory.mktemp("api-test")
    os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{(tmp_dir / 'tel.sqlite').as_posix()}"
    os.environ["SENTINEL_DISABLE_SHADOW"] = "1"
    os.environ["SENTINEL_BUILD_APP_AT_IMPORT"] = "0"

    from src.serving.api import create_app

    app = create_app()
    return TestClient(app)


def _sample_features(client: TestClient) -> dict[str, float]:
    """Pull a real X_test row to use as a valid feature payload."""
    bundle = client.app.state.bundle
    import pandas as pd
    from src.data.loader import SentinelDataLoader
    X = SentinelDataLoader().load_x_test().iloc[:1]
    row = X.reindex(columns=bundle.feature_columns).iloc[0]
    return {k: float(v) for k, v in row.to_dict().items()}


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["prod_version"]
    # Shadow disabled in this test fixture.
    assert body["shadow"]["enabled"] is False


def test_predict_returns_proba_and_label(client: TestClient) -> None:
    body = {"features": _sample_features(client)}
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert 0.0 <= out["probability"] <= 1.0
    assert out["label"] in (0, 1)
    assert out["request_id"]
    assert out["latency_ms"] >= 0.0


def test_predict_empty_features_rejected(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422


def test_metrics_predictions_aggregates(client: TestClient) -> None:
    body = {"features": _sample_features(client)}
    for _ in range(3):
        assert client.post("/predict", json=body).status_code == 200
    r = client.get("/metrics/predictions?hours=1")
    assert r.status_code == 200
    payload = r.json()
    prod = payload["by_role"].get("production", {})
    assert prod.get("count", 0) >= 3
