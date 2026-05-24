"""FastAPI serving layer basics -- Day 4 Phase 3, Day-7 hardened.

Exercises /healthz, /predict, /metrics/predictions against:
- an isolated sqlite telemetry store in tmp_path,
- a SYNTHETIC XGBoost model + feature columns injected via
  ``create_app(bundle=...)``.

Day 7 (2026-05-24): the original fixture loaded ``models/fraud_model.pkl``
and ``data/processed/X_test.csv`` from disk -- both DVC-tracked and absent
on a fresh CI runner. The synthetic bundle below removes that dependency,
so this whole module now runs on a hermetic environment with zero data
state. Trains in ~50 ms.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier


# The four made-up feature columns the synthetic model is trained on. Their
# names don't matter -- create_app(bundle=...) trusts the bundle's column list
# verbatim, and /predict reindexes the request payload onto these.
_FEATURE_COLS = ["f0", "f1", "f2", "f3"]


def _train_synthetic_bundle():
    """Train a tiny XGB on 200 random rows and wrap it as a ModelBundle.

    The model never has to be accurate -- it just has to be a valid
    ``predict_proba``-capable estimator the API can score against.
    """
    from src.serving.api import ModelBundle

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, len(_FEATURE_COLS))), columns=_FEATURE_COLS)
    # Cheap, learnable signal so XGB doesn't trip on a degenerate label vector.
    y = ((X["f0"] + X["f1"] - X["f2"]) > 0).astype(int).to_numpy()
    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=0,
    )
    model.fit(X, y)
    return ModelBundle(
        model=model,
        feature_columns=list(_FEATURE_COLS),
        version="test-synthetic",
        source="synthetic://test_api",
        loaded_at=pd.Timestamp.utcnow().isoformat(),
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp_dir = tmp_path_factory.mktemp("api-test")
    os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{(tmp_dir / 'tel.sqlite').as_posix()}"
    os.environ["SENTINEL_DISABLE_SHADOW"] = "1"
    os.environ["SENTINEL_BUILD_APP_AT_IMPORT"] = "0"

    from src.serving.api import create_app

    bundle = _train_synthetic_bundle()
    app = create_app(bundle=bundle)
    return TestClient(app)


def _sample_features(client: TestClient) -> dict[str, float]:
    """Build a synthetic feature payload matching the injected bundle."""
    bundle = client.app.state.bundle
    rng = np.random.default_rng(1)
    values = rng.normal(size=len(bundle.feature_columns))
    return {col: float(v) for col, v in zip(bundle.feature_columns, values)}


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["prod_version"] == "test-synthetic"
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
