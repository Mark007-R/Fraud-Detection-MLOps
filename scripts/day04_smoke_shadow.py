"""Day 4 Phase 3 — shadow-path smoke test.

Sets up MLflow registry aliases (@production v1, @staging v9) using the
already-registered versions from Day 3's auto-retrain runs, brings the API
up with shadow enabled, fires predictions, and verifies the
``/metrics/shadow_agreement`` endpoint pairs them correctly.

Run from the repo root:

    python scripts/day04_smoke_shadow.py

Output:
    results/day04_shadow_smoke.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TELEMETRY_PATH = PROJECT_ROOT / "data" / "telemetry_shadow_smoke.sqlite"
if TELEMETRY_PATH.exists():
    TELEMETRY_PATH.unlink()
os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{TELEMETRY_PATH.as_posix()}"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
os.environ["SENTINEL_DISABLE_SHADOW"] = "0"  # enable shadow
os.environ["SENTINEL_BUILD_APP_AT_IMPORT"] = "0"

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.data.loader import SentinelDataLoader  # noqa: E402
from src.serving.api import create_app  # noqa: E402
from src.serving.shadow import ShadowConfig  # noqa: E402

MODEL_NAME = "sentinel-fraud-xgboost"


def setup_aliases() -> dict[str, str]:
    """Pin @production and @staging to known versions from the registry."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    if len(versions) < 2:
        raise RuntimeError(
            f"Need at least 2 versions of '{MODEL_NAME}'; found {len(versions)}. "
            f"Run Day 2 / Day 3 first."
        )
    prod_version = versions[0].version  # v1 — original Day-1 temporal-split model
    staging_version = versions[-1].version  # latest auto-retrain candidate
    client.set_registered_model_alias(MODEL_NAME, "production", prod_version)
    client.set_registered_model_alias(MODEL_NAME, "staging", staging_version)
    print(f"[Shadow-Smoke] @production -> v{prod_version}, @staging -> v{staging_version}")
    return {"production": str(prod_version), "staging": str(staging_version)}


def main() -> dict:
    aliases = setup_aliases()

    app = create_app(shadow_config=ShadowConfig(model_name=MODEL_NAME, enabled=True))
    client = TestClient(app)

    health = client.get("/healthz").json()
    print(f"[Shadow-Smoke] /healthz prod={health['prod_version']} shadow={health['shadow']}")

    # 40 predictions, mix of high-proba + random rows.
    loader = SentinelDataLoader()
    X = loader.load_x_test().iloc[:5000].copy()
    import joblib
    artifact = joblib.load(loader.model_path())
    proba = artifact["model"].predict_proba(X)[:, 1]
    high_idx = np.argsort(proba)[::-1][:10]
    rng = np.random.default_rng(7)
    rand_idx = rng.choice(len(X), size=30, replace=False)
    picks = np.concatenate([high_idx, rand_idx])
    sample = X.iloc[picks].reset_index(drop=True)

    request_ids: list[str] = []
    latencies_ms: list[float] = []

    for i, row in sample.iterrows():
        body = {
            "request_id": f"shadow-smoke-{i:04d}-{uuid.uuid4().hex[:6]}",
            "features": {k: float(v) for k, v in row.to_dict().items()},
        }
        t0 = time.perf_counter()
        resp = client.post("/predict", json=body)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, (resp.status_code, resp.text)
        request_ids.append(resp.json()["request_id"])
        latencies_ms.append(wall_ms)

    # Give the shadow executor time to flush. It's a small daemon thread pool.
    if app.state.shadow is not None:
        app.state.shadow.shutdown(wait=True)
        # Rebuild a no-op shadow so app teardown doesn't double-shutdown.

    agreement = client.get("/metrics/shadow_agreement?hours=1").json()
    pred_metrics = client.get("/metrics/predictions?hours=1").json()
    print(f"[Shadow-Smoke] shadow_agreement: {agreement}")
    print(f"[Shadow-Smoke] prediction stats: {pred_metrics}")

    out = {
        "aliases": aliases,
        "health": health,
        "predict_calls": int(len(latencies_ms)),
        "predict_wall_ms_mean": float(np.mean(latencies_ms)),
        "predict_wall_ms_p95": float(np.percentile(latencies_ms, 95)),
        "agreement": agreement,
        "prediction_metrics": pred_metrics,
    }
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "day04_shadow_smoke.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[Shadow-Smoke] wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
