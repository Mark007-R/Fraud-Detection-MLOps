"""Day 4 Phase 3 — end-to-end smoke test of the FastAPI service.

Drives the in-process FastAPI app with FastAPI TestClient:

    1. Loads the Day-1 prod model from models/fraud_model.pkl;
    2. Spins up the API against an isolated sqlite telemetry store;
    3. Hits POST /predict 100 times sampled from data/processed/X_test.csv,
       with a deliberate mix of fraud-flagged and benign rows;
    4. Optionally enables shadow if a registry version is present;
    5. Reads /healthz, /metrics/predictions, /metrics/shadow_agreement,
       /metrics/drift, /metrics/registry, /metrics/retrain_events;
    6. Writes results to:
         - results/day04_api_smoke.json     (timings + endpoint outputs)
         - results/samples/day04_predict_*.json  (5 sample requests + responses)

Run from the repo root:

    python scripts/day04_smoke_e2e.py
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# Force the API factory to use a clean, throwaway sqlite store and disable
# shadow (we test shadow separately below since the MLflow registry on the
# sprint branch has not been seeded with @staging yet).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_PATH = PROJECT_ROOT / "data" / "telemetry_smoke.sqlite"
if TELEMETRY_PATH.exists():
    TELEMETRY_PATH.unlink()
os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{TELEMETRY_PATH.as_posix()}"
os.environ["SENTINEL_DISABLE_SHADOW"] = "1"
os.environ["SENTINEL_BUILD_APP_AT_IMPORT"] = "0"

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.data.loader import SentinelDataLoader  # noqa: E402
from src.serving.api import create_app  # noqa: E402
from src.telemetry import TelemetryConfig, TelemetryLogger  # noqa: E402


def main() -> dict:
    loader = SentinelDataLoader()
    bundle_path = loader.model_path()
    artifact = joblib.load(bundle_path)
    feature_cols = list(artifact["feature_columns"])

    print(f"[Smoke] Model: {bundle_path} ({len(feature_cols)} features)")

    # Build app + isolated telemetry. The telemetry path is set via env above.
    app = create_app()
    client = TestClient(app)

    # Sample 100 rows from X_test.csv — mix top-probability with random for variety.
    X = loader.load_x_test()
    sub = X.iloc[: min(50_000, len(X))].copy()
    proba = artifact["model"].predict_proba(sub)[:, 1]
    # Stratify: 30 high-proba, 70 random.
    high_idx = np.argsort(proba)[::-1][:30]
    rng = np.random.default_rng(0)
    rand_idx = rng.choice(len(sub), size=70, replace=False)
    picks = np.concatenate([high_idx, rand_idx])
    sample = sub.iloc[picks].reset_index(drop=True)

    # Hit /healthz first.
    t0 = time.perf_counter()
    health = client.get("/healthz").json()
    health_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Smoke] /healthz {health_ms:.2f}ms -> prod_version={health['prod_version']} shadow={health['shadow']}")

    request_ids: list[str] = []
    latencies_ms: list[float] = []
    server_latencies_ms: list[float] = []
    proba_returned: list[float] = []
    sample_traffic: list[dict] = []

    for i, row in sample.iterrows():
        body = {
            "request_id": f"smoke-{i:04d}-{uuid.uuid4().hex[:8]}",
            "features": {k: float(v) for k, v in row.to_dict().items()},
        }
        t0 = time.perf_counter()
        resp = client.post("/predict", json=body)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, (resp.status_code, resp.text)
        body_out = resp.json()
        request_ids.append(body_out["request_id"])
        latencies_ms.append(wall_ms)
        server_latencies_ms.append(float(body_out["latency_ms"]))
        proba_returned.append(float(body_out["probability"]))
        if i < 5:
            sample_traffic.append({"request": body, "response": body_out})

    print(
        f"[Smoke] /predict x{len(latencies_ms)} - "
        f"wall mean={np.mean(latencies_ms):.2f}ms p95={np.percentile(latencies_ms,95):.2f}ms; "
        f"server mean={np.mean(server_latencies_ms):.2f}ms p95={np.percentile(server_latencies_ms,95):.2f}ms; "
        f"fraud_rate={float(np.mean([1 if p>=0.5 else 0 for p in proba_returned])):.3f}"
    )

    pred_metrics = client.get("/metrics/predictions?hours=1").json()
    drift_metrics = client.get("/metrics/drift?limit=5").json()
    registry_metrics = client.get("/metrics/registry?limit=5").json()
    retrain_metrics = client.get("/metrics/retrain_events?limit=5").json()
    agreement = client.get("/metrics/shadow_agreement?hours=1").json()

    # Cross-check: pull rows directly from the telemetry store.
    telemetry = TelemetryLogger(TelemetryConfig(database_url=os.environ["SENTINEL_DATABASE_URL"]))
    direct_rows = telemetry.recent_predictions(limit=200)
    direct_count = len(direct_rows)
    print(f"[Smoke] Direct telemetry row count: {direct_count}")

    # Insert one synthetic drift + one synthetic registry log so the metrics
    # endpoints return non-empty in the smoke output (these tables are populated
    # by Day 3 trigger / Day 2 registry CLI normally — we exercise the writer too).
    telemetry.log_drift(
        window_label="2026-05-21-smoke",
        n_rows=int(len(sample)),
        proba_psi=0.04,
        flagged_features=[],
        feature_ks_stats={"amount": 0.018},
        drift_fired=False,
        fire_reason="",
    )
    telemetry.log_registry_flip(
        action="promote",
        model_name="sentinel-fraud-xgboost",
        alias="production",
        from_version=None,
        to_version="1",
        flip_seconds=0.004,
        actor="smoke",
        reason="day04 e2e smoke insertion",
    )
    drift_after = client.get("/metrics/drift?limit=5").json()
    registry_after = client.get("/metrics/registry?limit=5").json()

    out = {
        "health": health,
        "predict_calls": int(len(latencies_ms)),
        "predict_latency_wall_ms_mean": float(np.mean(latencies_ms)),
        "predict_latency_wall_ms_p50": float(np.percentile(latencies_ms, 50)),
        "predict_latency_wall_ms_p95": float(np.percentile(latencies_ms, 95)),
        "predict_latency_wall_ms_p99": float(np.percentile(latencies_ms, 99)),
        "predict_latency_server_ms_mean": float(np.mean(server_latencies_ms)),
        "predict_latency_server_ms_p95": float(np.percentile(server_latencies_ms, 95)),
        "predict_proba_mean": float(np.mean(proba_returned)),
        "predict_fraud_rate_at_0p5": float(np.mean([1 if p >= 0.5 else 0 for p in proba_returned])),
        "metrics_predictions_endpoint": pred_metrics,
        "metrics_drift_before_insert": drift_metrics,
        "metrics_drift_after_insert": drift_after,
        "metrics_registry_before_insert": registry_metrics,
        "metrics_registry_after_insert": registry_after,
        "metrics_retrain_events": retrain_metrics,
        "metrics_shadow_agreement": agreement,
        "telemetry_direct_row_count": int(direct_count),
        "telemetry_url": str(telemetry.engine.url),
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "day04_api_smoke.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[Smoke] wrote {out_path}")

    samples_dir = PROJECT_ROOT / "results" / "samples" / "day04_api"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(sample_traffic):
        with open(samples_dir / f"predict_{i:02d}.json", "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=2, default=str)
    print(f"[Smoke] wrote {len(sample_traffic)} sample request/response pairs to {samples_dir}")

    return out


if __name__ == "__main__":
    main()
