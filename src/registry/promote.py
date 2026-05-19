"""MLflow Model Registry — promote a run's logged model to a named alias.

Day 2 Phase 2a (2026-05-19). The Day-1 training script logs an XGBoost
artifact under ``artifact_path="xgboost_model"`` for every run. This CLI

    1. Looks up the run by ``--run-id`` (or by the latest run in ``--experiment``);
    2. Calls ``client.create_model_version`` to register that run's artifact
       under the registered model name (creating the registry entry on first
       use);
    3. Optionally sets a named alias (default: ``production``) onto the new
       version. **Aliases**, not the deprecated "Production" stage, are how
       MLflow 2.9+ designates the canonical version.

The function ``promote(...)`` is also importable from ``src/registry/rollback.py``
and from the throughput-/rollback-bench harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import mlflow
from mlflow.tracking import MlflowClient

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT


DEFAULT_MODEL_NAME = "sentinel-fraud-xgboost"
DEFAULT_ARTIFACT_PATH = "xgboost_model"
DEFAULT_ALIAS = "production"


def _resolve_tracking_uri() -> str:
    """Use the same sqlite-backed store the Day-1 training run wrote to."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )


@dataclass
class PromotionResult:
    """Outcome of a promote call (used by the rollback bench harness)."""

    model_name: str
    version: str
    alias: str | None
    run_id: str
    source_uri: str
    register_seconds: float
    alias_set_seconds: float


def _latest_run_in_experiment(client: MlflowClient, experiment_name: str) -> str:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"[Registry] Experiment '{experiment_name}' not found")
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"[Registry] No runs in experiment '{experiment_name}'")
    return runs[0].info.run_id


def promote(
    run_id: str | None = None,
    *,
    experiment: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    alias: str | None = DEFAULT_ALIAS,
    description: str | None = None,
    tracking_uri: str | None = None,
) -> PromotionResult:
    """Register a run's model into the registry and (optionally) tag with alias.

    Parameters
    ----------
    run_id : str | None
        MLflow run to register. If omitted, ``experiment`` must be provided
        and the latest run in that experiment is used.
    experiment : str | None
        Experiment to look in if ``run_id`` is not given.
    model_name : str
        Registered model name; created on first call.
    artifact_path : str
        Artifact subpath under the run where the model lives.
    alias : str | None
        Alias to set on the new version (e.g., ``"production"``). Pass
        ``None`` to register without aliasing — useful when staging a
        candidate version before shadow eval.
    description : str | None
        Optional description string set on the version.

    Returns
    -------
    PromotionResult
        Includes the new version number, alias, and per-step latencies.
    """
    mlflow.set_tracking_uri(tracking_uri or _resolve_tracking_uri())
    client = MlflowClient()

    if run_id is None:
        if experiment is None:
            raise ValueError("[Registry] Pass --run-id or --experiment")
        run_id = _latest_run_in_experiment(client, experiment)
        print(f"[Registry] Resolved latest run in '{experiment}' -> {run_id}")

    # Ensure the registered-model container exists (idempotent).
    try:
        client.create_registered_model(model_name)
        print(f"[Registry] Created registered model '{model_name}'")
    except mlflow.exceptions.MlflowException as exc:
        msg = str(exc).lower()
        if "already exists" not in msg and "resource_already_exists" not in msg:
            raise
        print(f"[Registry] Reusing existing registered model '{model_name}'")

    source_uri = f"runs:/{run_id}/{artifact_path}"
    t0 = time.perf_counter()
    mv = client.create_model_version(name=model_name, source=source_uri, run_id=run_id)
    register_seconds = time.perf_counter() - t0
    print(
        f"[Registry] Registered version v{mv.version} from {source_uri} "
        f"in {register_seconds:.3f}s"
    )

    if description:
        client.update_model_version(name=model_name, version=mv.version, description=description)

    alias_set_seconds = 0.0
    if alias:
        t0 = time.perf_counter()
        client.set_registered_model_alias(name=model_name, alias=alias, version=mv.version)
        alias_set_seconds = time.perf_counter() - t0
        print(
            f"[Registry] Set alias '@{alias}' -> v{mv.version} "
            f"in {alias_set_seconds:.3f}s"
        )

    return PromotionResult(
        model_name=model_name,
        version=str(mv.version),
        alias=alias,
        run_id=run_id,
        source_uri=source_uri,
        register_seconds=register_seconds,
        alias_set_seconds=alias_set_seconds,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a run's model in the MLflow registry and set an alias.")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run id to register.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="sentinel-day01-temporal-split",
        help="Experiment to pull latest run from when --run-id is omitted.",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--artifact-path", type=str, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--alias",
        type=str,
        default=DEFAULT_ALIAS,
        help="Alias to assign to the new version (default 'production'). Pass empty string to skip.",
    )
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    promote(
        run_id=args.run_id,
        experiment=args.experiment,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        alias=(args.alias or None),
        description=args.description,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
