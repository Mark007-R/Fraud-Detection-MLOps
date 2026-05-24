"""MLflow registry promote + rollback -- Day 7 Phase 6 (2026-05-24).

Spins up a hermetic sqlite-backed MLflow tracking store inside the test's
tmp_path, logs two trivial sklearn models as separate runs, promotes each
into the registry, flips the @production alias, then rolls back. Asserts:

- `promote()` returns a PromotionResult with monotonically increasing version
  numbers across calls,
- the alias points at the most recently promoted version,
- `rollback()` flips the alias to the prior version and records sub-second
  alias-flip latency,
- the post-rollback alias resolves to the prior version via the MLflow API
  (so the "tested rollback" claim is exercised end-to-end).

Hard rule #6 of the sprint: "A registry without a tested rollback is theater."
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _set_mlflow_env(tmp_path: Path) -> None:
    db = tmp_path / "mlflow.db"
    artifacts = tmp_path / "mlartifacts"
    artifacts.mkdir(exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db.as_posix()}"
    # Force the artifact root onto the same tmp tree so registered model
    # versions can be created via runs:/<run_id>/<artifact_path>.
    os.environ["MLFLOW_ARTIFACT_LOCATION"] = artifacts.as_posix()


def _log_dummy_model(experiment_name: str, run_name: str, c_value: float) -> str:
    """Log a tiny sklearn model under 'sklearn_model' artifact path. Returns run_id."""
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1])
        clf = LogisticRegression(C=c_value).fit(X, y)
        mlflow.log_param("C", c_value)
        mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")
        return run.info.run_id


@pytest.fixture()
def mlflow_env(tmp_path: Path) -> Path:
    _set_mlflow_env(tmp_path)
    # Import after env is set so the tracking URI is picked up.
    import importlib
    import mlflow

    importlib.reload(mlflow)
    return tmp_path


def test_promote_then_rollback_flips_alias(mlflow_env: Path) -> None:
    from src.registry.promote import promote
    from src.registry.rollback import rollback
    from mlflow.tracking import MlflowClient

    model_name = "sentinel-test-registry"
    experiment = "sentinel-test-registry-exp"
    artifact = "sklearn_model"

    run_v1 = _log_dummy_model(experiment, "v1", c_value=1.0)
    run_v2 = _log_dummy_model(experiment, "v2", c_value=10.0)

    p1 = promote(
        run_id=run_v1,
        model_name=model_name,
        artifact_path=artifact,
        alias="production",
        description="test v1",
    )
    p2 = promote(
        run_id=run_v2,
        model_name=model_name,
        artifact_path=artifact,
        alias="production",
        description="test v2",
    )

    assert int(p2.version) > int(p1.version), (
        f"expected v2.version > v1.version, got {p1.version=}, {p2.version=}"
    )
    assert p2.alias == "production"
    # Promotion is a single sqlite write -- on local store should be fast.
    assert p2.alias_set_seconds < 1.0

    client = MlflowClient()
    aliased = client.get_model_version_by_alias(model_name, "production")
    assert str(aliased.version) == str(p2.version)

    rb = rollback(
        model_name=model_name,
        alias="production",
        target_version=p1.version,
        previous_alias="previous",
    )
    assert str(rb.to_version) == str(p1.version)
    assert str(rb.from_version) == str(p2.version)
    assert rb.alias_flip_seconds < 1.0, (
        f"alias flip should be sub-second on sqlite, got {rb.alias_flip_seconds:.3f}s"
    )

    # The rollback flipped the live alias to v1 -- confirm via the API.
    post = client.get_model_version_by_alias(model_name, "production")
    assert str(post.version) == str(p1.version)

    # And the rolled-back version is tagged for the audit trail.
    tags = client.get_model_version(name=model_name, version=p2.version).tags
    assert "rolled_back_at" in tags


def test_rollback_into_current_alias_raises(mlflow_env: Path) -> None:
    from src.registry.promote import promote
    from src.registry.rollback import rollback

    model_name = "sentinel-test-registry-noop"
    experiment = "sentinel-test-registry-noop-exp"
    run_v1 = _log_dummy_model(experiment, "v1", c_value=1.0)

    p1 = promote(
        run_id=run_v1,
        model_name=model_name,
        artifact_path="sklearn_model",
        alias="production",
    )
    with pytest.raises(ValueError):
        rollback(
            model_name=model_name,
            alias="production",
            target_version=p1.version,
            previous_alias=None,
        )
