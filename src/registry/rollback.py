"""MLflow Model Registry — rollback the production alias to a prior version.

Day 2 Phase 2a (2026-05-19). A registry without a tested rollback is theater:
when a freshly promoted model degrades in shadow eval or in live AUPRC, ops
need to flip the ``@production`` alias back to a known-good prior version
with a single command in seconds, not minutes.

This CLI:

    1. Looks up all versions of ``--model-name`` in the registry;
    2. Resolves the target version: either the explicit ``--target-version``,
       or the version currently holding the ``--previous-alias`` (default:
       ``"previous"``), or the most recent version that is NOT the currently
       aliased one (auto-pick-prior heuristic);
    3. Flips the ``--alias`` (default ``"production"``) to that version;
    4. Demotes the previously aliased version by tagging it
       ``rolled_back_at=<utc-iso>`` for auditability.

Latency is measured for the alias-flip step alone — the per-step number that
matters for the runbook is "how fast can ops get traffic onto the known-good
version after they hit the button."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _dt
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
DEFAULT_ALIAS = "production"
DEFAULT_PREVIOUS_ALIAS = "previous"


def _resolve_tracking_uri() -> str:
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )


@dataclass
class RollbackResult:
    """Outcome + timing of a rollback flip."""

    model_name: str
    alias: str
    from_version: str | None
    to_version: str
    alias_flip_seconds: float
    audit_tag_seconds: float


def _resolve_current_aliased_version(
    client: MlflowClient, model_name: str, alias: str
) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return mv.version
    except mlflow.exceptions.MlflowException:
        return None


def _resolve_target_version(
    client: MlflowClient,
    model_name: str,
    *,
    target_version: str | None,
    previous_alias: str | None,
    current_version: str | None,
) -> str:
    if target_version is not None:
        return str(target_version)

    if previous_alias:
        try:
            mv = client.get_model_version_by_alias(model_name, previous_alias)
            return mv.version
        except mlflow.exceptions.MlflowException:
            pass

    # Auto-pick-prior: walk all versions newest-first and pick the most recent
    # that is NOT the currently aliased version.
    versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )
    for mv in versions:
        if current_version is None or str(mv.version) != str(current_version):
            return str(mv.version)
    raise ValueError(
        f"[Registry] No alternative version to rollback to for model '{model_name}'"
    )


def rollback(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    target_version: str | None = None,
    previous_alias: str | None = DEFAULT_PREVIOUS_ALIAS,
    tag_previous: bool = True,
    tracking_uri: str | None = None,
) -> RollbackResult:
    """Flip ``alias`` on ``model_name`` back to a prior version.

    Returns
    -------
    RollbackResult
        Includes the from/to versions and per-step latencies.
    """
    mlflow.set_tracking_uri(tracking_uri or _resolve_tracking_uri())
    client = MlflowClient()

    current_version = _resolve_current_aliased_version(client, model_name, alias)
    print(f"[Rollback] Current '@{alias}' on '{model_name}' = v{current_version}")

    target = _resolve_target_version(
        client,
        model_name,
        target_version=target_version,
        previous_alias=previous_alias,
        current_version=current_version,
    )
    if current_version is not None and str(target) == str(current_version):
        raise ValueError(
            f"[Rollback] Target v{target} already holds '@{alias}' on '{model_name}' — nothing to do"
        )
    print(f"[Rollback] Flipping '@{alias}' from v{current_version} -> v{target}")

    # Alias flip — the operation whose latency matters for ops.
    t0 = time.perf_counter()
    client.set_registered_model_alias(name=model_name, alias=alias, version=str(target))
    alias_flip_seconds = time.perf_counter() - t0
    print(f"[Rollback] Alias flip: {alias_flip_seconds:.4f}s")

    audit_tag_seconds = 0.0
    if tag_previous and current_version is not None:
        t0 = time.perf_counter()
        client.set_model_version_tag(
            name=model_name,
            version=str(current_version),
            key="rolled_back_at",
            value=_dt.datetime.utcnow().isoformat(timespec="seconds"),
        )
        # Also move the auto-pick-prior helper alias so a future rollback knows
        # which version was just unbound (cheap convention).
        if previous_alias:
            client.set_registered_model_alias(
                name=model_name, alias=previous_alias, version=str(current_version)
            )
        audit_tag_seconds = time.perf_counter() - t0
        print(f"[Rollback] Audit-tag + previous-alias bookkeeping: {audit_tag_seconds:.4f}s")

    return RollbackResult(
        model_name=model_name,
        alias=alias,
        from_version=current_version,
        to_version=str(target),
        alias_flip_seconds=alias_flip_seconds,
        audit_tag_seconds=audit_tag_seconds,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback the production alias to a prior model version.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", type=str, default=DEFAULT_ALIAS)
    parser.add_argument("--target-version", type=str, default=None, help="Explicit version to roll back to.")
    parser.add_argument(
        "--previous-alias",
        type=str,
        default=DEFAULT_PREVIOUS_ALIAS,
        help="Helper alias that tracks the previously-promoted version.",
    )
    parser.add_argument(
        "--no-tag-previous",
        action="store_true",
        help="Skip the post-rollback audit tag + previous-alias bookkeeping.",
    )
    parser.add_argument("--tracking-uri", type=str, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rollback(
        model_name=args.model_name,
        alias=args.alias,
        target_version=args.target_version,
        previous_alias=(args.previous_alias or None),
        tag_previous=not args.no_tag_previous,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
