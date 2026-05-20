"""Auto-retrain trigger — N-consecutive-day drift -> retrain -> shadow eval -> promote.

Day 3 Phase 2b (2026-05-20). The drift detector in
``src/drift/detector.py`` produces a daily fire/no-fire signal. This module
turns that signal into a *controlled* retrain decision:

    1. **Consecutive-day rule.** A single noisy day is not enough to ship a
       new model. The trigger watches the running ``consecutive_fires``
       counter; only when it reaches ``n_consecutive_days`` does retraining
       fire. This is the policy difference between a noisy alert system and
       an auto-deploy gate.

    2. **Retrain on the latest N-day window.** We re-fit XGBoost on exactly
       the window that drifted — the assumption is that the drifted
       distribution IS the new production distribution. Using a longer
       window would dilute the new signal with stale data.

    3. **Shadow eval on the most recent 24h.** Before flipping the
       ``@production`` alias, score the candidate on the *latest* observed
       day's labelled data and compute AUPRC. This is the gate.

    4. **Auto-promote if shadow_auprc >= prod_auprc - tolerance.** The
       tolerance is 1pp by default — i.e. the new model has to be no worse
       than 1pp below current production on the latest window. The
       asymmetric tolerance reflects the asymmetric cost: leaving a stale
       model in place under drift loses real money daily, but
       shipping a worse model is also bad.

    5. **Always register, sometimes promote.** Every retrain creates a new
       MLflow registry version (so it is auditable + rollback-able). The
       alias only flips when the gate passes — and the prior version stays
       under the ``previous`` alias so the Day-2 rollback CLI can flip back
       in 4ms if shadow eval was over-optimistic.

End-to-end "drift detected -> traffic on new model" latency is measured.
That is Sentinel's headline MLOps number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROJECT_ROOT
from src.drift.detector import DriftDetector
from src.registry.promote import promote


DEFAULT_MODEL_NAME = "sentinel-fraud-xgboost"
EXPERIMENT_NAME = "sentinel-day03-drift-retrain"


def _setup_mlflow() -> None:
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


@dataclass
class RetrainEvent:
    """Outcome of a single retrain attempt — emitted to the audit log."""

    first_fired_day: int
    consecutive_days: int
    triggered_on_day: int
    train_window_days: list[int]
    train_rows: int
    shadow_day: int
    shadow_rows: int
    shadow_auprc: float
    shadow_auc: float
    prod_auprc: float
    promote_decision: bool
    promote_reason: str
    new_model_version: str | None = None
    run_id: str | None = None
    seconds_detect_to_retrain_start: float = 0.0
    seconds_train: float = 0.0
    seconds_shadow_eval: float = 0.0
    seconds_register_and_alias: float = 0.0
    seconds_end_to_end: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TriggerState:
    """Stateful counter used to debounce single-day noise."""

    consecutive_fires: int = 0
    first_fired_day: int | None = None
    history: list[bool] = field(default_factory=list)

    def step(self, fired: bool, day: int, *, n_consecutive_days: int) -> bool:
        """Advance one day. Returns True iff today is the trigger day."""
        self.history.append(bool(fired))
        if fired:
            if self.consecutive_fires == 0:
                self.first_fired_day = day
            self.consecutive_fires += 1
        else:
            self.consecutive_fires = 0
            self.first_fired_day = None
        return self.consecutive_fires >= n_consecutive_days


def _train_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 50.0,
    run_name: str = "auto_retrain",
) -> tuple[XGBClassifier, str, float]:
    """Fit a candidate XGB inside an MLflow run; return (model, run_id, fit_seconds)."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "scale_pos_weight": scale_pos_weight,
                "trigger": "drift_auto_retrain",
                "train_rows": int(len(X_train)),
            }
        )
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        mlflow.log_metric("fit_seconds", fit_seconds)
        # Best-effort artifact logging; we still register from the resulting run.
        try:
            mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        except Exception as exc:
            print(f"[Trigger] mlflow.xgboost.log_model failed (non-fatal): {exc}")
        return model, run.info.run_id, fit_seconds


def _shadow_eval(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    if y.nunique() < 2:
        return 0.0, 0.0
    proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))
    auprc = float(average_precision_score(y, proba))
    return auprc, auc


def run_drift_retrain_simulation(
    *,
    daily_X: Sequence[pd.DataFrame],
    daily_y: Sequence[pd.Series],
    detector: DriftDetector,
    prod_model_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    n_consecutive_days: int = 2,
    promote_tolerance: float = 0.01,
    train_window_min_days: int = 2,
) -> tuple[TriggerState, list[RetrainEvent]]:
    """Replay daily windows, fire the trigger, retrain, eval, conditionally promote.

    Parameters
    ----------
    daily_X, daily_y : sequences indexed by day
        Feature + label frames for each daily window. ``daily_X[d]`` must
        contain the model's expected feature columns; extras are ignored.
    detector : DriftDetector
        Pre-configured with a saved reference.
    prod_model_path : Path
        Where the current production XGB artifact lives. Used both to score
        for drift and to compute the "prod_auprc" baseline on the shadow day.
    n_consecutive_days : int
        Debounce window — N consecutive firing days before retrain. Default 2.
    promote_tolerance : float
        Maximum AUPRC drop vs prod for auto-promotion. Default 0.01 (=1pp).
    train_window_min_days : int
        Floor on the retrain window. Even when the trigger debounces at N=2,
        we want at least this many days of newly-drifted data in the training
        set so the candidate sees a representative sample.

    Returns
    -------
    (state, events)
        Final TriggerState (history of per-day fire flags) plus the list of
        RetrainEvent records produced (one per trigger event).
    """
    _setup_mlflow()
    prod_artifact = joblib.load(prod_model_path)
    prod_model = prod_artifact["model"]
    feature_cols: list[str] = list(prod_artifact["feature_columns"])

    state = TriggerState()
    events: list[RetrainEvent] = []

    n_days = len(daily_X)
    for d in range(n_days):
        X_d = daily_X[d][feature_cols]
        y_d = daily_y[d]
        proba_d = prod_model.predict_proba(X_d)[:, 1]
        report = detector.score(X_d, proba_d, window_label=f"day_{d:02d}")
        fired = bool(report.drift_fired)
        triggered = state.step(fired, d, n_consecutive_days=n_consecutive_days)

        if not triggered:
            continue

        t_event_start = time.perf_counter()

        # 1) Build the retrain window from drift-firing days STRICTLY BEFORE the
        #    shadow day. Day d is the shadow window and MUST NOT appear in
        #    training, otherwise shadow AUPRC reports an in-sample number.
        first = state.first_fired_day if state.first_fired_day is not None else d
        window_end_exclusive = d  # shadow_day = d
        window_start = min(first, window_end_exclusive - train_window_min_days)
        window_start = max(window_start, 0)
        window_days = list(range(window_start, window_end_exclusive))
        if not window_days:
            window_days = [max(0, window_end_exclusive - 1)]
        X_train_parts = [daily_X[i][feature_cols] for i in window_days]
        y_train_parts = [daily_y[i] for i in window_days]
        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = pd.concat(y_train_parts, ignore_index=True).astype(int)

        seconds_detect_to_retrain_start = time.perf_counter() - t_event_start

        # 2) Train the candidate.
        candidate, run_id, fit_seconds = _train_candidate(
            X_train, y_train, run_name=f"day{d:02d}_auto_retrain"
        )

        # 3) Shadow eval on the last day's labelled data (d is "today" — the
        #    most recent fully-labelled window).
        t0 = time.perf_counter()
        shadow_X = daily_X[d][feature_cols]
        shadow_y = daily_y[d].astype(int)
        shadow_auprc, shadow_auc = _shadow_eval(candidate, shadow_X, shadow_y)
        prod_auprc, _ = _shadow_eval(prod_model, shadow_X, shadow_y)
        seconds_shadow_eval = time.perf_counter() - t0
        mlflow.log_metrics(
            {
                "shadow_auprc": shadow_auprc,
                "shadow_auc": shadow_auc,
                "prod_auprc": prod_auprc,
            },
            run_id=run_id,
        )

        # 4) Promote-or-not decision.
        promote_decision = shadow_auprc >= (prod_auprc - promote_tolerance)
        promote_reason = (
            f"shadow_auprc {shadow_auprc:.4f} >= prod_auprc {prod_auprc:.4f} - tol {promote_tolerance:.3f}"
            if promote_decision
            else f"shadow_auprc {shadow_auprc:.4f} < prod_auprc {prod_auprc:.4f} - tol {promote_tolerance:.3f}"
        )

        # 5) Always register; alias only on a pass.
        t0 = time.perf_counter()
        alias = "production" if promote_decision else None
        promo = promote(
            run_id=run_id,
            model_name=model_name,
            alias=alias,
            description=(
                f"day{d:02d} auto-retrain — window {window_days} — "
                f"shadow_auprc={shadow_auprc:.4f} vs prod={prod_auprc:.4f}"
            ),
        )
        seconds_register_and_alias = time.perf_counter() - t0

        seconds_end_to_end = time.perf_counter() - t_event_start
        mlflow.log_metric("seconds_end_to_end_retrain", seconds_end_to_end, run_id=run_id)

        events.append(
            RetrainEvent(
                first_fired_day=int(first),
                consecutive_days=int(state.consecutive_fires),
                triggered_on_day=int(d),
                train_window_days=window_days,
                train_rows=int(len(X_train)),
                shadow_day=int(d),
                shadow_rows=int(len(shadow_X)),
                shadow_auprc=round(shadow_auprc, 4),
                shadow_auc=round(shadow_auc, 4),
                prod_auprc=round(prod_auprc, 4),
                promote_decision=promote_decision,
                promote_reason=promote_reason,
                new_model_version=str(promo.version),
                run_id=run_id,
                seconds_detect_to_retrain_start=round(seconds_detect_to_retrain_start, 4),
                seconds_train=round(fit_seconds, 4),
                seconds_shadow_eval=round(seconds_shadow_eval, 4),
                seconds_register_and_alias=round(seconds_register_and_alias, 4),
                seconds_end_to_end=round(seconds_end_to_end, 4),
            )
        )

        if promote_decision:
            prod_model = candidate
            # Reset state — we have a freshly aligned model, debounce restarts.
            state.consecutive_fires = 0
            state.first_fired_day = None

    return state, events


__all__ = [
    "DEFAULT_MODEL_NAME",
    "EXPERIMENT_NAME",
    "RetrainEvent",
    "TriggerState",
    "run_drift_retrain_simulation",
]
