"""Auto-retrain trigger debounce logic -- Day 7 Phase 6 (2026-05-24).

The full retrain simulation (`run_drift_retrain_simulation`) writes to MLflow
and trains XGBoost models, so its end-to-end exercise lives in the Day-3
script `tests/synthetic_drift.py` (which runs the 30-day replay and is gated
on the real dataset). Here we unit-test the debounce policy that decides
WHEN to retrain -- the part of the trigger that has no MLflow or model
dependencies and that absolutely must not regress.

Asserts:
- A single fire does not trigger (debounce n=2).
- Two consecutive fires DO trigger on the second day.
- A no-fire day resets the consecutive counter -- a fire-then-no-fire-then-fire
  sequence does not trigger.
- The first_fired_day is tracked across the active streak and reset on a gap.
"""

from __future__ import annotations

from src.drift.trigger import TriggerState


def test_single_fire_does_not_trigger() -> None:
    state = TriggerState()
    assert state.step(True, day=0, n_consecutive_days=2) is False
    assert state.consecutive_fires == 1
    assert state.first_fired_day == 0


def test_two_consecutive_fires_trigger_on_second_day() -> None:
    state = TriggerState()
    assert state.step(True, day=10, n_consecutive_days=2) is False
    triggered = state.step(True, day=11, n_consecutive_days=2)
    assert triggered is True
    assert state.consecutive_fires == 2
    assert state.first_fired_day == 10


def test_three_consecutive_required_when_n_is_three() -> None:
    state = TriggerState()
    assert state.step(True, day=0, n_consecutive_days=3) is False
    assert state.step(True, day=1, n_consecutive_days=3) is False
    assert state.step(True, day=2, n_consecutive_days=3) is True


def test_no_fire_day_resets_counter() -> None:
    state = TriggerState()
    state.step(True, day=0, n_consecutive_days=2)
    assert state.step(False, day=1, n_consecutive_days=2) is False
    assert state.consecutive_fires == 0
    assert state.first_fired_day is None
    # Now a single fire again -- still should not trigger.
    assert state.step(True, day=2, n_consecutive_days=2) is False


def test_streak_after_gap_tracks_new_first_fired_day() -> None:
    state = TriggerState()
    state.step(True, day=0, n_consecutive_days=2)
    state.step(False, day=1, n_consecutive_days=2)
    state.step(True, day=2, n_consecutive_days=2)
    triggered = state.step(True, day=3, n_consecutive_days=2)
    assert triggered is True
    # first_fired_day should be the start of the LATEST streak, not day 0.
    assert state.first_fired_day == 2


def test_history_records_per_day_signal() -> None:
    state = TriggerState()
    for day, fired in enumerate([True, False, True, True, False]):
        state.step(fired, day=day, n_consecutive_days=2)
    assert state.history == [True, False, True, True, False]
