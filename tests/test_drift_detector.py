"""Drift detector behavior -- Day 7 Phase 6 (2026-05-24).

Verifies the KS + PSI detector on synthetic data:
- score on the SAME distribution does not fire,
- score on a 2sigma-shifted feature fires the KS-only path,
- score on a different probability distribution fires the PSI path,
- PSI value is 0 when inputs are identical.

No data dependency, no MLflow, runs in <1s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift.detector import DriftDetector, fit_reference, psi


RNG_SEED = 0


def _ref_and_window(seed: int = RNG_SEED, n: int = 2_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame(
        {
            "amount": rng.lognormal(3.0, 1.0, size=n),
            "hour_of_day": rng.integers(0, 24, size=n),
            "card_amount_zscore": rng.normal(0, 1, size=n),
        }
    )
    rng2 = np.random.default_rng(seed + 1)
    same = pd.DataFrame(
        {
            "amount": rng2.lognormal(3.0, 1.0, size=n),
            "hour_of_day": rng2.integers(0, 24, size=n),
            "card_amount_zscore": rng2.normal(0, 1, size=n),
        }
    )
    return ref, same


def test_psi_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    assert psi(x, x) < 1e-6


def test_psi_fires_on_distribution_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=1.0, size=5_000)
    b = rng.normal(loc=2.0, scale=1.0, size=5_000)
    score = psi(a, b)
    assert score > 0.25, f"expected PSI >> 0.25 on a 2sigma loc shift, got {score:.3f}"


def test_detector_does_not_fire_on_identical_distribution() -> None:
    ref_df, same_df = _ref_and_window()
    ref = fit_reference(ref_df, feature_columns=list(ref_df.columns))
    det = DriftDetector(ref, ks_stat_threshold=0.15, psi_threshold=0.25)
    rng = np.random.default_rng(42)
    proba_ref = rng.uniform(0, 0.05, size=len(ref_df))
    # Rebuild reference with proba.
    ref = fit_reference(ref_df, feature_columns=list(ref_df.columns), proba=proba_ref)
    det = DriftDetector(ref)
    proba_now = np.random.default_rng(43).uniform(0, 0.05, size=len(same_df))
    report = det.score(same_df, proba_now, window_label="baseline")
    assert not report.drift_fired, (
        f"unexpected fire: psi={report.proba_psi:.3f} flagged={report.flagged_features}"
    )


def test_detector_fires_on_feature_shift() -> None:
    ref_df, _ = _ref_and_window()
    ref = fit_reference(ref_df, feature_columns=list(ref_df.columns))
    det = DriftDetector(ref, ks_stat_threshold=0.15, psi_threshold=0.25)

    rng = np.random.default_rng(99)
    n = len(ref_df)
    shifted = pd.DataFrame(
        {
            "amount": rng.lognormal(3.0 + 2.0, 1.0, size=n),  # 2sigma loc shift
            "hour_of_day": rng.integers(0, 24, size=n),
            "card_amount_zscore": rng.normal(0, 1, size=n),
        }
    )
    # No proba passed -- detector should still flag the feature path.
    report = det.score(shifted, proba=None, window_label="shifted")
    assert "amount" in report.flagged_features, (
        f"expected 'amount' to flag, got {report.flagged_features}"
    )


def test_detector_fires_on_proba_shift_alone() -> None:
    ref_df, same_df = _ref_and_window()
    rng = np.random.default_rng(7)
    proba_ref = rng.uniform(0, 0.05, size=len(ref_df))
    ref = fit_reference(ref_df, feature_columns=list(ref_df.columns), proba=proba_ref)
    det = DriftDetector(ref, psi_threshold=0.25)

    proba_now = np.random.default_rng(8).uniform(0.5, 0.95, size=len(same_df))
    report = det.score(same_df, proba=proba_now, window_label="proba_shift")
    assert report.proba_psi_flag, f"expected PSI flag, got psi={report.proba_psi:.3f}"
    assert report.drift_fired


def test_detector_report_serializable() -> None:
    ref_df, same_df = _ref_and_window()
    ref = fit_reference(ref_df, feature_columns=list(ref_df.columns))
    det = DriftDetector(ref)
    report = det.score(same_df, proba=None, window_label="x")
    d = report.to_dict()
    assert "drift_fired" in d
    assert "feature_ks_stats" in d
    assert isinstance(d["feature_ks_stats"], dict)
