"""Drift detector — per-feature KS-test + PSI on predicted-probability bins.

Day 3 Phase 2b (2026-05-20). The drift surface is the behavioral features
engineered in Day 2 + the model probabilities served by the temporal-split
model from Day 1. Two complementary signals are computed against a frozen
reference distribution (captured at training time):

1. **Per-feature KS-test** — for every continuous feature, the
   two-sample Kolmogorov-Smirnov statistic between the reference sample and
   the current monitoring window. A small p-value means the cumulative
   distributions differ. We additionally require the KS statistic itself to
   exceed ``ks_stat_threshold`` so that microscopic shifts at large N do not
   trip the detector — pure p-value thresholds are too noisy when each daily
   window has 50K+ rows.

2. **PSI on predicted probability** — the Population Stability Index over a
   fixed 10-bucket discretisation of model output probability. PSI captures
   prediction-distribution drift even when individual features are stable
   (e.g., when correlations among features shift in ways no single KS would
   notice). The conventional thresholds are PSI < 0.1 (stable), 0.1–0.25
   (moderate), ≥ 0.25 (significant) — we default to ``psi_threshold=0.25``
   for the auto-retrain trigger.

The reference distribution is saved as an artifact (numeric per-feature
reference values + the production model's predicted-probability histogram on
the held-out window) so a serving deployment can stream new transactions in
and score them without re-loading training data.

Why both KS and PSI, not just one?
    KS catches single-feature distribution shift. PSI catches model-output
    drift (which is what users actually see). A change can fire either alone
    — e.g., adversarial fraud patterns can move predicted probabilities even
    when no single feature's marginal moves much, and a benign data-pipeline
    bug (clock reset, sensor recalibration) can move one feature without
    moving probabilities. The trigger ORs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DEFAULT_PSI_BINS = 10
PSI_EPSILON = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = DEFAULT_PSI_BINS) -> float:
    """Population Stability Index between two 1-D samples.

    PSI = sum_i ( p_curr_i - p_ref_i ) * log( p_curr_i / p_ref_i ).

    Bin edges are taken from ``reference`` (quantile bins) so the PSI is
    measured against the training distribution's natural quantile shape.

    Parameters
    ----------
    reference, current : np.ndarray
        1-D numeric samples.
    bins : int
        Number of quantile bins on the reference. Default 10.

    Returns
    -------
    float
        PSI value. >= 0; 0 means identical distributions.
    """
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    # Collapse duplicate edges (heavy-tailed reference distributions); always
    # bracket -inf / +inf so out-of-range current values are still counted.
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_p = (ref_counts + PSI_EPSILON) / (ref_counts.sum() + PSI_EPSILON * edges.size)
    cur_p = (cur_counts + PSI_EPSILON) / (cur_counts.sum() + PSI_EPSILON * edges.size)

    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


@dataclass
class DriftReport:
    """Per-window drift result. One per scored daily window."""

    window_label: str
    n_rows: int
    feature_ks_stats: dict[str, float] = field(default_factory=dict)
    feature_ks_pvalues: dict[str, float] = field(default_factory=dict)
    flagged_features: list[str] = field(default_factory=list)
    proba_psi: float = 0.0
    proba_psi_flag: bool = False
    drift_fired: bool = False
    fire_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fit_reference(
    reference_df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    proba: np.ndarray | None = None,
    max_reference_sample: int = 20_000,
    rng_seed: int = 42,
) -> dict:
    """Build the frozen reference snapshot the detector compares against.

    The KS test is O(n log n) and dominated by sort cost; a 20K sub-sample
    of the reference is more than enough for KS at any reasonable alpha
    (the test power saturates well before 5K). The predicted-probability
    histogram is stored as raw values so PSI can rebin against any
    monitoring window.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-time feature dataframe (the X_train surface).
    feature_columns : iterable[str]
        Columns to monitor. Constant / binary columns will simply produce a
        KS statistic of 0 against a similar window.
    proba : np.ndarray | None
        Predicted probabilities on the reference window. If omitted, only
        feature drift will be available (PSI on probabilities is skipped).
    max_reference_sample : int
        Cap on the per-feature reference sample size.

    Returns
    -------
    dict
        Serialisable reference snapshot.
    """
    rng = np.random.default_rng(rng_seed)
    feature_columns = list(feature_columns)

    sample_idx = (
        rng.choice(len(reference_df), size=max_reference_sample, replace=False)
        if len(reference_df) > max_reference_sample
        else np.arange(len(reference_df))
    )

    feature_samples: dict[str, list[float]] = {}
    for col in feature_columns:
        if col not in reference_df.columns:
            continue
        s = pd.to_numeric(reference_df[col].iloc[sample_idx], errors="coerce").dropna()
        feature_samples[col] = s.astype("float64").tolist()

    proba_sample: list[float] = []
    if proba is not None:
        p = np.asarray(proba, dtype="float64")
        p = p[np.isfinite(p)]
        if p.size > max_reference_sample:
            p = rng.choice(p, size=max_reference_sample, replace=False)
        proba_sample = p.tolist()

    return {
        "feature_columns": feature_columns,
        "feature_samples": feature_samples,
        "proba_sample": proba_sample,
        "max_reference_sample": max_reference_sample,
        "rng_seed": rng_seed,
    }


class DriftDetector:
    """Stream-friendly drift detector.

    Usage
    -----
    >>> ref = fit_reference(X_train_df, FEATURE_COLS, proba=train_proba)
    >>> det = DriftDetector(ref, ks_pvalue_threshold=0.01,
    ...                     ks_stat_threshold=0.15, psi_threshold=0.25)
    >>> report = det.score(X_day_df, proba_day, window_label="2026-05-20")
    >>> report.drift_fired
    False
    """

    def __init__(
        self,
        reference: dict,
        *,
        ks_pvalue_threshold: float = 0.01,
        ks_stat_threshold: float = 0.15,
        psi_threshold: float = 0.25,
        min_flag_features: int = 1,
        monitored_features: Iterable[str] | None = None,
    ) -> None:
        """Initialise the detector against a saved reference snapshot.

        Parameters
        ----------
        reference : dict
            Output of ``fit_reference``.
        ks_pvalue_threshold : float
            KS test p-value below which a feature is "shifted." Stand-alone
            p-value cutoffs over-fire on large windows; we also require
            ``ks_stat_threshold``.
        ks_stat_threshold : float
            Minimum KS statistic magnitude to count a feature as shifted.
            0.15 corresponds to a clearly visible CDF gap.
        psi_threshold : float
            PSI value above which predicted-probability drift fires. 0.25 is
            the conventional "significant drift" cutoff.
        min_flag_features : int
            Minimum count of shifted features to fire on feature drift alone.
            Default 1 — any single shifted feature is enough.
        monitored_features : iterable[str] | None
            Restrict scoring to this subset of features. Useful to skip
            constant / binary one-hot columns where KS is uninformative.
        """
        self.reference = reference
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.ks_stat_threshold = ks_stat_threshold
        self.psi_threshold = psi_threshold
        self.min_flag_features = min_flag_features

        all_feats = list(reference.get("feature_samples", {}).keys())
        self.monitored_features = (
            list(monitored_features)
            if monitored_features is not None
            else all_feats
        )

    # ----- I/O -------------------------------------------------------------

    @classmethod
    def from_artifact(cls, path: str | Path, **kwargs) -> "DriftDetector":
        """Load a saved reference JSON and build a detector from it."""
        with open(path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        return cls(ref, **kwargs)

    @staticmethod
    def save_reference(reference: dict, path: str | Path) -> None:
        """Persist a reference snapshot to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reference, f)

    # ----- Scoring ---------------------------------------------------------

    def score(
        self,
        window_df: pd.DataFrame,
        proba: np.ndarray | None = None,
        *,
        window_label: str = "",
    ) -> DriftReport:
        """Score a monitoring window against the reference snapshot."""
        ref_samples: Mapping[str, list[float]] = self.reference.get("feature_samples", {})
        report = DriftReport(window_label=window_label, n_rows=int(len(window_df)))

        for col in self.monitored_features:
            if col not in ref_samples or col not in window_df.columns:
                continue
            ref = np.asarray(ref_samples[col], dtype="float64")
            cur = pd.to_numeric(window_df[col], errors="coerce").dropna().to_numpy()
            if ref.size < 30 or cur.size < 30:
                continue
            stat, p = ks_2samp(ref, cur, alternative="two-sided", method="auto")
            report.feature_ks_stats[col] = float(stat)
            report.feature_ks_pvalues[col] = float(p)
            if p < self.ks_pvalue_threshold and stat >= self.ks_stat_threshold:
                report.flagged_features.append(col)

        ref_proba = np.asarray(self.reference.get("proba_sample", []), dtype="float64")
        if proba is not None and ref_proba.size > 0:
            report.proba_psi = psi(ref_proba, np.asarray(proba, dtype="float64"))
            report.proba_psi_flag = report.proba_psi >= self.psi_threshold

        if report.proba_psi_flag and len(report.flagged_features) >= self.min_flag_features:
            report.drift_fired = True
            report.fire_reason = (
                f"feature_ks (n={len(report.flagged_features)}) + proba_psi={report.proba_psi:.3f}"
            )
        elif report.proba_psi_flag:
            report.drift_fired = True
            report.fire_reason = f"proba_psi={report.proba_psi:.3f} >= {self.psi_threshold}"
        elif len(report.flagged_features) >= self.min_flag_features:
            report.drift_fired = True
            report.fire_reason = f"feature_ks shifted: {report.flagged_features[:5]}"
        else:
            report.drift_fired = False
            report.fire_reason = ""

        return report


__all__ = [
    "DriftDetector",
    "DriftReport",
    "fit_reference",
    "psi",
]
