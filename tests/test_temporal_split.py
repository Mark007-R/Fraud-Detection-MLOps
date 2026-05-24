"""Temporal-split regression guard -- Day 7 Phase 6 (2026-05-24).

The Day-1 audit fixed `src/train.py` to use `temporal_split_per_source`
instead of `train_test_split(stratify=y)`. The failure mode the fix
prevents: a future-dated row appearing in the training set. This test
asserts that invariant directly, on a synthetic two-source frame, so any
future refactor that accidentally re-introduces a random shuffle fails CI
loudly before the model goes anywhere near a registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.train import temporal_split_per_source


def _two_source_frame(n_per_source: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Source A: timestamps 0..n-1
    a = pd.DataFrame(
        {
            "txn_timestamp": np.arange(n_per_source, dtype="int64"),
            "is_fraud": rng.integers(0, 2, size=n_per_source),
            "source_paysim": 1,
            "source_sparkov": 0,
            "feat0": rng.normal(size=n_per_source),
        }
    )
    # Source B: timestamps offset so the two sources interleave in absolute
    # time -- the split must STILL be per-source, not by global timestamp.
    b = pd.DataFrame(
        {
            "txn_timestamp": np.arange(n_per_source, dtype="int64") + 50_000,
            "is_fraud": rng.integers(0, 2, size=n_per_source),
            "source_paysim": 0,
            "source_sparkov": 1,
            "feat0": rng.normal(size=n_per_source),
        }
    )
    # Shuffle rows so the function cannot rely on insertion order.
    out = pd.concat([a, b], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_no_future_timestamp_in_train_per_source() -> None:
    df = _two_source_frame(n_per_source=500)
    train_df, test_df = temporal_split_per_source(df, test_size=0.2)

    for source_col in ("source_paysim", "source_sparkov"):
        tr = train_df.loc[train_df[source_col] == 1, "txn_timestamp"]
        te = test_df.loc[test_df[source_col] == 1, "txn_timestamp"]
        assert len(tr) > 0 and len(te) > 0, f"empty split for {source_col}"
        # The temporal invariant: every train timestamp < every test timestamp.
        assert tr.max() < te.min(), (
            f"future leak in {source_col}: train_max={tr.max()} >= test_min={te.min()}"
        )


def test_test_size_fraction_is_per_source_within_one_row() -> None:
    df = _two_source_frame(n_per_source=500)
    train_df, test_df = temporal_split_per_source(df, test_size=0.2)
    for source_col in ("source_paysim", "source_sparkov"):
        total = int(df[source_col].sum())
        test_count = int((test_df[source_col] == 1).sum())
        expected = int(np.ceil(total * 0.2))
        # Implementation rounds with np.ceil; allow 1-row slack for safety.
        assert abs(test_count - expected) <= 1, (
            f"{source_col}: got {test_count} test rows, expected ~{expected}"
        )


def test_missing_source_columns_raises() -> None:
    bad = pd.DataFrame({"txn_timestamp": [1, 2, 3], "is_fraud": [0, 1, 0]})
    try:
        temporal_split_per_source(bad, test_size=0.2)
    except KeyError:
        return
    raise AssertionError("Expected KeyError on missing source_* columns")


def test_missing_timestamp_column_raises() -> None:
    bad = pd.DataFrame(
        {"is_fraud": [0, 1, 0], "source_paysim": [1, 1, 1], "source_sparkov": [0, 0, 0]}
    )
    try:
        temporal_split_per_source(bad, test_size=0.2)
    except KeyError:
        return
    raise AssertionError("Expected KeyError on missing txn_timestamp")
