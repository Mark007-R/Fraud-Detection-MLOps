"""Pandas == Dask determinism on the behavioral feature engineer.

Day 7 Phase 6 (2026-05-24). Day-2 claimed the two backends produce bit-exact
outputs (max abs diff 5.5e-12). This test is the regression gate -- if either
implementation drifts numerically, CI fails before the user sees a model
flipped onto a slightly-different feature space.

Uses an in-memory synthetic Sparkov-shaped frame so the test runs in <2s and
has no data dependency.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.features.engineer import (
    BEHAVIORAL_FEATURE_COLUMNS,
    engineer_dask,
    engineer_pandas,
)


def _synthetic_frame(n_rows: int = 1_000, n_cards: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cc_num = rng.integers(low=4_000_000_000_000_000, high=4_999_999_999_999_999, size=n_cards)
    cards = rng.choice(cc_num, size=n_rows)
    start = pd.Timestamp("2024-01-01T00:00:00")
    offsets = pd.to_timedelta(rng.integers(0, 90 * 24 * 3600, size=n_rows), unit="s")
    home_lats = {c: rng.uniform(25, 49) for c in cc_num}
    home_lons = {c: rng.uniform(-122, -75) for c in cc_num}
    lat = np.array([home_lats[c] for c in cards])
    long = np.array([home_lons[c] for c in cards])
    merch_lat = lat + rng.normal(0, 0.5, size=n_rows)
    merch_long = long + rng.normal(0, 0.5, size=n_rows)
    return pd.DataFrame(
        {
            "cc_num": cards,
            "amt": np.clip(rng.lognormal(3.0, 1.3, size=n_rows), 1.0, 5000.0),
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "trans_date_trans_time": (start + offsets).astype(str),
            "is_fraud": rng.integers(0, 2, size=n_rows),
        }
    )


def test_pandas_and_dask_produce_identical_features() -> None:
    df = _synthetic_frame()
    pandas_out = engineer_pandas(df).reset_index(drop=True)
    ddf = dd.from_pandas(df, npartitions=4)
    dask_out = engineer_dask(ddf).compute().reset_index(drop=True)

    # Compare on the same row ordering. The Dask path joins per-card aggregates
    # via map_partitions, so partition-shuffle should not reorder rows -- but
    # sort both by a stable column hash to make the assertion shuffle-invariant
    # if the Dask planner ever changes.
    sort_key = ["amount", "tx_amount_log", "distance_to_home_km"]
    pandas_sorted = pandas_out.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
    dask_sorted = dask_out.sort_values(sort_key, kind="mergesort").reset_index(drop=True)

    assert list(pandas_sorted.columns) == list(dask_sorted.columns)

    for col in BEHAVIORAL_FEATURE_COLUMNS:
        a = pandas_sorted[col].to_numpy(dtype="float64")
        b = dask_sorted[col].to_numpy(dtype="float64")
        np.testing.assert_allclose(
            a,
            b,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Backend divergence on '{col}'",
        )

    # is_fraud must match exactly (integer label).
    assert (pandas_sorted["is_fraud"].to_numpy() == dask_sorted["is_fraud"].to_numpy()).all()


def test_engineer_pandas_columns_and_shape() -> None:
    df = _synthetic_frame(n_rows=200, n_cards=10, seed=7)
    out = engineer_pandas(df)
    assert len(out) == 200
    assert list(out.columns) == list(BEHAVIORAL_FEATURE_COLUMNS) + ["is_fraud"]
    # tod buckets are one-hot -- exactly one bucket per row.
    bucket_cols = [c for c in out.columns if c.startswith("tod_bucket_")]
    bucket_sums = out[bucket_cols].sum(axis=1)
    assert (bucket_sums == 1).all()
