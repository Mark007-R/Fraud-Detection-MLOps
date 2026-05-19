"""Behavioral feature engineering — Dask and Pandas implementations.

Day 2 Phase 2a (2026-05-19). This module adds the behavioral features that the
Day 1 pipeline never had (per-card spending velocity, amount z-scores by card,
time-of-day bucketing, distance-to-home location signals). It exposes the SAME
logic in two forms — Dask DataFrame and Pandas DataFrame — so the throughput
benchmark in ``src/features/benchmark.py`` can compare them on identical inputs.

Both implementations operate on the raw Sparkov schema (``cc_num``, ``amt``,
``lat``/``long``, ``merch_lat``/``merch_long``, ``trans_date_trans_time``,
``is_fraud``) — these are the columns the combine stage in
``src/combine_datasets.py`` strips, and they are exactly the fields that
behavioral features need.

The output feature matrix is identical between the two backends — that is
asserted by ``tests/test_features_determinism.py``.
"""

from __future__ import annotations

import math
from typing import Iterable

import dask.dataframe as dd
import numpy as np
import pandas as pd


BEHAVIORAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "amount",
    "tx_amount_log",
    "hour_of_day",
    "is_night",
    "is_weekend",
    "day_of_month",
    "card_txn_count",
    "card_amount_mean",
    "card_amount_std",
    "card_amount_zscore",
    "card_lat_mean",
    "card_long_mean",
    "distance_to_home_km",
    "card_merch_lat_std",
    "card_merch_long_std",
    "tod_bucket_morning",
    "tod_bucket_afternoon",
    "tod_bucket_evening",
    "tod_bucket_night",
)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres.

    Works elementwise on numpy/pandas/dask Series — uses numpy ufuncs only.
    """
    earth_radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return earth_radius_km * c


def _tod_bucket_labels(hour: pd.Series) -> pd.DataFrame:
    """Time-of-day one-hot: morning(6-12), afternoon(12-18), evening(18-22), night(22-6)."""
    h = hour.astype("int64")
    morning = ((h >= 6) & (h < 12)).astype("int64")
    afternoon = ((h >= 12) & (h < 18)).astype("int64")
    evening = ((h >= 18) & (h < 22)).astype("int64")
    night = ((h >= 22) | (h < 6)).astype("int64")
    return pd.DataFrame(
        {
            "tod_bucket_morning": morning.to_numpy(),
            "tod_bucket_afternoon": afternoon.to_numpy(),
            "tod_bucket_evening": evening.to_numpy(),
            "tod_bucket_night": night.to_numpy(),
        },
        index=hour.index,
    )


# ----------------------------------------------------------------------------
# Pandas implementation (single-node baseline).
# ----------------------------------------------------------------------------

def engineer_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioral features on a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Sparkov rows containing ``cc_num``, ``amt``, ``lat``, ``long``,
        ``merch_lat``, ``merch_long``, ``trans_date_trans_time``, ``is_fraud``.

    Returns
    -------
    pd.DataFrame
        Engineered feature dataframe with ``BEHAVIORAL_FEATURE_COLUMNS`` plus
        ``is_fraud``.
    """
    out = pd.DataFrame(index=df.index)
    amount = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)
    out["amount"] = amount
    out["tx_amount_log"] = np.log1p(amount.clip(lower=0))

    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype("int64")
    dow = dt.dt.dayofweek.fillna(0).astype("int64")
    out["hour_of_day"] = hour
    out["day_of_month"] = dt.dt.day.fillna(1).astype("int64")
    out["is_night"] = (((hour >= 0) & (hour < 6)) | (hour >= 22)).astype("int64")
    out["is_weekend"] = (dow >= 5).astype("int64")

    # Per-card aggregates — broadcast back via transform / groupby+merge.
    card = df["cc_num"]
    grp = amount.groupby(card)
    out["card_txn_count"] = grp.transform("count").astype("int64").to_numpy()
    out["card_amount_mean"] = grp.transform("mean").to_numpy()
    out["card_amount_std"] = grp.transform("std").fillna(0.0).to_numpy()
    out["card_amount_zscore"] = (
        (amount - out["card_amount_mean"]) / (out["card_amount_std"] + 1e-8)
    ).to_numpy()

    lat = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
    lon = pd.to_numeric(df["long"], errors="coerce").fillna(0.0)
    mlat = pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0.0)
    mlon = pd.to_numeric(df["merch_long"], errors="coerce").fillna(0.0)

    out["card_lat_mean"] = lat.groupby(card).transform("mean").to_numpy()
    out["card_long_mean"] = lon.groupby(card).transform("mean").to_numpy()
    out["distance_to_home_km"] = _haversine_km(lat, lon, mlat, mlon).to_numpy()
    out["card_merch_lat_std"] = mlat.groupby(card).transform("std").fillna(0.0).to_numpy()
    out["card_merch_long_std"] = mlon.groupby(card).transform("std").fillna(0.0).to_numpy()

    tod = _tod_bucket_labels(hour)
    for col in tod.columns:
        out[col] = tod[col].to_numpy()

    out["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype("int64").to_numpy()
    return out[list(BEHAVIORAL_FEATURE_COLUMNS) + ["is_fraud"]]


# ----------------------------------------------------------------------------
# Dask implementation (distributed).
# ----------------------------------------------------------------------------

def engineer_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """Compute the same behavioral features on a Dask DataFrame.

    The per-card aggregates are computed once with ``groupby().agg(...)`` and
    then merged back to the partitioned frame — this is the canonical Dask
    pattern for "broadcast a small per-key table back to a large frame".

    Parameters
    ----------
    ddf : dd.DataFrame
        Dask frame partitioned over the raw Sparkov rows.

    Returns
    -------
    dd.DataFrame
        Lazy feature frame. Caller must call ``.compute()``.
    """
    ddf = ddf.assign(
        amt=dd.to_numeric(ddf["amt"], errors="coerce").fillna(0.0),
        lat=dd.to_numeric(ddf["lat"], errors="coerce").fillna(0.0),
        long=dd.to_numeric(ddf["long"], errors="coerce").fillna(0.0),
        merch_lat=dd.to_numeric(ddf["merch_lat"], errors="coerce").fillna(0.0),
        merch_long=dd.to_numeric(ddf["merch_long"], errors="coerce").fillna(0.0),
        is_fraud=dd.to_numeric(ddf["is_fraud"], errors="coerce").fillna(0).astype("int64"),
    )

    # Parse timestamps once with map_partitions to land hour/dow/day.
    def _time_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(pdf["trans_date_trans_time"], errors="coerce")
        return pd.DataFrame(
            {
                "hour_of_day": dt.dt.hour.fillna(0).astype("int64"),
                "day_of_week": dt.dt.dayofweek.fillna(0).astype("int64"),
                "day_of_month": dt.dt.day.fillna(1).astype("int64"),
            },
            index=pdf.index,
        )

    time_meta = pd.DataFrame(
        {
            "hour_of_day": pd.Series(dtype="int64"),
            "day_of_week": pd.Series(dtype="int64"),
            "day_of_month": pd.Series(dtype="int64"),
        }
    )
    time_df = ddf.map_partitions(_time_partition, meta=time_meta)
    ddf = ddf.assign(
        hour_of_day=time_df["hour_of_day"],
        day_of_week=time_df["day_of_week"],
        day_of_month=time_df["day_of_month"],
    )

    # Per-card aggregates — compute the small per-card table eagerly with the
    # distributed scheduler, then broadcast it through ``map_partitions`` for
    # the join. Dask 2026.x's expression planner trips a KeyError when merging
    # a named-agg result back onto the source frame, but a partition-level
    # pandas merge against a small materialised lookup avoids that path.
    agg_pdf = ddf.groupby("cc_num").agg(
        card_txn_count=("amt", "count"),
        card_amount_mean=("amt", "mean"),
        card_amount_std=("amt", "std"),
        card_lat_mean=("lat", "mean"),
        card_long_mean=("long", "mean"),
        card_merch_lat_std=("merch_lat", "std"),
        card_merch_long_std=("merch_long", "std"),
    ).compute().reset_index()

    merge_cols = [
        "card_txn_count",
        "card_amount_mean",
        "card_amount_std",
        "card_lat_mean",
        "card_long_mean",
        "card_merch_lat_std",
        "card_merch_long_std",
    ]

    def _broadcast_agg(pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.merge(agg_pdf, on="cc_num", how="left")

    # Build meta by appending the broadcast columns directly to the source meta
    # (the partition merge preserves source-column ordering, but pandas.merge on
    # an empty meta moves the join key, breaking Dask's column-order check).
    merged_meta = ddf._meta.copy()
    for col in merge_cols:
        merged_meta[col] = pd.Series(dtype=agg_pdf[col].dtype)
    merged = ddf.map_partitions(_broadcast_agg, meta=merged_meta)

    def _finalize(pdf: pd.DataFrame) -> pd.DataFrame:
        amount = pdf["amt"].fillna(0.0)
        std = pdf["card_amount_std"].fillna(0.0)
        out = pd.DataFrame(index=pdf.index)
        out["amount"] = amount
        out["tx_amount_log"] = np.log1p(amount.clip(lower=0))
        out["hour_of_day"] = pdf["hour_of_day"].astype("int64")
        h = out["hour_of_day"]
        out["is_night"] = (((h >= 0) & (h < 6)) | (h >= 22)).astype("int64")
        out["is_weekend"] = (pdf["day_of_week"] >= 5).astype("int64")
        out["day_of_month"] = pdf["day_of_month"].astype("int64")
        out["card_txn_count"] = pdf["card_txn_count"].fillna(0).astype("int64")
        out["card_amount_mean"] = pdf["card_amount_mean"].fillna(0.0)
        out["card_amount_std"] = std
        out["card_amount_zscore"] = (amount - pdf["card_amount_mean"].fillna(0.0)) / (std + 1e-8)
        out["card_lat_mean"] = pdf["card_lat_mean"].fillna(0.0)
        out["card_long_mean"] = pdf["card_long_mean"].fillna(0.0)
        out["distance_to_home_km"] = _haversine_km(
            pdf["lat"], pdf["long"], pdf["merch_lat"], pdf["merch_long"]
        )
        out["card_merch_lat_std"] = pdf["card_merch_lat_std"].fillna(0.0)
        out["card_merch_long_std"] = pdf["card_merch_long_std"].fillna(0.0)
        tod = _tod_bucket_labels(h)
        for col in tod.columns:
            out[col] = tod[col].to_numpy()
        out["is_fraud"] = pdf["is_fraud"].astype("int64")
        return out[list(BEHAVIORAL_FEATURE_COLUMNS) + ["is_fraud"]]

    meta_dict: dict[str, str] = {col: "float64" for col in BEHAVIORAL_FEATURE_COLUMNS}
    for int_col in (
        "hour_of_day",
        "is_night",
        "is_weekend",
        "day_of_month",
        "card_txn_count",
        "tod_bucket_morning",
        "tod_bucket_afternoon",
        "tod_bucket_evening",
        "tod_bucket_night",
    ):
        meta_dict[int_col] = "int64"
    meta_dict["is_fraud"] = "int64"
    meta_df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in meta_dict.items()})

    return merged.map_partitions(_finalize, meta=meta_df)


__all__ = [
    "BEHAVIORAL_FEATURE_COLUMNS",
    "engineer_pandas",
    "engineer_dask",
]
