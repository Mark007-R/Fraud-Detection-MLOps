"""DVC-aware loader basics -- Day 4 Phase 3, Day-7 hardened.

The original ``test_loader_reads_x_test_and_y_test`` read the real
``data/processed/X_test.csv`` + ``y_test.csv`` from disk, which are
DVC-tracked and not present on a fresh CI runner. The Day-7 rewrite points
the loader at synthetic CSVs in a tmp_path so the test is now hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import LoaderConfig, SentinelDataLoader, dvc_status


def test_loader_resolves_paths_from_params() -> None:
    cfg = LoaderConfig.from_params()
    paths = cfg.resolve()
    assert paths.features.name == "features.csv"
    assert paths.x_test.name == "X_test.csv"
    assert paths.model.name == "fraud_model.pkl"


def test_loader_reads_x_test_and_y_test(tmp_path: Path) -> None:
    """Read synthetic X_test / y_test from a tmp_path-rooted LoaderConfig."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "hour_of_day": [9, 14, 22, 3],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="is_fraud")
    X.to_csv(processed / "X_test.csv", index=False)
    y.to_csv(processed / "y_test.csv", index=False)

    cfg = LoaderConfig(project_root=tmp_path, use_dvc_pull=False)
    loader = SentinelDataLoader(cfg)
    X_loaded = loader.load_x_test()
    y_loaded = loader.load_y_test()

    assert isinstance(X_loaded, pd.DataFrame)
    assert isinstance(y_loaded, pd.Series)
    assert len(X_loaded) == len(y_loaded) == 4
    assert y_loaded.name == "is_fraud"
    assert set(y_loaded.unique()).issubset({0, 1})
    assert list(X_loaded.columns) == ["amount", "hour_of_day"]


def test_dvc_status_is_safe_without_dvc(tmp_path: Path) -> None:
    # Function never raises; reports availability bool.
    status = dvc_status(tmp_path)
    assert "dvc_available" in status
