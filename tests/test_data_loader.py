"""DVC-aware loader basics — Day 4 Phase 3."""

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


def test_loader_reads_x_test_and_y_test() -> None:
    loader = SentinelDataLoader()
    X = loader.load_x_test()
    y = loader.load_y_test()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert y.name == "is_fraud"
    assert set(y.unique()).issubset({0, 1})


def test_dvc_status_is_safe_without_dvc(tmp_path: Path) -> None:
    # Function never raises; reports availability bool.
    status = dvc_status(tmp_path)
    assert "dvc_available" in status
