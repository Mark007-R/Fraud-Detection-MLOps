"""DVC-aware data loader — Day 4 Phase 3 production refactor.

This module is the single entry point for any other Sentinel module that
needs to read a tracked dataset. It centralises three things that used to be
duplicated across train.py / preprocess.py / benchmark_fdb.py:

1. **Path resolution.** Paths come from ``params.yaml`` -> ``data.*`` and are
   resolved against ``PROJECT_ROOT`` so callers do not need to manage cwd.

2. **DVC awareness.** Before loading any tracked file, the loader checks
   whether DVC reports it as missing or stale. If DVC is available and the
   file is gone (cache cleared, fresh checkout), the loader runs
   ``dvc pull <path>`` for that file specifically. If DVC is not installed
   or the path is not tracked, the loader degrades to a plain read and the
   missing-file FileNotFoundError surfaces as usual.

3. **Pydantic-typed config.** ``LoaderConfig`` validates the params.yaml
   shape so a malformed file fails fast at startup rather than at the first
   pd.read_csv call.

The serving layer (``src/serving/api.py``) and the auto-retrain trigger
(``src/drift/trigger.py``) both depend on this — so the DVC fallback is the
difference between a clean checkout being one ``docker compose up`` away and
needing a manual ``dvc pull`` first.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import PROJECT_ROOT, load_params


class DataPaths(BaseModel):
    """Resolved on-disk locations of every Sentinel tracked artifact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_paysim: Path
    raw_sparkov: Path
    raw_sparkov_train: Path | None
    raw_sparkov_test: Path | None
    combined: Path
    features: Path
    x_test: Path
    y_test: Path
    model: Path
    feature_thresholds: Path
    metrics: Path
    benchmark: Path


class LoaderConfig(BaseModel):
    """Validated view of the ``data:`` block from params.yaml."""

    # `model_path` collides with Pydantic's reserved ``model_`` namespace; opt out.
    model_config = ConfigDict(protected_namespaces=())

    raw_paysim_path: str = "data/raw/paysim.csv"
    raw_sparkov_path: str = "data/raw/sparkov.csv"
    raw_sparkov_train_path: str = "data/raw/sparkov_train.csv"
    raw_sparkov_test_path: str = "data/raw/sparkov_test.csv"
    combined_path: str = "data/processed/combined_transactions.csv"
    features_path: str = "data/processed/features.csv"
    x_test_path: str = "data/processed/X_test.csv"
    y_test_path: str = "data/processed/y_test.csv"
    model_path: str = "models/fraud_model.pkl"
    feature_thresholds_path: str = "data/processed/feature_thresholds.json"
    metrics_path: str = "metrics/scores.json"
    benchmark_path: str = "metrics/fdb_benchmark.json"
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    use_dvc_pull: bool = True

    @field_validator("project_root")
    @classmethod
    def _root_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"project_root '{v}' does not exist")
        return v

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> "LoaderConfig":
        """Build a LoaderConfig from the parsed params.yaml ``data:`` block."""
        params = params if params is not None else load_params()
        data_block = dict(params.get("data", {}))
        return cls(**{k: v for k, v in data_block.items() if k in cls.model_fields})

    def resolve(self) -> DataPaths:
        """Return a DataPaths with all paths anchored to project_root."""
        root = self.project_root

        def _opt(p: str) -> Path | None:
            path = (root / p).resolve()
            return path

        return DataPaths(
            raw_paysim=(root / self.raw_paysim_path).resolve(),
            raw_sparkov=(root / self.raw_sparkov_path).resolve(),
            raw_sparkov_train=_opt(self.raw_sparkov_train_path),
            raw_sparkov_test=_opt(self.raw_sparkov_test_path),
            combined=(root / self.combined_path).resolve(),
            features=(root / self.features_path).resolve(),
            x_test=(root / self.x_test_path).resolve(),
            y_test=(root / self.y_test_path).resolve(),
            model=(root / self.model_path).resolve(),
            feature_thresholds=(root / self.feature_thresholds_path).resolve(),
            metrics=(root / self.metrics_path).resolve(),
            benchmark=(root / self.benchmark_path).resolve(),
        )


def dvc_status(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Best-effort ``dvc status`` summary. Returns {} when DVC is unavailable."""
    if shutil.which("dvc") is None:
        return {"dvc_available": False}
    try:
        out = subprocess.run(
            ["dvc", "status", "--json"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0:
            return {"dvc_available": True, "error": out.stderr.strip()}
        return {"dvc_available": True, "stdout": out.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"dvc_available": True, "error": "dvc_status_timeout"}


class SentinelDataLoader:
    """Read tracked datasets, optionally healing missing files via ``dvc pull``.

    Usage
    -----
    >>> loader = SentinelDataLoader()
    >>> X_test = loader.load_x_test()
    >>> y_test = loader.load_y_test()
    >>> raw    = loader.load_raw("sparkov_test")
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig.from_params()
        self.paths = self.config.resolve()

    # ----- Internal helpers ------------------------------------------------

    def _ensure_present(self, path: Path) -> Path:
        """If ``path`` is missing and DVC can fetch it, attempt ``dvc pull``."""
        if path.exists():
            return path
        if not self.config.use_dvc_pull or shutil.which("dvc") is None:
            raise FileNotFoundError(f"[Loader] Missing tracked file: {path}")
        rel = path.relative_to(self.config.project_root)
        print(f"[Loader] {rel} missing — attempting `dvc pull {rel}`")
        out = subprocess.run(
            ["dvc", "pull", str(rel)],
            cwd=str(self.config.project_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode != 0:
            raise FileNotFoundError(
                f"[Loader] dvc pull failed for {rel}: {out.stderr.strip()}"
            )
        if not path.exists():
            raise FileNotFoundError(
                f"[Loader] dvc pull reported success but {rel} still missing"
            )
        return path

    # ----- Tracked-artifact accessors --------------------------------------

    def load_features(self) -> pd.DataFrame:
        return pd.read_csv(self._ensure_present(self.paths.features))

    def load_x_test(self) -> pd.DataFrame:
        return pd.read_csv(self._ensure_present(self.paths.x_test))

    def load_y_test(self) -> pd.Series:
        df = pd.read_csv(self._ensure_present(self.paths.y_test))
        # y_test is saved as a single-column dataframe; recover the Series.
        return df.iloc[:, 0].astype(int).rename("is_fraud")

    def load_raw(self, key: str) -> pd.DataFrame:
        """Load a raw dataset by short key: 'paysim' | 'sparkov' | 'sparkov_test' | 'sparkov_train'."""
        mapping = {
            "paysim": self.paths.raw_paysim,
            "sparkov": self.paths.raw_sparkov,
            "sparkov_train": self.paths.raw_sparkov_train,
            "sparkov_test": self.paths.raw_sparkov_test,
        }
        if key not in mapping or mapping[key] is None:
            raise KeyError(f"[Loader] Unknown raw key '{key}'. Choose from {list(mapping)}")
        return pd.read_csv(self._ensure_present(mapping[key]))

    def model_path(self) -> Path:
        """Return the production model path (does NOT run dvc pull — models are pickled jobs)."""
        return self.paths.model


__all__ = [
    "DataPaths",
    "LoaderConfig",
    "SentinelDataLoader",
    "dvc_status",
]
