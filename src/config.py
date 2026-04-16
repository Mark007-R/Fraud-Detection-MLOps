"""Shared configuration utilities for SENTINEL pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root is the parent of the src/ directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from YAML.

    Resolves relative paths against the project root so the function works
    regardless of the caller's current working directory.

    Parameters
    ----------
    path : str | Path
        Path to params file. If relative, resolved against project root.

    Returns
    -------
    dict[str, Any]
        Parsed parameters dictionary.
    """
    params_path = Path(path)
    if not params_path.is_absolute():
        params_path = PROJECT_ROOT / params_path
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
