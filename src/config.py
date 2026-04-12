"""Shared configuration utilities for SENTINEL pipeline stages."""

from __future__ import annotations

from typing import Any

import yaml


def load_params(path: str = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from YAML.

    Parameters
    ----------
    path : str
        Path to params file.

    Returns
    -------
    dict[str, Any]
        Parsed parameters dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
