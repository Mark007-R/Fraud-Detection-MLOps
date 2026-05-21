"""Training package — Day 4 module layout.

Re-exports ``temporal_split_per_source`` and the ``main`` entry point from
the long-standing ``src/train.py`` so callers can import via either path.
The DVC pipeline still wires ``src/train.py`` directly so the Day-1 split
fix stays binding.
"""

from src.training.train import (
    TrainConfig,
    main,
    temporal_split_per_source,
    train_xgboost,
)

__all__ = [
    "TrainConfig",
    "main",
    "temporal_split_per_source",
    "train_xgboost",
]
