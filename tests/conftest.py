"""Pytest configuration -- shared markers for the Sentinel test suite.

Day 7 Phase 6 (2026-05-24). The CI workflow at .github/workflows/ci.yml runs
`pytest tests/ -m "not requires_data"` so the GitHub runner skips tests that
need the DVC-tracked data and trained model artifacts (which are not in git).
This conftest registers the `requires_data` marker so pytest does not emit a
PytestUnknownMarkWarning when the marker is used.
"""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_data: test depends on DVC-tracked data and/or the trained "
        "model artifact in models/. Skipped in CI; run locally after "
        "`dvc repro train`.",
    )
