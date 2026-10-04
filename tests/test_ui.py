"""Streamlit UI smoke tests: every page renders, and the batch helpers behave.

Runs without the DVC data or the model artifact — pages that need the model
stop with a notice instead of raising, which is what CI exercises. With
models/fraud_model.pkl present locally the scoring paths run as well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_kit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = {  # page -> text from its header
    "views/overview.py": "keeps itself honest",
    "views/score.py": "Score a <em>transaction</em>",
    "views/screen.py": "Screen a <em>batch</em>",
    "views/model.py": "How the model <em>performs</em>",
    "views/ops.py": "The loop, <em>measured</em>",
}


@pytest.mark.parametrize("page", list(PAGES))
def test_page_renders_without_exception(page: str) -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    if page != "views/overview.py":
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(PAGES[page] in m.value for m in at.markdown)


def test_demo_batch_is_seeded_and_in_schema() -> None:
    a, b = ui_kit.demo_batch(), ui_kit.demo_batch()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ui_kit.RAW_COLUMNS
    assert set(a["source"]) == {"paysim", "sparkov"}
    card = a[a["source"] == "sparkov"]
    assert card["balance_change_orig"].isna().all() and (card["has_balance_info"] == 0).all()
    assert set(card["transaction_type"]) <= set(ui_kit.SPARKOV_CATEGORIES)


def test_to_row_matches_combine_datasets_conventions() -> None:
    pay = ui_kit.to_row("paysim", "TRANSFER", 500.0, 2, 14, before=500.0, after=0.0).iloc[0]
    assert pay["balance_change_orig"] == 500.0
    assert pay["balance_ratio"] == pytest.approx(0.0)
    assert pay["has_balance_info"] == 1
    card = ui_kit.to_row("sparkov", "grocery_pos", 64.2, 18, 12).iloc[0]
    assert np.isnan(card["balance_change_orig"]) and card["has_balance_info"] == 0


def test_normalise_batch_fills_defaults_and_infers_source() -> None:
    raw = pd.DataFrame({"Amount": ["120.5", "x", "99"], "transaction_type": ["PAYMENT", "PAYMENT", "travel"]})
    out = ui_kit.normalise_batch(raw)
    assert len(out) == 2  # the non-numeric amount is dropped
    assert list(out["source"]) == ["paysim", "sparkov"]
    assert set(ui_kit.RAW_COLUMNS) <= set(out.columns)


def test_normalise_batch_rejects_files_without_amount() -> None:
    with pytest.raises(ValueError):
        ui_kit.normalise_batch(pd.DataFrame({"foo": [1]}))


def test_pct_never_rounds_to_the_extremes() -> None:
    assert ui_kit.pct(0.00004) == "0.00"
    assert ui_kit.pct(0.0012) == "0.12"
    assert ui_kit.pct(0.4567) == "45.7"
    assert ui_kit.pct(0.99934) == "99.93"
