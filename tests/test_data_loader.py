"""
Tests for src.data_loader.

Checks parser output against a small synthetic file: shapes, NaN handling for
time-varying membership, sorted dates and float dtypes.
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import parse_index_file


@pytest.fixture()
def synthetic_file(tmp_path: Path) -> Path:
    """A 3-day toy index where constituent membership changes over time."""
    raw = {
        "2009-01-02T00:00:00Z": {"return": 0.01, "ts": {"1": 0.02, "2": -0.01}},
        "2009-01-05T00:00:00Z": {"return": -0.005, "ts": {"1": 0.00, "3": 0.04}},
        "2009-01-06T00:00:00Z": {"return": 0.008, "ts": {"2": 0.01, "3": -0.02}},
    }
    p = tmp_path / "TOY_json.bz2"
    with bz2.open(p, "wt", encoding="utf-8") as fh:
        json.dump(raw, fh)
    return p


def test_index_series_shape_and_order(synthetic_file: Path) -> None:
    idx, _ = parse_index_file(synthetic_file)
    assert len(idx) == 3
    assert idx.index.is_monotonic_increasing      # dates sorted (index, not values)
    assert idx.dtype == np.float64
    assert idx.iloc[0] == pytest.approx(0.01)


def test_constituent_panel_membership_nan(synthetic_file: Path) -> None:
    _, panel = parse_index_file(synthetic_file)
    # Union of ids across all dates -> 3 columns.
    assert set(panel.columns) == {"1", "2", "3"}
    assert panel.shape == (3, 3)
    # Constituent "3" was absent on day 1 -> must be NaN.
    assert pd.isna(panel.loc["2009-01-02", "3"])
    # Constituent "1" was absent on day 3 -> must be NaN.
    assert pd.isna(panel.loc["2009-01-06", "1"])
    # A present value is preserved exactly.
    assert panel.loc["2009-01-05", "3"] == pytest.approx(0.04)


def test_index_and_panel_share_dates(synthetic_file: Path) -> None:
    idx, panel = parse_index_file(synthetic_file)
    assert list(idx.index) == list(panel.index)
