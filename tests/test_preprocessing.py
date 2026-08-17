"""
Tests for src.preprocessing.

Focus on membership handling: the point-in-time mask, and that capped
forward-fill bridges only short gaps inside a membership span.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_membership_mask,
    cap_forward_fill,
    winsorize_returns,
    clean_index_series,
)


@pytest.fixture()
def panel() -> pd.DataFrame:
    """Five-day panel: A joins late, B has a one-day gap, C leaves early."""
    dates = pd.to_datetime(
        ["2009-01-02", "2009-01-05", "2009-01-06", "2009-01-07", "2009-01-08"])
    return pd.DataFrame(
        {
            "A": [np.nan, np.nan, 0.01, 0.02, -0.01],   # joins on day 3
            "B": [0.01, np.nan, 0.02, 0.00, 0.01],      # 1-day gap on day 2
            "C": [0.02, -0.01, np.nan, np.nan, np.nan], # leaves after day 2
        },
        index=dates,
    )


def test_membership_mask_is_pointwise(panel: pd.DataFrame) -> None:
    mask = build_membership_mask(panel)
    assert mask.loc["2009-01-02", "A"] == False   # A not yet a member
    assert mask.loc["2009-01-06", "A"] == True    # A joined
    assert mask.loc["2009-01-08", "C"] == False   # C already left
    assert mask.values.dtype == bool


def test_capped_ffill_bridges_only_internal_gaps(panel: pd.DataFrame) -> None:
    mask = build_membership_mask(panel)
    filled, n = cap_forward_fill(panel, mask, max_days=1)
    # B's single internal gap (day 2) is filled from day 1's 0.01.
    assert filled.loc["2009-01-05", "B"] == pytest.approx(0.01)
    # A's leading non-membership must NOT be filled.
    assert pd.isna(filled.loc["2009-01-02", "A"])
    # C's trailing non-membership must NOT be filled.
    assert pd.isna(filled.loc["2009-01-08", "C"])
    assert n == 1  # exactly one cell filled


def test_capped_ffill_respects_cap() -> None:
    dates = pd.to_datetime(
        ["2009-01-02", "2009-01-05", "2009-01-06", "2009-01-07"])
    p = pd.DataFrame({"X": [0.01, np.nan, np.nan, 0.02]}, index=dates)  # 2-day gap
    mask = p.notna()
    filled, n = cap_forward_fill(p, mask, max_days=1)
    # With cap=1, only the first missing day is bridged; the second stays NaN.
    assert filled.loc["2009-01-05", "X"] == pytest.approx(0.01)
    assert pd.isna(filled.loc["2009-01-06", "X"])
    assert n == 1


def test_winsorize_clips_extremes() -> None:
    p = pd.DataFrame({"X": [0.0, 0.0, 0.0, 0.0, 10.0]})  # one huge spike
    clipped, n = winsorize_returns(p, 0.0, 0.8)
    assert clipped["X"].max() < 10.0
    assert n >= 1


def test_clean_index_series_reindexes_and_fills() -> None:
    idx = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2009-01-02", "2009-01-06"]),
    )
    union = pd.to_datetime(
        ["2009-01-02", "2009-01-05", "2009-01-06", "2009-01-07"])
    out = clean_index_series(idx, pd.DatetimeIndex(union), max_ffill=1)
    # Missing 2009-01-05 (inside span) is forward-filled from 0.01.
    assert out.loc["2009-01-05"] == pytest.approx(0.01)
    # 2009-01-07 is outside the market's own span -> stays NaN.
    assert pd.isna(out.loc["2009-01-07"])


def test_sparse_day_detection_logic() -> None:
    """Days with too few active names must be identifiable for removal."""
    dates = pd.to_datetime(
        ["2009-01-01", "2009-01-02", "2009-01-05"])
    # Day 1: only 1 of 3 names active (artefact). Days 2-3: all active.
    panel = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [np.nan, 0.01, 0.00],
            "C": [np.nan, -0.01, 0.02],
        },
        index=dates,
    )
    membership = build_membership_mask(panel)
    daily_active = membership.sum(axis=1)
    min_active = 2
    sparse = daily_active[(daily_active > 0) & (daily_active < min_active)].index
    assert list(sparse) == [pd.Timestamp("2009-01-01")]
    # After clearing, that day has zero active names.
    membership.loc[sparse, :] = False
    assert membership.loc["2009-01-01"].sum() == 0
