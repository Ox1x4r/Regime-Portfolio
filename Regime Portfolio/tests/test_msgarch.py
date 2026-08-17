"""
Tests for src.msgarch.

Covers the Python post-processing around the R call: relabelling by volatility,
decoding from smoothed probabilities, and the per-regime summary. The R fit
itself needs R, so it is not exercised here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.msgarch import (
    relabel_states_by_vol,
    decode_from_smoothed,
    summarise_regimes,
)


def test_relabel_orders_states_ascending_vol() -> None:
    # Raw state 0 is high-vol, state 1 is low-vol -> should swap.
    state_vol = np.array([0.30, 0.10])
    mapping = relabel_states_by_vol(state_vol)
    assert mapping[1] == 0        # low-vol raw state -> label 0
    assert mapping[0] == 1        # high-vol raw state -> label 1


def test_relabel_three_states() -> None:
    state_vol = np.array([0.20, 0.40, 0.10])   # mid, high, low
    mapping = relabel_states_by_vol(state_vol)
    # low(0.10)->0, mid(0.20)->1, high(0.40)->2
    assert list(mapping) == [1, 2, 0]


def test_decode_from_smoothed_argmax() -> None:
    smoothed = np.array([[0.8, 0.2],
                         [0.3, 0.7],
                         [0.6, 0.4]])
    states = decode_from_smoothed(smoothed)
    assert list(states) == [0, 1, 0]


def test_summarise_regimes_stats() -> None:
    idx = pd.bdate_range("2010-01-01", periods=6)
    r = pd.Series([0.01, 0.012, -0.03, -0.04, 0.008, 0.009], index=idx)
    states = np.array([0, 0, 1, 1, 0, 0])
    stats = summarise_regimes(r, states, n_states=2)
    assert stats.loc[0, "n_days"] == 4
    assert stats.loc[1, "n_days"] == 2
    # Regime 1 (the negative block) is more volatile than regime 0.
    assert stats.loc[1, "vol_ann"] > stats.loc[0, "vol_ann"]
    assert stats.loc[1, "mean_ann"] < 0


def test_smoothed_relabelling_reorders_columns() -> None:
    """After relabelling, probability columns must follow the new labels."""
    smoothed = np.array([[0.9, 0.1], [0.2, 0.8]])
    raw_states = decode_from_smoothed(smoothed)          # [0, 1]
    state_vol = np.array([0.30, 0.10])                   # state 0 high vol
    mapping = relabel_states_by_vol(state_vol)           # swap
    new_states = mapping[raw_states]
    new_smoothed = smoothed[:, np.argsort(mapping)]
    # Row 0 was raw-state 0 (now label 1): its high prob should sit in col 1.
    assert new_states[0] == 1
    assert new_smoothed[0, 1] == pytest.approx(0.9)
