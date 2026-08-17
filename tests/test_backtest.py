"""
Tests for src.backtest.

Covers the engine mechanics: rebalance schedule, holding-period return
accounting and transaction-cost deduction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    generate_rebalance_dates,
    _holding_period_returns,
)


def test_rebalance_dates_are_period_ends_after_train_end() -> None:
    idx = pd.bdate_range("2011-01-01", "2013-12-31")
    dates = generate_rebalance_dates(idx, pd.Timestamp("2012-01-01"), "M")
    # All after train_end, monotonic, and one per month (~24 months).
    assert all(d >= pd.Timestamp("2012-01-01") for d in dates)
    assert 22 <= len(dates) <= 24
    assert list(dates) == sorted(dates)


def test_rebalance_quarterly_fewer_than_monthly() -> None:
    idx = pd.bdate_range("2011-01-01", "2013-12-31")
    m = generate_rebalance_dates(idx, pd.Timestamp("2012-01-01"), "M")
    q = generate_rebalance_dates(idx, pd.Timestamp("2012-01-01"), "Q")
    assert len(q) < len(m)


def test_holding_period_returns_weighted_sum() -> None:
    dates = pd.bdate_range("2012-02-01", periods=3)
    holding = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01], "B": [0.00, -0.01, 0.02]}, index=dates)
    w = pd.Series({"A": 0.5, "B": 0.5})
    port = _holding_period_returns(holding, w)
    # Day 1: 0.5*0.01 + 0.5*0.00 = 0.005, etc.
    assert port.iloc[0] == pytest.approx(0.005)
    assert port.iloc[1] == pytest.approx(0.005)
    assert port.iloc[2] == pytest.approx(0.005)


def test_holding_period_handles_missing_asset_returns() -> None:
    dates = pd.bdate_range("2012-02-01", periods=2)
    holding = pd.DataFrame({"A": [0.01, np.nan]}, index=dates)  # B absent entirely
    w = pd.Series({"A": 0.5, "B": 0.5})
    port = _holding_period_returns(holding, w)
    # B has no column -> treated as 0; A's NaN on day 2 -> 0.
    assert port.iloc[0] == pytest.approx(0.5 * 0.01)
    assert port.iloc[1] == pytest.approx(0.0)


def test_holding_period_no_lookahead_uses_only_given_window() -> None:
    # Sanity: function only touches the rows it is given (the holding period),
    # never future rows -- there are none to touch here by construction.
    dates = pd.bdate_range("2012-02-01", periods=1)
    holding = pd.DataFrame({"A": [0.03]}, index=dates)
    w = pd.Series({"A": 1.0})
    port = _holding_period_returns(holding, w)
    assert len(port) == 1
    assert port.iloc[0] == pytest.approx(0.03)


def test_apply_transaction_cost_is_exact() -> None:
    """Re-pricing to a new cost level must equal charging it directly."""
    from src.backtest import apply_transaction_cost

    idx = pd.bdate_range("2012-01-02", periods=10)
    gross = pd.Series(np.full(10, 0.001), index=idx)
    turn = pd.Series([0.4, 0.25], index=[idx[0], idx[5]])

    # Charge 10bp directly.
    at10 = gross.copy()
    at10.loc[turn.index] -= (10.0 / 1e4) * turn.values
    # Charge 20bp directly.
    at20 = gross.copy()
    at20.loc[turn.index] -= (20.0 / 1e4) * turn.values

    repriced = apply_transaction_cost(at10, turn, from_bps=10.0, to_bps=20.0)
    assert np.allclose(repriced.values, at20.values, atol=1e-15)

    # Round-trip back to the original level.
    back = apply_transaction_cost(repriced, turn, from_bps=20.0, to_bps=10.0)
    assert np.allclose(back.values, at10.values, atol=1e-15)


def test_apply_transaction_cost_noop_when_unchanged() -> None:
    from src.backtest import apply_transaction_cost
    idx = pd.bdate_range("2012-01-02", periods=5)
    r = pd.Series(np.arange(5) * 0.001, index=idx)
    turn = pd.Series([0.5], index=[idx[0]])
    same = apply_transaction_cost(r, turn, from_bps=10.0, to_bps=10.0)
    assert np.allclose(same.values, r.values)
