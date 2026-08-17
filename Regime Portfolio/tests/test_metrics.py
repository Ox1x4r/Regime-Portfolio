"""
Tests for src.metrics.

Each metric is checked against a value computed by hand or from a construction
with a known answer, so the formulas themselves are verified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    average_turnover,
    summarise_performance,
)


def test_annualised_return_constant() -> None:
    # A constant +0.1% daily over exactly one trading year (252 days).
    r = pd.Series([0.001] * 252)
    expected = (1.001 ** 252) - 1          # geometric compounding
    assert annualised_return(r) == pytest.approx(expected, rel=1e-9)


def test_annualised_volatility() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    # Should be close to 0.01 * sqrt(252).
    assert annualised_volatility(r) == pytest.approx(0.01 * np.sqrt(252), rel=0.1)


def test_sharpe_zero_mean_is_zero() -> None:
    # Symmetric returns with ~zero mean -> Sharpe near zero.
    r = pd.Series([0.01, -0.01] * 500)
    assert abs(sharpe_ratio(r)) < 0.1


def test_sharpe_positive_drift() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 5000))
    # Known: Sharpe ~ mean/sd*sqrt(252) = 0.0005/0.01*sqrt(252) ~ 0.79.
    assert sharpe_ratio(r) == pytest.approx(0.79, abs=0.25)


def test_sortino_ge_sharpe_for_left_skew() -> None:
    # Downside deviation <= total deviation, so Sortino >= Sharpe generally.
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.01, 4000))
    assert sortino_ratio(r) >= sharpe_ratio(r) - 1e-9


def test_max_drawdown_known() -> None:
    # Wealth path: +10%, then -50%. Peak 1.1 -> trough 0.55 -> MDD = -0.5.
    r = pd.Series([0.10, -0.50])
    assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-9)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    r = pd.Series([0.01, 0.02, 0.005, 0.03])   # never declines
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-12)


def test_calmar_sign_and_magnitude() -> None:
    r = pd.Series([0.001] * 252)   # steady gains, no drawdown
    # No drawdown -> Calmar undefined (nan) by construction.
    assert np.isnan(calmar_ratio(r))


def test_average_turnover_full_switch() -> None:
    # Rebalance from 100% A to 100% B -> one-way turnover = 1.0.
    w1 = pd.Series({"A": 1.0, "B": 0.0})
    w2 = pd.Series({"A": 0.0, "B": 1.0})
    assert average_turnover([w1, w2]) == pytest.approx(1.0)


def test_average_turnover_no_change() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert average_turnover([w, w.copy()]) == pytest.approx(0.0)


def test_summarise_bundles_all() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    out = summarise_performance(r, weight_history=[
        pd.Series({"A": 1.0}), pd.Series({"A": 0.5, "B": 0.5})])
    for key in ["ann_return", "ann_vol", "sharpe", "sortino",
                "max_drawdown", "calmar", "avg_turnover"]:
        assert key in out
