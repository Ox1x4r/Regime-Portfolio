"""
Tests for the regime-conditioning mechanism.

Conditioning must enter through one channel only -- observation weights from
the current filtered posterior -- so that ``method_rc`` minus ``method``
isolates it. These check the weighting, the weighted estimators, and that every
strategy produces a valid portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.portfolios import (
    factorial_strategies,
    parse_strategy,
    regime_observation_weights,
    estimate_covariance,
    estimate_expected_returns,
    build_weights,
)


def _two_regime_window(n: int = 400, seed: int = 0):
    """
    Window with two regimes, differing in both mean and volatility.

    In regime 0 asset A has the higher mean and lower volatility; in regime 1 the
    roles swap. Volatility differs because covariance-only methods respond to the
    second moment alone.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    states = np.array([0] * (n // 2) + [1] * (n - n // 2))
    sd_a = np.where(states == 0, 0.004, 0.014)
    sd_b = np.where(states == 0, 0.014, 0.004)
    a = np.where(states == 0, 0.002, -0.001) + rng.normal(0, 1, n) * sd_a
    b = np.where(states == 1, 0.002, -0.001) + rng.normal(0, 1, n) * sd_b
    win = pd.DataFrame({"A": a, "B": b}, index=idx)
    post = pd.DataFrame({"p_state_0": (states == 0).astype(float),
                         "p_state_1": (states == 1).astype(float)}, index=idx)
    return win, post, states


def test_factorial_strategy_list_is_complete() -> None:
    strats = factorial_strategies()
    assert "equal_weight" in strats and "index_benchmark" in strats
    for m in ["mean_variance", "min_variance", "hrp", "cvar", "equilibrium"]:
        assert m in strats and f"{m}_rc" in strats
    # 5 methods x 2 conditions + 2 benchmarks
    assert len(strats) == 12


def test_parse_strategy() -> None:
    assert parse_strategy("hrp") == ("hrp", False)
    assert parse_strategy("hrp_rc") == ("hrp", True)


def test_observation_weights_normalised_and_selective() -> None:
    win, post, states = _two_regime_window()
    w = regime_observation_weights(win.index, post, np.array([1.0, 0.0]))
    # normalised to sum to the window length
    assert w.sum() == pytest.approx(len(win))
    # all mass on regime-0 days
    assert w[states == 0].sum() == pytest.approx(len(win))
    assert w[states == 1].sum() == pytest.approx(0.0)


def test_observation_weights_blend_with_mixed_posterior() -> None:
    win, post, states = _two_regime_window()
    w = regime_observation_weights(win.index, post, np.array([0.5, 0.5]))
    # an even posterior weights both halves equally
    assert w[states == 0].sum() == pytest.approx(w[states == 1].sum(), rel=1e-6)


def test_weighted_mean_recovers_regime_specific_returns() -> None:
    win, post, states = _two_regime_window()
    w0 = regime_observation_weights(win.index, post, np.array([1.0, 0.0]))
    mu0 = estimate_expected_returns(win, w0)
    mu_uncond = estimate_expected_returns(win, None)
    # In regime 0, A is the good asset; conditioning must lift A's estimate
    # above its unconditional value.
    assert mu0[0] > mu_uncond[0]
    assert mu0[0] > mu0[1]


def test_weighted_covariance_matches_manual() -> None:
    win, post, _ = _two_regime_window(n=200, seed=3)
    w = regime_observation_weights(win.index, post, np.array([0.7, 0.3]))
    S = estimate_covariance(win, "sample", w)
    X = win.values
    wn = w * (len(w) / w.sum())
    mu = (wn[:, None] * X).sum(axis=0) / wn.sum()
    Xc = X - mu
    expected = (Xc * wn[:, None]).T @ Xc / (wn.sum() - 1.0)
    assert np.allclose(S, expected)
    assert np.allclose(S, S.T)          # symmetric


def test_all_factorial_strategies_produce_valid_weights() -> None:
    win, post, _ = _two_regime_window(n=300, seed=5)
    cfg = load_config()
    ctx = {"regime_posteriors": post,
           "regime_posterior": np.array([0.8, 0.2])}
    for strat in factorial_strategies(include_benchmarks=False) + ["equal_weight"]:
        w = build_weights(strat, win, cfg, ctx)
        assert abs(w.sum() - 1.0) < 1e-6, f"{strat} does not sum to 1"
        assert (w >= -1e-9).all(), f"{strat} has negative weights"


def test_rc_strategy_requires_context() -> None:
    win, _, _ = _two_regime_window(n=200)
    with pytest.raises(ValueError):
        build_weights("hrp_rc", win, load_config(), None)


def test_conditional_differs_from_unconditional() -> None:
    """Conditioning must change the portfolio, else the comparison is vacuous."""
    win, post, _ = _two_regime_window(n=300, seed=7)
    cfg = load_config()
    ctx = {"regime_posteriors": post,
           "regime_posterior": np.array([1.0, 0.0])}
    for base in ["mean_variance", "min_variance", "hrp", "cvar"]:
        w_u = build_weights(base, win, cfg, None)
        w_c = build_weights(base + "_rc", win, cfg, ctx)
        assert not np.allclose(w_u.values, w_c.values, atol=1e-6), base
