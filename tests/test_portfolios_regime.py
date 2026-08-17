"""
Tests for src.portfolios_regime.

Checks the portfolio contract for both methods and their distinctive
behaviour: CVaR avoids a fat-left-tail asset that variance-based methods
tolerate, and the equilibrium allocation tilts toward the asset favoured by the
active regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.portfolios_regime import (
    cvar_min,
    black_litterman_posterior,
    regime_specific_means,
    equilibrium_allocation,
    current_posterior,
)


def _valid(w: pd.Series) -> bool:
    return abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-9).all()


def test_cvar_valid_and_avoids_fat_tail() -> None:
    """
    Asset D matches C in variance but has a fat left tail; CVaR should
            prefer C.
    """
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.normal(0.0004, 0.008, n)
    b = rng.normal(0.0004, 0.008, n)
    c = rng.normal(0.0004, 0.015, n)
    # D: mostly small moves but occasional severe crashes (fat left tail).
    d = rng.normal(0.0006, 0.008, n)
    crash = rng.random(n) < 0.03
    d[crash] -= rng.uniform(0.05, 0.12, crash.sum())
    idx = pd.bdate_range("2015-01-01", periods=n)
    win = pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)

    w = cvar_min(win, confidence=0.95)
    assert _valid(w)
    # The fat-left-tail asset should be underweighted relative to naive 1/N.
    assert w["D"] < 0.25


def test_bl_posterior_between_prior_and_view() -> None:
    """With P=I, the BL posterior lies between the prior and the view."""
    N = 3
    Sigma = np.diag([0.04, 0.04, 0.04]) + 0.01
    Pi = np.array([0.00, 0.00, 0.00])          # flat prior
    Q = np.array([0.10, -0.05, 0.02])          # views
    post = black_litterman_posterior(Sigma, Pi, Q, tau=0.05, view_confidence=1.0)
    # Each posterior element should share the sign / ordering of the views,
    # and be pulled from the prior toward the view.
    assert post[0] > post[2] > post[1]
    assert np.all(np.abs(post) <= np.abs(Q) + 1e-9)


def test_regime_specific_means_fallback() -> None:
    idx = pd.bdate_range("2015-01-01", periods=100)
    win = pd.DataFrame({"X": np.linspace(-0.01, 0.01, 100),
                        "Y": np.linspace(0.01, -0.01, 100)}, index=idx)
    states = pd.Series([0] * 90 + [1] * 10, index=idx)   # regime 1 has few days
    mus = regime_specific_means(win, states, n_states=2, min_days=20)
    # regime 1 (only 10 days < min_days) must fall back to the overall mean
    assert np.allclose(mus[1], win.mean().values)
    # regime 0 has enough days -> its own mean, distinct from overall
    assert not np.allclose(mus[0], win.mean().values)


def test_equilibrium_tilts_with_active_regime() -> None:
    """
    Asset A wins in regime 0 and B in regime 1; the allocation should
            follow whichever the posterior favours.
    """
    from src.portfolios import regime_observation_weights

    rng = np.random.default_rng(1)
    n = 600
    idx = pd.bdate_range("2015-01-01", periods=n)
    states = np.array([0] * 300 + [1] * 300)
    a = np.where(states == 0, 0.0015, 0.0000) + rng.normal(0, 0.008, n)
    b = np.where(states == 1, 0.0015, 0.0000) + rng.normal(0, 0.008, n)
    c = rng.normal(0.0002, 0.008, n)
    win = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    # Hard posteriors matching the true states.
    post = pd.DataFrame({"p_state_0": (states == 0).astype(float),
                         "p_state_1": (states == 1).astype(float)}, index=idx)

    cfg = load_config()
    w0 = equilibrium_allocation(
        win, cfg, regime_observation_weights(idx, post, np.array([0.9, 0.1])))
    w1 = equilibrium_allocation(
        win, cfg, regime_observation_weights(idx, post, np.array([0.1, 0.9])))
    assert _valid(w0) and _valid(w1)
    assert w0["A"] > w1["A"]     # regime-0 posterior lifts A
    assert w1["B"] > w0["B"]     # regime-1 posterior lifts B


def test_equilibrium_unconditional_is_valid() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=400)
    win = pd.DataFrame(rng.normal(0.0004, 0.01, (400, 4)),
                       columns=list("ABCD"), index=idx)
    w = equilibrium_allocation(win, load_config(), None)
    assert _valid(w)


def test_current_posterior_picks_last_available() -> None:
    idx = pd.bdate_range("2015-01-01", periods=5)
    post = pd.DataFrame(
        {"p_state_0": [0.9, 0.8, 0.2, 0.1, 0.3],
         "p_state_1": [0.1, 0.2, 0.8, 0.9, 0.7]}, index=idx)
    pi = current_posterior(post, idx[3])
    assert np.allclose(pi, [0.1, 0.9])
