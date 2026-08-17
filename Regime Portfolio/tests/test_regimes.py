"""
Tests for src.regimes.

Builds a series with two known volatility regimes and checks that the HMM
recovers them, that states are relabelled by volatility, and that the AIC/BIC
parameter counts are right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regimes import (
    build_features,
    fit_hmm,
    _n_params,
    select_model,
    relabel_by_volatility,
    regime_statistics,
)


@pytest.fixture()
def two_regime_returns() -> tuple[pd.Series, np.ndarray]:
    """Series alternating between low-volatility and high-volatility blocks."""
    rng = np.random.default_rng(0)
    blocks, truth = [], []
    for _ in range(6):
        calm = rng.normal(0.0005, 0.005, 120)      # low vol
        turb = rng.normal(-0.001, 0.025, 80)       # high vol
        blocks += [calm, turb]
        truth += [0] * 120 + [1] * 80
    r = np.concatenate(blocks)
    dates = pd.bdate_range("2010-01-01", periods=len(r))
    return pd.Series(r, index=dates), np.array(truth)


def test_build_features_shapes() -> None:
    r = pd.Series(np.random.normal(0, 0.01, 100),
                  index=pd.bdate_range("2010-01-01", periods=100))
    f = build_features(r, ["return", "realised_vol"], vol_window=21)
    assert list(f.columns) == ["return", "realised_vol"]
    assert len(f) == 100 - 20            # warm-up rows dropped
    assert f.notna().all().all()


def test_n_params_full_cov() -> None:
    # K=2, d=2, full cov: trans=2, start=1, means=4, cov=2*3=6 -> 13
    assert _n_params(2, 2, "full") == 13
    # K=3, d=1, full cov: trans=6, start=2, means=3, cov=3 -> 14
    assert _n_params(3, 1, "full") == 14


def test_hmm_recovers_two_regimes(two_regime_returns) -> None:
    r, truth = two_regime_returns
    feats = build_features(r, ["return", "realised_vol"], vol_window=21)
    X = feats.values
    model, ll = fit_hmm(X, n_states=2, cov_type="full",
                        n_iter=500, n_init=5, seed=42)
    states = model.predict(X)
    ret_col = feats["return"].values
    states, _ = relabel_by_volatility(model, states, ret_col)
    # Align decoded states with the (truncated) truth and check agreement.
    truth_aligned = truth[-len(states):]
    agreement = max(
        (states == truth_aligned).mean(),
        (states == (1 - truth_aligned)).mean(),  # label invariance guard
    )
    assert agreement > 0.85          # recovers regimes well


def test_relabel_orders_by_volatility(two_regime_returns) -> None:
    r, _ = two_regime_returns
    feats = build_features(r, ["return", "realised_vol"], vol_window=21)
    X = feats.values
    model, _ = fit_hmm(X, 2, "full", 500, 5, 42)
    states = model.predict(X)
    ret_col = feats["return"].values
    relabelled, _ = relabel_by_volatility(model, states, ret_col)
    stats = regime_statistics(relabelled, ret_col, 2)
    # Regime 0 must be calmer than regime 1 by construction of the relabelling.
    assert stats.loc[0, "vol_ann"] < stats.loc[1, "vol_ann"]


def test_select_model_picks_by_bic(two_regime_returns) -> None:
    r, _ = two_regime_returns
    feats = build_features(r, ["return", "realised_vol"], vol_window=21)
    X = feats.values
    reg = {"covariance_type": "full", "n_iter": 300, "n_init": 3,
           "model_selection": "bic"}
    table, k, model = select_model(X, [2, 3], reg, seed=42)
    assert set(table.index) == {2, 3}
    assert "BIC" in table.columns
    assert k in (2, 3)
    assert model.n_components == k
