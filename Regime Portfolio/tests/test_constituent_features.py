"""
Tests for src.constituent_features and filtered inference.

Feature tests use panels with known cross-sectional structure, so each feature
is checked against the value it should recover. The inference tests check that
filtered and smoothed probabilities coincide at the final observation but
differ mid-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from src.constituent_features import (
    constituent_features,
    _daily_cross_sectional,
    _avg_pairwise_correlation,
    _eigenvalue_share,
)
from src.regimes import filtered_probabilities


def _panel(n_days: int, n_names: int, rho: float, seed: int = 0,
           scale: float = 0.01) -> pd.DataFrame:
    """Panel with a known equicorrelation ``rho`` between all names."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, (n_days, 1))
    idio = rng.normal(0, 1, (n_days, n_names))
    x = (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio) * scale
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    return pd.DataFrame(x, index=idx,
                        columns=[f"c{i}" for i in range(n_names)])


def test_daily_cross_sectional_dispersion_and_breadth() -> None:
    idx = pd.bdate_range("2012-01-02", periods=3)
    # Day 1: all positive; Day 2: all negative; Day 3: mixed.
    df = pd.DataFrame({"a": [0.01, -0.01, 0.01],
                       "b": [0.02, -0.02, -0.01],
                       "c": [0.03, -0.03, 0.00]}, index=idx)
    out = _daily_cross_sectional(df)
    assert out.loc[idx[0], "frac_negative"] == pytest.approx(0.0)
    assert out.loc[idx[1], "frac_negative"] == pytest.approx(1.0)
    assert out.loc[idx[2], "frac_negative"] == pytest.approx(1 / 3)
    # dispersion is in percentage points and strictly positive here
    assert out.loc[idx[0], "xs_dispersion"] > 0


def test_avg_pairwise_correlation_recovers_rho() -> None:
    """The variance-ratio identity must recover a known equicorrelation."""
    rho_true = 0.4
    p = _panel(600, 120, rho_true, seed=1)
    rho = _avg_pairwise_correlation(p, window=250).dropna()
    assert rho.mean() == pytest.approx(rho_true, abs=0.08)


def test_avg_pairwise_correlation_orders_regimes() -> None:
    low = _avg_pairwise_correlation(_panel(400, 80, 0.1, seed=2), 250).dropna()
    high = _avg_pairwise_correlation(_panel(400, 80, 0.7, seed=2), 250).dropna()
    assert high.mean() > low.mean()


def test_eigenvalue_share_rises_with_correlation() -> None:
    """A stronger common factor must raise the leading-eigenvalue share."""
    low = _eigenvalue_share(_panel(400, 60, 0.05, seed=3), 250, 60).dropna()
    high = _eigenvalue_share(_panel(400, 60, 0.8, seed=3), 250, 60).dropna()
    assert high.mean() > low.mean()
    # share is bounded in (0, 1]
    assert (high > 0).all() and (high <= 1.0).all()


def test_constituent_features_shape_and_no_lookahead() -> None:
    p = _panel(400, 50, 0.3, seed=4)
    f = constituent_features(p, window=63, max_names_for_eig=50)
    assert list(f.columns) == ["xs_dispersion", "frac_negative", "xs_skew",
                              "avg_pairwise_corr", "eig1_share"]
    assert f.notna().all().all()
    # Warm-up rows are dropped, so the feature series starts after the window.
    assert f.index[0] >= p.index[62]
    # Truncating the panel must not change earlier feature values (no look-ahead).
    f_short = constituent_features(p.iloc[:300], window=63, max_names_for_eig=50)
    common = f.index.intersection(f_short.index)
    assert len(common) > 50
    pd.testing.assert_frame_equal(
        f.loc[common, ["xs_dispersion", "frac_negative", "xs_skew"]],
        f_short.loc[common, ["xs_dispersion", "frac_negative", "xs_skew"]],
    )


def test_filtered_probabilities_properties() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.005, (150, 2)),
                   rng.normal(0, 0.02, (150, 2))])
    m = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=200, random_state=0).fit(X)
    fl = filtered_probabilities(m, X)
    sm = m.predict_proba(X)
    # valid distribution
    assert np.allclose(fl.sum(axis=1), 1.0)
    assert (fl >= 0).all()
    # filtered == smoothed at the final observation (no future information)
    assert np.allclose(fl[-1], sm[-1], atol=1e-8)
    # but they must genuinely differ mid-sample, else it is not the forward pass
    assert not np.allclose(fl[75], sm[75], atol=1e-6)
