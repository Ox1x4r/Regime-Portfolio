"""
Tests for src.portfolios.

Each strategy is checked for the portfolio contract (long-only, sums to one)
and for correct behaviour on data with known structure: minimum variance must
overweight a low-variance asset, equal weight must be uniform.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolios import (
    investable_window,
    estimate_covariance,
    equal_weight,
    min_variance,
    mean_variance,
    hrp,
)


@pytest.fixture()
def structured_returns() -> pd.DataFrame:
    """Four assets: A low-vol, B and C correlated, D high-vol."""
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0.0003, 0.004, n)
    common = rng.normal(0, 0.01, n)
    b = 0.0004 + 0.7 * common + rng.normal(0, 0.004, n)
    c = 0.0004 + 0.7 * common + rng.normal(0, 0.004, n)
    d = rng.normal(0.0002, 0.030, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)


def _valid_weights(w: pd.Series) -> bool:
    return (
        abs(w.sum() - 1.0) < 1e-6
        and (w >= -1e-9).all()
    )


def test_equal_weight_uniform(structured_returns) -> None:
    w = equal_weight(structured_returns)
    assert _valid_weights(w)
    assert np.allclose(w.values, 0.25)


def test_min_variance_overweights_low_vol(structured_returns) -> None:
    w = min_variance(structured_returns, cov_method="sample")
    assert _valid_weights(w)
    # A (low vol) should get more weight than D (high vol).
    assert w["A"] > w["D"]
    # High-vol asset D should be strongly underweighted.
    assert w["D"] < 0.15


def test_mean_variance_valid(structured_returns) -> None:
    w = mean_variance(structured_returns, cov_method="ledoit_wolf")
    assert _valid_weights(w)


def test_hrp_valid_and_diversified(structured_returns) -> None:
    w = hrp(structured_returns, cov_method="sample")
    assert _valid_weights(w)
    # HRP should give the lone high-vol asset less than naive 1/N.
    assert w["D"] < 0.25
    # All four assets retain some allocation (risk parity spreads weight).
    assert (w > 0).all()


def test_ledoit_wolf_is_well_conditioned() -> None:
    """With N close to T, Ledoit-Wolf must stay invertible where sample fails."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0, 0.01, size=(60, 50)))  # N=50, T=60
    S_lw = estimate_covariance(X, "ledoit_wolf")
    # Condition number should be finite and the matrix invertible.
    assert np.isfinite(np.linalg.cond(S_lw))
    np.linalg.inv(S_lw)  # must not raise


def test_investable_window_respects_lookback_and_coverage() -> None:
    idx = pd.bdate_range("2015-01-01", periods=300)
    panel = pd.DataFrame(
        {"A": np.random.normal(0, 0.01, 300),
         "B": np.random.normal(0, 0.01, 300)},
        index=idx,
    )
    # B leaves the universe for the last 100 days -> not full coverage.
    panel.loc[idx[-100:], "B"] = np.nan
    win = investable_window(panel, idx[-1], lookback=252, min_coverage=1.0)
    assert win.shape[0] == 252
    assert "A" in win.columns and "B" not in win.columns  # B excluded


def test_max_universe_cap() -> None:
    idx = pd.bdate_range("2015-01-01", periods=300)
    panel = pd.DataFrame(
        {f"S{i}": np.random.normal(0, 0.01, 300) for i in range(20)},
        index=idx,
    )
    win = investable_window(panel, idx[-1], lookback=252,
                            min_coverage=1.0, max_universe=10)
    assert win.shape[1] == 10
