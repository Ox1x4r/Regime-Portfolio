"""
Tests for src.dependence.

Checks that the DCC-GARCH implementation recovers a known correlation from
synthetic data, and that the contagion and Epps diagnostics detect injected
effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dependence import (
    weekly_correlations,
    epps_comparison,
    static_correlations,
    rolling_pairwise_correlations,
    average_rolling_correlation,
    crisis_vs_calm,
    fit_dcc,
)


@pytest.fixture()
def correlated_returns() -> pd.DataFrame:
    """Three series with a known constant correlation of 0.5."""
    rng = np.random.default_rng(0)
    n = 2500
    rho = 0.5
    cov = np.full((3, 3), rho)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 3)) @ L.T
    dates = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(z * 0.01, columns=["A", "B", "C"], index=dates)


def test_static_correlation_recovers_rho(correlated_returns) -> None:
    pearson, spearman = static_correlations(correlated_returns)
    off = pearson.values[np.triu_indices(3, k=1)]
    assert np.allclose(off, 0.5, atol=0.08)
    assert pearson.shape == (3, 3)


def test_rolling_shapes(correlated_returns) -> None:
    roll = rolling_pairwise_correlations(correlated_returns, window=63)
    assert roll.shape[1] == 3          # 3 pairs from 3 series
    avg = average_rolling_correlation(roll)
    assert avg.name == "avg_pairwise_corr"
    assert 0.3 < avg.dropna().mean() < 0.7


def test_crisis_contagion_detected() -> None:
    """Inject a high-correlation window and confirm it is flagged."""
    rng = np.random.default_rng(1)
    n = 1000
    dates = pd.bdate_range("2019-01-01", periods=n)
    # Calm: near-zero correlation.
    calm = rng.standard_normal((n, 3)) * 0.01
    df = pd.DataFrame(calm, columns=["A", "B", "C"], index=dates)
    # Crisis window: overwrite with highly correlated shocks.
    cstart, cend = dates[400], dates[460]
    common_shock = rng.standard_normal(61)[:, None] * 0.02
    df.loc[cstart:cend] = common_shock + rng.standard_normal((61, 3)) * 0.002
    crisis = [{"name": "test crisis", "start": str(cstart.date()),
               "end": str(cend.date())}]
    out = crisis_vs_calm(df, crisis)
    assert out.loc["ALL CRISIS", "avg_corr"] > out.loc["CALM (rest)", "avg_corr"]
    assert out.loc["ALL CRISIS", "contagion_vs_calm"] > 0


def test_dcc_recovers_constant_correlation(correlated_returns) -> None:
    """With constant-correlation data, DCC's mean matrix ~ the true matrix."""
    res = fit_dcc(correlated_returns)
    m = res["mean_matrix"].values
    off = m[np.triu_indices(3, k=1)]
    # DCC on constant-corr data should centre near 0.5.
    assert np.allclose(off, 0.5, atol=0.12)
    # Parameters must satisfy the stationarity constraint.
    assert 0 <= res["a"] < 1
    assert 0 <= res["b"] < 1
    assert res["persistence"] < 1.0


def test_weekly_correlation_shape(correlated_returns) -> None:
    wc = weekly_correlations(correlated_returns)
    assert wc.shape == (3, 3)
    assert np.allclose(np.diag(wc), 1.0)


def test_epps_uplift_detected_for_lagged_market() -> None:
    """Weekly correlation must exceed daily when one market lags the other."""
    rng = np.random.default_rng(7)
    n = 1500
    dates = pd.bdate_range("2012-01-02", periods=n)
    common = rng.normal(0, 0.01, n)
    a = common + rng.normal(0, 0.003, n)
    # B sees the same shock with a one-day delay (different time zone).
    b = np.concatenate([[0.0], common[:-1]]) + rng.normal(0, 0.003, n)
    df = pd.DataFrame({"A": a, "B": b}, index=dates)

    out = epps_comparison(df)
    row = out.iloc[0]
    # Weekly correlation must exceed daily correlation for the lagged pair.
    assert row["weekly_corr"] > row["daily_corr"]
    assert row["epps_uplift"] > 0
