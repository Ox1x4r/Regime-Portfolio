"""
Tests for src.eda.

Checks the test battery against series with known properties: normal iid data
should not reject normality, and a GARCH-like series should show excess
kurtosis and significant ARCH effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda import series_stylised_facts, _panel_moment_facts


@pytest.fixture()
def normal_series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.01, size=2000))


@pytest.fixture()
def garch_series() -> pd.Series:
    """GARCH(1,1)-like process with volatility clustering."""
    rng = np.random.default_rng(1)
    n = 3000
    omega, alpha, beta = 1e-6, 0.1, 0.88
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return pd.Series(eps)


def test_moments_on_normal(normal_series: pd.Series) -> None:
    f = series_stylised_facts(normal_series, lb_lags=10, arch_lags=10)
    assert abs(f["skew"]) < 0.3                 # ~symmetric
    assert abs(f["excess_kurtosis"]) < 0.5      # ~mesokurtic
    assert f["jarque_bera_p"] > 0.01            # do not reject normality


def test_arch_detected_on_garch(garch_series: pd.Series) -> None:
    f = series_stylised_facts(garch_series, lb_lags=10, arch_lags=10)
    # Volatility clustering => fat tails and significant ARCH-LM.
    assert f["excess_kurtosis"] > 0.5
    assert f["arch_lm_p"] < 0.05
    # Squared-return autocorrelation should be significant.
    assert f["ljungbox_sq_p"] < 0.05


def test_returns_stationary(garch_series: pd.Series) -> None:
    f = series_stylised_facts(garch_series, lb_lags=10, arch_lags=10)
    # Daily returns are stationary: ADF should reject the unit-root null.
    assert f["adf_p"] < 0.05


def test_short_series_returns_minimal() -> None:
    f = series_stylised_facts(pd.Series([0.01, -0.02, 0.0]),
                              lb_lags=10, arch_lags=10)
    assert f["n_obs"] == 3.0
    assert "skew" not in f            # too short: no moments computed


def test_panel_moment_facts_shape() -> None:
    df = pd.DataFrame(
        {"A": np.random.normal(0, 0.01, 300),
         "B": np.random.normal(0, 0.02, 300)})
    facts = _panel_moment_facts(df)
    assert set(facts.index) == {"A", "B"}
    assert "vol_ann" in facts.columns
    assert facts.loc["B", "vol_ann"] > facts.loc["A", "vol_ann"]
