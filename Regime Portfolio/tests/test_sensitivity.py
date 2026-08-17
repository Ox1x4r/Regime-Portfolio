"""
Tests for src.sensitivity.

Covers config cloning, which must not mutate the original, and the crisis
sub-period breakdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.sensitivity import cfg_with, crisis_breakdown


def test_cfg_with_overrides_and_isolation() -> None:
    cfg = load_config()
    orig_vc = cfg.portfolio.get("bl_view_confidence")
    c2 = cfg_with(cfg, portfolio={"bl_view_confidence": 0.123})
    # the clone has the new value
    assert c2.portfolio["bl_view_confidence"] == 0.123
    # the original is untouched (no shared mutable state)
    assert cfg.portfolio.get("bl_view_confidence") == orig_vc


def test_cfg_with_multiple_sections() -> None:
    cfg = load_config()
    c2 = cfg_with(cfg,
                  portfolio={"default_lookback": 63},
                  backtest={"transaction_cost_bps": 25.0})
    assert c2.portfolio["default_lookback"] == 63
    assert c2.backtest["transaction_cost_bps"] == 25.0


def test_crisis_breakdown_slices_and_full() -> None:
    idx = pd.bdate_range("2019-06-01", "2020-12-31")
    rng = np.random.default_rng(0)
    # Make the COVID window distinctly negative.
    daily = pd.Series(rng.normal(0.0005, 0.008, len(idx)), index=idx)
    covid = (idx >= pd.Timestamp("2020-02-19")) & (idx <= pd.Timestamp("2020-04-30"))
    daily[covid] = -0.01
    returns = {"stratA": daily}
    periods = [{"name": "COVID-19 crash", "start": "2020-02-19",
                "end": "2020-04-30"}]
    df = crisis_breakdown(returns, periods)
    # one crisis row + one FULL row
    assert set(df["period"]) == {"COVID-19 crash", "FULL"}
    covid_row = df[df["period"] == "COVID-19 crash"].iloc[0]
    full_row = df[df["period"] == "FULL"].iloc[0]
    # the crisis window must show a worse (more negative) max drawdown region
    assert covid_row["ann_return"] < full_row["ann_return"]
    assert covid_row["n_days"] > 0


def test_crisis_breakdown_multiple_strategies() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    r = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
    returns = {"A": r, "B": r * 0.5}
    periods = [{"name": "COVID-19 crash", "start": "2020-02-19",
                "end": "2020-04-30"}]
    df = crisis_breakdown(returns, periods)
    assert set(df["strategy"]) == {"A", "B"}
    assert len(df) == 4        # 2 strategies x (1 crisis + FULL)
