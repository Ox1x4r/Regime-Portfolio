"""
Risk-adjusted performance metrics.

All functions take a daily net-of-cost return series. Returns are simple
(arithmetic); the risk-free rate is given annually and converted to daily.
Annualisation uses 252 trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: pd.Series) -> pd.Series:
    return returns.dropna().astype(float)


def annualised_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Geometric annualised return."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    growth = float((1.0 + r).prod())
    years = len(r) / trading_days
    if years <= 0 or growth <= 0:
        return np.nan
    return growth ** (1.0 / years) - 1.0


def annualised_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio using daily excess returns."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    excess = r - rf_daily
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio (downside-deviation-adjusted)."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    excess = r - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return np.nan
    return float(excess.mean() / dd * np.sqrt(trading_days))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline of cumulative wealth (a negative number)."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    ann = annualised_return(returns, trading_days)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(ann / abs(mdd))


def average_turnover(weight_history: list[pd.Series]) -> float:
    """Mean one-way turnover across rebalances.

    One-way turnover at a rebalance is 0.5 * sum |w_new - w_old| over the union
    of assets. The first rebalance (from all-cash) is excluded from the average.
    """
    if len(weight_history) < 2:
        return np.nan
    turnovers = []
    for prev, curr in zip(weight_history[:-1], weight_history[1:]):
        allassets = prev.index.union(curr.index)
        p = prev.reindex(allassets).fillna(0.0)
        c = curr.reindex(allassets).fillna(0.0)
        turnovers.append(0.5 * float((c - p).abs().sum()))
    return float(np.mean(turnovers)) if turnovers else np.nan


def summarise_performance(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    weight_history: list[pd.Series] | None = None,
    trading_days: int = TRADING_DAYS,
) -> dict[str, float]:
    """Bundle the full metric set into one dict (one row per strategy)."""
    out = {
        "ann_return": annualised_return(returns, trading_days),
        "ann_vol": annualised_volatility(returns, trading_days),
        "sharpe": sharpe_ratio(returns, risk_free_annual, trading_days),
        "sortino": sortino_ratio(returns, risk_free_annual, trading_days),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, trading_days),
    }
    if weight_history is not None:
        out["avg_turnover"] = average_turnover(weight_history)
    return out
