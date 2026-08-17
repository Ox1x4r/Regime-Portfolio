"""
Robustness and sensitivity analysis.

Sweeps the backtest across its design choices and breaks performance down by
crisis sub-period:

* view confidence and tau for the equilibrium allocation
* estimation lookback window
* transaction cost level (computed analytically, no re-run needed)
* rebalancing frequency

Writes the ``sensitivity_*.csv`` tables.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .backtest import (
    run_backtest_for_index,
    generate_rebalance_dates,
    WalkForwardRegimes,
)
from .portfolios import investable_window
from .metrics import summarise_performance


# ---------------------------------------------------------------------------
# Config cloning
# ---------------------------------------------------------------------------
def cfg_with(cfg: Config, **sections) -> Config:
    """Deep-copied Config with the given sections updated."""
    raw = copy.deepcopy(cfg.raw)
    for section, updates in sections.items():
        if updates:
            raw[section].update(updates)
    return Config(raw=raw)


# ---------------------------------------------------------------------------
# Precompute walk-forward regime contexts once per market
# ---------------------------------------------------------------------------
def precompute_regime_contexts(name: str, cfg: Config) -> dict:
    """
    Compute the walk-forward regime context at each rebalance date once.

    Reused across a parameter grid so the HMM is fitted once per market rather than
    once per combination.
    """
    panel = pd.read_csv(
        cfg.processed_dir / f"{name}_constituents_clean.csv",
        index_col=0, parse_dates=True)
    index_returns = pd.read_csv(
        cfg.processed_dir / f"{name}_index_clean.csv",
        index_col=0, parse_dates=True)["index_return"]

    p, bt = cfg.portfolio, cfg.backtest
    lookback = int(p["default_lookback"])
    coverage = float(p["min_window_coverage"])
    max_universe = p.get("max_universe")
    rebal = generate_rebalance_dates(
        panel.index, pd.Timestamp(bt["train_end"]), bt["rebalance_frequency"])

    # The panel is needed for constituent-level features, and passing it keeps
    # the sweep's regime signal identical to the backtest's.
    wf = WalkForwardRegimes(index_returns, cfg, panel)
    contexts = {}
    for i in range(len(rebal) - 1):
        t = rebal[i]
        window = investable_window(panel, t, lookback, coverage, max_universe)
        if window.shape[1] == 0:
            continue
        contexts[t] = wf.context_at(t, window)
    return contexts


# ---------------------------------------------------------------------------
# 1) Regime-BL view sweep
# ---------------------------------------------------------------------------
def sweep_bl_views(
    cfg: Config | None = None,
    markets: list[str] | None = None,
    write_csv: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """Sweep the equilibrium view-confidence and tau grid."""
    from .parallel import parallel_map

    cfg = cfg or load_config()
    s = cfg.raw["sensitivity"]
    markets = markets or (s.get("sweep_markets") or
                          [spec["name"] for spec in cfg.indices])
    markets = [m for m in markets
               if (cfg.processed_dir / f"{m}_constituents_clean.csv").exists()]
    vc_grid = s["bl_view_confidence_grid"]
    tau_grid = s["bl_tau_grid"]

    jobs = [(m, cfg, vc_grid, tau_grid) for m in markets]
    results = parallel_map(_bl_market_job, jobs, workers, desc="bl-sweep")
    rows = [r for chunk in results for r in chunk]

    df = pd.DataFrame(rows)
    if write_csv and not df.empty:
        cfg.tables_dir.mkdir(parents=True, exist_ok=True)
        df = df.sort_values(["market", "view_confidence"])
        df.to_csv(cfg.tables_dir / "sensitivity_bl_view.csv", index=False)
        print("[io] wrote sensitivity_bl_view.csv")
    return df


# ---------------------------------------------------------------------------
# 2) Design-choice sweeps (all strategies)
# ---------------------------------------------------------------------------
def _run_and_collect(name, cfg, tag_col, tag_val) -> list[dict]:
    res = run_backtest_for_index(name, cfg, verbose=False)
    rows = []
    for strat, m in res["metrics"].items():
        rows.append({"market": name, "strategy": strat, tag_col: tag_val, **m})
    return rows


# --- module-level workers so sweeps can run in parallel worker processes ---
def _lookback_job(job):
    name, cfg, lb = job
    c = cfg_with(cfg, portfolio={"default_lookback": lb})
    return _run_and_collect(name, c, "lookback", lb)


def _rebalance_job(job):
    name, cfg, freq = job
    c = cfg_with(cfg, backtest={"rebalance_frequency": freq})
    return _run_and_collect(name, c, "rebalance_freq", freq)


def _bl_market_job(job):
    """Sweep the whole view grid for one market."""
    name, cfg, vc_grid, tau_grid = job
    contexts = precompute_regime_contexts(name, cfg)
    rows = []
    for vc in vc_grid:
        for tau in tau_grid:
            c = cfg_with(cfg, portfolio={"bl_view_confidence": vc,
                                         "bl_tau": tau})
            res = run_backtest_for_index(
                name, c, strategies=["equilibrium_rc"],
                verbose=False, regime_contexts=contexts)
            m = res["metrics"].get("equilibrium_rc", {})
            rows.append({"market": name, "view_confidence": vc,
                         "tau": tau, **m})
    return rows


def _crisis_job(job):
    name, cfg = job
    return name, crisis_breakdown_for_market(name, cfg, write_csv=True)


def sweep_lookback(cfg=None, markets=None, write_csv=True,
                   workers: int = 1) -> pd.DataFrame:
    """Vary the estimation lookback window for all strategies."""
    from .parallel import parallel_map

    cfg = cfg or load_config()
    s = cfg.raw["sensitivity"]
    markets = markets or (s.get("sweep_markets") or
                          [spec["name"] for spec in cfg.indices])
    markets = [m for m in markets
               if (cfg.processed_dir / f"{m}_constituents_clean.csv").exists()]
    jobs = [(m, cfg, lb) for m in markets for lb in s["lookback_grid"]]
    results = parallel_map(_lookback_job, jobs, workers, desc="lookback")
    rows = [r for chunk in results for r in chunk]
    df = pd.DataFrame(rows)
    if write_csv and not df.empty:
        df = df.sort_values(["market", "lookback", "strategy"])
        df.to_csv(cfg.tables_dir / "sensitivity_lookback.csv", index=False)
    return df


def sweep_transaction_cost(cfg=None, markets=None, write_csv=True) -> pd.DataFrame:
    """
    Vary the transaction cost for all strategies.

    Computed analytically from one backtest per market: weights do not depend on
    the cost rate, so re-running the loop per level would be wasted work.
    """
    from .backtest import apply_transaction_cost
    from .metrics import summarise_performance

    cfg = cfg or load_config()
    s = cfg.raw["sensitivity"]
    markets = markets or (s.get("sweep_markets") or
                          [spec["name"] for spec in cfg.indices])
    base_bps = float(cfg.backtest["transaction_cost_bps"])
    rf = float(cfg.backtest["risk_free_annual"])
    tdays = int(cfg.backtest["trading_days_per_year"])

    rows = []
    for name in markets:
        res = run_backtest_for_index(name, cfg, verbose=False)
        for tc in s["transaction_cost_grid_bps"]:
            for strat, daily in res["returns"].items():
                turn = res["turnover"].get(strat)
                repriced = apply_transaction_cost(
                    daily, turn, from_bps=base_bps, to_bps=float(tc))
                m = summarise_performance(
                    repriced, rf, res["weights"].get(strat), tdays)
                rows.append({"market": name, "strategy": strat,
                             "tc_bps": tc, **m})
            print(f"[txcost] {name} tc={tc}bps done (analytic)")
    df = pd.DataFrame(rows)
    if write_csv and not df.empty:
        df.to_csv(cfg.tables_dir / "sensitivity_txcost.csv", index=False)
    return df


def sweep_rebalance(cfg=None, markets=None, write_csv=True,
                    workers: int = 1) -> pd.DataFrame:
    """Vary the rebalancing frequency for all strategies."""
    from .parallel import parallel_map

    cfg = cfg or load_config()
    s = cfg.raw["sensitivity"]
    markets = markets or (s.get("sweep_markets") or
                          [spec["name"] for spec in cfg.indices])
    markets = [m for m in markets
               if (cfg.processed_dir / f"{m}_constituents_clean.csv").exists()]
    jobs = [(m, cfg, f) for m in markets
            for f in s["rebalance_frequency_grid"]]
    results = parallel_map(_rebalance_job, jobs, workers, desc="rebalance")
    rows = [r for chunk in results for r in chunk]
    df = pd.DataFrame(rows)
    if write_csv and not df.empty:
        df = df.sort_values(["market", "rebalance_freq", "strategy"])
        df.to_csv(cfg.tables_dir / "sensitivity_rebalance.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3) Crisis sub-period breakdown
# ---------------------------------------------------------------------------
def crisis_breakdown(
    returns_by_strategy: dict[str, pd.Series],
    crisis_periods: list[dict],
    risk_free_annual: float = 0.0,
) -> pd.DataFrame:
    """Recompute performance within each crisis window and over the full sample."""
    rows = []
    for strat, daily in returns_by_strategy.items():
        for cp in crisis_periods:
            sl = daily[(daily.index >= pd.Timestamp(cp["start"])) &
                       (daily.index <= pd.Timestamp(cp["end"]))]
            m = summarise_performance(sl, risk_free_annual)
            rows.append({"strategy": strat, "period": cp["name"],
                         "n_days": len(sl), **m})
        m_full = summarise_performance(daily, risk_free_annual)
        rows.append({"strategy": strat, "period": "FULL",
                     "n_days": len(daily), **m_full})
    return pd.DataFrame(rows)


def crisis_breakdown_for_market(name, cfg=None, write_csv=True) -> pd.DataFrame:
    """
    Break a market's backtest down by crisis window.

    Re-aggregates the saved return series rather than re-running the backtest.
    """
    cfg = cfg or load_config()
    crisis_periods = cfg.raw["eda"]["crisis_periods"]
    rf = float(cfg.backtest["risk_free_annual"])

    rpath = cfg.tables_dir / f"backtest_{name}_returns.csv"
    if rpath.exists():
        saved = pd.read_csv(rpath, index_col=0, parse_dates=True)
        returns = {c: saved[c].dropna() for c in saved.columns}
    else:
        print(f"[info] {name}: no saved returns; running the backtest")
        returns = run_backtest_for_index(name, cfg, verbose=False)["returns"]

    df = crisis_breakdown(returns, crisis_periods, rf)
    if write_csv and not df.empty:
        df.to_csv(cfg.tables_dir /
                  f"sensitivity_crisis_breakdown_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    cfg = load_config()
    print("=== Regime-BL view-confidence sweep ===")
    bl = sweep_bl_views(cfg)
    if not bl.empty:
        print(bl.round(3).to_string(index=False))
