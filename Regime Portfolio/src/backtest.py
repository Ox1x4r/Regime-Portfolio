"""
Walk-forward backtest engine.

Rebalances on the last trading day of each period from ``train_end`` onward.
At each rebalance only data up to that date is used, for both the estimation
window and the regime signal, so the output contains no look-ahead.

Weights set at t apply to the days after t up to the next rebalance.
Transaction costs are charged per unit of one-way turnover.

Writes ``backtest_<INDEX>_metrics.csv``, ``backtest_<INDEX>_returns.csv`` and
``backtest_<INDEX>_regime_effect.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .portfolios import investable_window, build_weights
from .regimes import (
    build_features, build_regime_features, fit_hmm,
    relabel_by_volatility, filtered_probabilities,
)
from .metrics import summarise_performance


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------
def generate_rebalance_dates(
    trading_index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    frequency: str = "M",
) -> list[pd.Timestamp]:
    """Last trading day of each period after ``train_end``."""
    idx = trading_index[trading_index >= train_end]
    if len(idx) == 0:
        return []
    freq_map = {"W": "W", "M": "M", "Q": "Q"}
    period = idx.to_period(freq_map.get(frequency, "M"))
    s = pd.Series(idx, index=period)
    last_per_period = s.groupby(level=0).max()
    return [pd.Timestamp(d) for d in last_per_period.values]


# ---------------------------------------------------------------------------
# Walk-forward regime handler
# ---------------------------------------------------------------------------
class WalkForwardRegimes:
    """
    Refits the HMM on a cadence and serves look-ahead-free regime signals.

    For a rebalance at t, parameters come from the most recent refit on data up to
    t, and the state probability is filtered on data up to t.
    """

    def __init__(self, index_returns: pd.Series, cfg: Config,
                 panel: pd.DataFrame | None = None):
        self.cfg = cfg
        reg = cfg.regimes
        # Computed over the full sample, but each value at t uses only data up
        # to t, so slicing .loc[:t] is look-ahead free.
        self.features_all = build_regime_features(
            cfg, index_returns, panel, reg.get("feature_source", "index"))
        # Kept separately: the constituent feature set has no return column,
        # and states are ordered by index return volatility.
        self.index_returns = index_returns
        self.k_grid = list(reg["n_states_grid"])
        self.cov_type = reg["covariance_type"]
        self.n_iter = int(reg["n_iter"])
        self.n_init = int(cfg.backtest.get("regime_refit_n_init", 5))
        self.seed = cfg.seed
        self.criterion = reg.get("model_selection", "bic").upper()
        self._model = None
        self._mapping = None
        self._last_fit_year = None

    def _select_k_and_fit(self, X: np.ndarray):
        """Fit each candidate K and return the best model by BIC or AIC.

        A given K can occasionally fail to converge on a particular expanding
        window; those are skipped. Returns ``None`` if no K fits, in which case
        the caller keeps the previous model.
        """
        from .regimes import _n_params
        n_obs, n_feat = X.shape
        best = None
        for k in self.k_grid:
            try:
                model, ll = fit_hmm(X, k, self.cov_type, self.n_iter,
                                    self.n_init, self.seed)
            except Exception:
                continue
            p = _n_params(k, n_feat, self.cov_type)
            aic = -2 * ll + 2 * p
            bic = -2 * ll + p * np.log(n_obs)
            score = bic if self.criterion == "BIC" else aic
            if best is None or score < best[0]:
                best = (score, model)
        return best[1] if best is not None else None

    def _maybe_refit(self, t: pd.Timestamp) -> None:
        """Refit if the configured cadence calls for it."""
        cadence = self.cfg.backtest.get("regime_refit_frequency", "annual")
        year = pd.Timestamp(t).year
        need = (
            self._model is None
            or cadence == "each"
            or (cadence == "annual" and year != self._last_fit_year)
        )
        if not need:
            return
        feats = self.features_all.loc[:t]
        if len(feats) < 60:
            return
        model = self._select_k_and_fit(feats.values)
        if model is None:
            # Nothing converged on this window; keep whatever we had.
            return
        raw_states = model.predict(feats.values)
        ret_col = self.index_returns.reindex(feats.index).values
        _, mapping = relabel_by_volatility(model, raw_states, ret_col)
        self._model, self._mapping = model, mapping
        self._last_fit_year = year

    def context_at(
        self, t: pd.Timestamp, window: pd.DataFrame
    ) -> dict | None:
        """Return the regime context for a rebalance at ``t``."""
        self._maybe_refit(t)
        if self._model is None:
            return None
        feats = self.features_all.loc[:t]
        if feats.empty:
            return None
        raw_states = self._model.predict(feats.values)
        # Filtered, not smoothed: an allocation at t may only condition on
        # P(s_t | x_1..x_t). The smoothed posterior uses the whole sample.
        posteriors = filtered_probabilities(self._model, feats.values)
        states = self._mapping[raw_states]
        inv = np.argsort(self._mapping)
        posteriors = posteriors[:, inv]

        state_series = pd.Series(states, index=feats.index)
        states_in_window = state_series.reindex(window.index).ffill().bfill()
        posterior_now = posteriors[-1]
        # The full history is needed to weight observations in the estimation
        # window for the regime-conditional strategies.
        post_df = pd.DataFrame(
            posteriors, index=feats.index,
            columns=[f"p_state_{j}" for j in range(posteriors.shape[1])])
        return {"regime_states": states_in_window.astype(int),
                "regime_posterior": posterior_now,
                "regime_posteriors": post_df}


# ---------------------------------------------------------------------------
# Backtest for a single index
# ---------------------------------------------------------------------------
def _holding_period_returns(
    holding: pd.DataFrame, w: pd.Series
) -> pd.Series:
    """Daily portfolio returns over a holding period at fixed weights."""
    held = holding.reindex(columns=w.index).fillna(0.0)
    daily = held.values @ w.values
    return pd.Series(daily, index=holding.index)


def run_backtest_for_index(
    name: str,
    cfg: Config,
    strategies: list[str] | None = None,
    verbose: bool = True,
    regime_contexts: dict | None = None,
) -> dict:
    """
    Run the walk-forward backtest for one index.

    ``regime_contexts`` optionally supplies precomputed regime contexts by
    rebalance date, so a parameter sweep can reuse one set of regime fits instead
    of refitting per combination.
    """
    panel_path = cfg.processed_dir / f"{name}_constituents_clean.csv"
    index_path = cfg.processed_dir / f"{name}_index_clean.csv"
    if not panel_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"{name}: need cleaned constituent panel and index series (Step 2).")

    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    index_returns = pd.read_csv(
        index_path, index_col=0, parse_dates=True)["index_return"]

    p = cfg.portfolio
    bt = cfg.backtest
    strategies = strategies or list(p["strategies"])
    lookback = int(p["default_lookback"])
    coverage = float(p["min_window_coverage"])
    max_universe = p.get("max_universe")
    tc_rate = float(bt["transaction_cost_bps"]) / 1e4
    base_tc_bps = float(bt["transaction_cost_bps"])
    rf = float(bt["risk_free_annual"])
    tdays = int(bt["trading_days_per_year"])
    train_end = pd.Timestamp(bt["train_end"])

    rebal_dates = generate_rebalance_dates(
        panel.index, train_end, bt["rebalance_frequency"])
    if len(rebal_dates) < 2:
        raise RuntimeError(f"{name}: too few rebalance dates.")

    # Only "_rc" strategies need the regime signal, so skip the work otherwise.
    need_regimes = any(s.endswith("_rc") for s in strategies)
    wf = None
    if need_regimes and regime_contexts is None:
        wf = WalkForwardRegimes(index_returns, cfg, panel)

    acc = {s: {"returns": [], "weights": [], "turnover": []}
           for s in strategies}
    prev_w = {s: None for s in strategies}

    for i in range(len(rebal_dates) - 1):
        t, t_next = rebal_dates[i], rebal_dates[i + 1]
        window = investable_window(panel, t, lookback, coverage, max_universe)
        if window.shape[1] == 0:
            continue
        holding = panel.loc[(panel.index > t) & (panel.index <= t_next)]
        if holding.empty:
            continue

        if not need_regimes:
            context = None
        elif regime_contexts is not None:
            context = regime_contexts.get(t)
        else:
            context = wf.context_at(t, window)

        for s in strategies:
            if s == "index_benchmark":
                # Passive: the index's own return, no turnover and no cost.
                bench = index_returns.reindex(holding.index)
                acc[s]["returns"].append(bench.fillna(0.0))
                continue
            try:
                w = build_weights(s, window, cfg, context)
            except Exception as exc:
                if verbose:
                    print(f"  [warn] {name}/{s} @ {t.date()}: {exc}")
                continue
            if w.empty:
                continue
            # One-way turnover against the previous target weights.
            if prev_w[s] is None:
                turnover = 1.0
            else:
                alla = prev_w[s].index.union(w.index)
                turnover = 0.5 * float(
                    (w.reindex(alla).fillna(0) -
                     prev_w[s].reindex(alla).fillna(0)).abs().sum())
            cost = tc_rate * turnover

            port_daily = _holding_period_returns(holding, w)
            if len(port_daily) > 0:
                port_daily.iloc[0] -= cost
                # Recorded so other cost levels can be priced analytically;
                # see apply_transaction_cost().
                acc[s]["turnover"].append((port_daily.index[0], turnover))
            acc[s]["returns"].append(port_daily)
            acc[s]["weights"].append(w)
            prev_w[s] = w

        if verbose and (i % 12 == 0):
            print(f"  {name}: rebalanced {t.date()} "
                  f"({i + 1}/{len(rebal_dates) - 1})")

    # Assemble per-strategy series and summary metrics.
    out = {"returns": {}, "metrics": {}, "weights": {}, "turnover": {}}
    for s in strategies:
        if not acc[s]["returns"]:
            continue
        daily = pd.concat(acc[s]["returns"]).sort_index()
        daily = daily[~daily.index.duplicated(keep="first")]
        out["returns"][s] = daily
        out["weights"][s] = acc[s]["weights"]
        if acc[s]["turnover"]:
            d, v = zip(*acc[s]["turnover"])
            out["turnover"][s] = pd.Series(v, index=pd.DatetimeIndex(d))
        out["metrics"][s] = summarise_performance(
            daily, rf, acc[s]["weights"], tdays)
    return out


def _backtest_one(args):
    """Worker for parallel execution. Must be module-level to be picklable."""
    name, cfg = args
    try:
        return name, run_backtest_for_index(name, cfg, verbose=False), None
    except Exception as exc:              # pragma: no cover - worker failure
        return name, None, str(exc)


def run_backtest(cfg: Config | None = None, write_csv: bool = True,
                 jobs: int = 1) -> dict:
    """
    Backtest every index and write the summary tables.

    ``jobs > 1`` runs markets in parallel. Output is identical either way, since
    all random draws are seeded.
    """
    cfg = cfg or load_config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

    names = [spec["name"] for spec in cfg.indices
             if (cfg.processed_dir / f"{spec['name']}_constituents_clean.csv").exists()]
    labels = {spec["name"]: spec["label"] for spec in cfg.indices}
    for spec in cfg.indices:
        if spec["name"] not in names:
            print(f"[skip] {spec['name']}: no cleaned panel")

    results: dict[str, dict] = {}
    if jobs and jobs > 1 and len(names) > 1:
        from concurrent.futures import ProcessPoolExecutor
        n_workers = min(int(jobs), len(names))
        print(f"[run]  backtesting {len(names)} markets on {n_workers} workers ...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for name, res, err in ex.map(_backtest_one,
                                         [(n, cfg) for n in names]):
                if err:
                    print(f"[fail] {name}: {err}")
                    continue
                results[name] = res
                print(f"[ok]   {labels[name]}")
    else:
        for name in names:
            print(f"[run]  backtesting {labels[name]} ...")
            results[name] = run_backtest_for_index(name, cfg)

    for name, res in results.items():
        metrics = pd.DataFrame(res["metrics"]).T
        print(f"\n=== {labels[name]} ===")
        print(metrics.round(3).to_string())
        effect = regime_effect_table(metrics)
        if not effect.empty:
            print("\n  within-method regime effect (rc - base):")
            print(effect[[c for c in effect.columns
                          if c.startswith("d_")]].round(3).to_string())
        if write_csv:
            metrics.to_csv(cfg.tables_dir / f"backtest_{name}_metrics.csv")
            pd.DataFrame(res["returns"]).to_csv(
                cfg.tables_dir / f"backtest_{name}_returns.csv")
            if not effect.empty:
                effect.to_csv(
                    cfg.tables_dir / f"backtest_{name}_regime_effect.csv")
    if write_csv:
        print(f"\n[io]   wrote backtest tables to {cfg.tables_dir}")
    return results


if __name__ == "__main__":
    run_backtest()


def regime_effect_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Difference between each method and its regime-conditional variant.

    Both run through the same loop, universe and cost model, differing only in
    whether the input moments are regime-weighted.
    """
    from .portfolios import BASE_METHODS, RC_SUFFIX

    rows = []
    for base in BASE_METHODS:
        rc = base + RC_SUFFIX
        if base not in metrics.index or rc not in metrics.index:
            continue
        row = {"method": base}
        for col in ["sharpe", "sortino", "ann_return", "ann_vol",
                    "max_drawdown", "calmar", "avg_turnover"]:
            if col in metrics.columns:
                row[f"{col}_base"] = metrics.loc[base, col]
                row[f"{col}_rc"] = metrics.loc[rc, col]
                row[f"d_{col}"] = metrics.loc[rc, col] - metrics.loc[base, col]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def apply_transaction_cost(
    gross_or_net: pd.Series,
    turnover: pd.Series,
    from_bps: float,
    to_bps: float,
) -> pd.Series:
    """
    Re-price a return series from one cost level to another.

    Weights do not depend on the cost rate, so only the per-rebalance charge
    changes. Given the recorded turnover the new series follows exactly, which
    makes a cost sweep free rather than a full re-run.
    """
    out = gross_or_net.copy()
    delta = (to_bps - from_bps) / 1e4
    if delta == 0 or turnover is None or turnover.empty:
        return out
    common = out.index.intersection(turnover.index)
    out.loc[common] = out.loc[common] - delta * turnover.loc[common].values
    return out
