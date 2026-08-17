"""
Portfolio construction: estimation layer and core methods.

Methods operate on a trailing window of returns for the investable universe and
return long-only, fully invested weights:

* ``equal_weight``   1/N benchmark
* ``min_variance``   global minimum variance, convex QP
* ``mean_variance``  maximum Sharpe or maximum utility
* ``hrp``            Hierarchical Risk Parity (Lopez de Prado, 2016)

Assets can outnumber observations at constituent level, making the sample
covariance singular, so Ledoit-Wolf shrinkage is the default estimator.

CVaR and the equilibrium-anchored allocation are in portfolios_regime.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from .config import Config, load_config


# ---------------------------------------------------------------------------
# Investable universe & input estimation
# ---------------------------------------------------------------------------
def investable_window(
    panel: pd.DataFrame,
    end_date: pd.Timestamp,
    lookback: int,
    min_coverage: float = 1.0,
    max_universe: int | None = None,
) -> pd.DataFrame:
    """
    Trailing window of returns for the investable universe at ``end_date``.

    An asset qualifies if present for at least ``min_coverage`` of the window. Uses
    only data up to ``end_date``.
    """
    hist = panel.loc[:end_date]
    window = hist.iloc[-lookback:]
    if len(window) < lookback:
        return pd.DataFrame(index=window.index)  # not enough history yet

    coverage = window.notna().mean()
    eligible = coverage[coverage >= min_coverage].index
    window = window[eligible]

    if max_universe is not None and window.shape[1] > max_universe:
        # keep the names with the most complete history in the window
        fullest = window.notna().sum().sort_values(ascending=False)
        window = window[fullest.index[:max_universe]]

    # At min_coverage = 1.0 the window is already complete. Below that, gaps
    # are zero-filled, which biases variance downward, so warn rather than do
    # it silently.
    n_missing = int(window.isna().sum().sum())
    if n_missing:
        print(f"[warn] investable_window @ {pd.Timestamp(end_date).date()}: "
              f"imputing {n_missing} missing returns as 0 "
              f"(min_coverage={min_coverage}); this biases variance downward.")
    return window.fillna(0.0)


def regime_observation_weights(
    window_index: pd.DatetimeIndex,
    regime_posteriors: pd.DataFrame,
    current_posterior: np.ndarray,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Weight each observation by its resemblance to the current regime.

    Observation s gets w_s = sum_k pi_k * P(s_s = k), where pi is the current
    filtered posterior, normalised to sum to the window length. Days like today
    dominate; others are downweighted rather than dropped.

    This is the only channel through which conditioning enters.
    """
    P = regime_posteriors.reindex(window_index)
    # Dates without a regime estimate fall back to the unconditional weight.
    missing = P.isna().any(axis=1).values
    Pv = np.nan_to_num(P.values, nan=0.0)
    w = Pv @ np.asarray(current_posterior, dtype=float)
    w = np.where(missing, float(np.mean(current_posterior ** 2)), w)
    w = np.maximum(w, floor)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(window_index))
    return w * (len(window_index) / total)


def estimate_covariance(
    window: pd.DataFrame,
    method: str = "ledoit_wolf",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Covariance of the window, optionally weighted.

    With weights, Ledoit-Wolf is fitted on observations scaled by sqrt(w) so the
    shrinkage intensity is computed on the reweighted sample.
    """
    X = window.values
    if weights is None:
        if method == "ledoit_wolf":
            return LedoitWolf().fit(X).covariance_
        if method == "sample":
            return np.cov(X, rowvar=False)
        raise ValueError(f"Unknown covariance method: {method}")

    w = np.asarray(weights, dtype=float)
    w = w * (len(w) / w.sum())
    mu = (w[:, None] * X).sum(axis=0) / w.sum()
    Xc = X - mu
    if method == "sample":
        return (Xc * w[:, None]).T @ Xc / max(w.sum() - 1.0, 1.0)
    if method == "ledoit_wolf":
        Xw = Xc * np.sqrt(w)[:, None]
        return LedoitWolf(assume_centered=True).fit(Xw).covariance_
    raise ValueError(f"Unknown covariance method: {method}")


def estimate_expected_returns(
    window: pd.DataFrame, weights: np.ndarray | None = None
) -> np.ndarray:
    """Window mean returns, optionally regime-weighted."""
    if weights is None:
        return window.mean().values
    w = np.asarray(weights, dtype=float)
    return (w[:, None] * window.values).sum(axis=0) / w.sum()


# ---------------------------------------------------------------------------
# Strategy: equal weight
# ---------------------------------------------------------------------------
def equal_weight(window: pd.DataFrame) -> pd.Series:
    """1/N over the investable universe."""
    assets = window.columns
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=assets)


# ---------------------------------------------------------------------------
# Strategy: minimum variance (convex QP, long-only)
# ---------------------------------------------------------------------------
def min_variance(window: pd.DataFrame, cov_method: str = "ledoit_wolf",
                 weights: np.ndarray | None = None) -> pd.Series:
    """Global minimum-variance portfolio; long-only, fully invested."""
    import cvxpy as cp

    S = estimate_covariance(window, cov_method, weights)
    n = S.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))),
        [cp.sum(w) == 1, w >= 0],
    )
    prob.solve()
    weights = _clean_weights(w.value, n)
    return pd.Series(weights, index=window.columns)


# ---------------------------------------------------------------------------
# Strategy: mean-variance (max Sharpe or max utility, long-only)
# ---------------------------------------------------------------------------
def mean_variance(
    window: pd.DataFrame,
    cov_method: str = "ledoit_wolf",
    objective: str = "max_sharpe",
    risk_aversion: float = 1.0,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """Mean-variance optimal portfolio; long-only, fully invested."""
    import cvxpy as cp

    S = estimate_covariance(window, cov_method, weights)
    mu = estimate_expected_returns(window, weights)
    n = S.shape[0]

    if objective == "max_utility":
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(S))),
            [cp.sum(w) == 1, w >= 0],
        )
        prob.solve()
        return pd.Series(_clean_weights(w.value, n), index=window.columns)

    # max_sharpe: only meaningful if some expected return is positive.
    if np.all(mu <= 0):
        # fall back to minimum variance when no asset has positive expected return
        return min_variance(window, cov_method)

    # Schur reformulation: maximise Sharpe via a convex QP in y, then rescale.
    y = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(S))),
        [mu @ y == 1, y >= 0],
    )
    prob.solve()
    if y.value is None:
        return min_variance(window, cov_method, weights)
    w = np.maximum(y.value, 0)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    return pd.Series(w, index=window.columns)


# ---------------------------------------------------------------------------
# Strategy: Hierarchical Risk Parity (from first principles)
# ---------------------------------------------------------------------------
def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """Leaf order that quasi-diagonalises the linkage matrix."""
    link = link.astype(int)
    n = link[-1, 3]  # total number of original items
    sort_ix = [link[-1, 0], link[-1, 1]]
    # iteratively replace cluster ids (>= n_items) by their children
    n_items = link.shape[0] + 1
    changed = True
    while changed:
        changed = False
        new = []
        for item in sort_ix:
            if item >= n_items:
                row = link[item - n_items]
                new.append(int(row[0]))
                new.append(int(row[1]))
                changed = True
            else:
                new.append(int(item))
        sort_ix = new
    return sort_ix


def _hrp_recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """
    Allocate by recursive bisection along the quasi-diagonal order.

    Uses numpy indexing; pandas label indexing dominates the runtime otherwise.
    """
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [np.asarray(sort_ix, dtype=int)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) > 1:
                half = len(c) // 2
                nxt.append(c[:half])
                nxt.append(c[half:])
        clusters = nxt
        for i in range(0, len(clusters) - 1, 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            denom = v0 + v1
            alpha = 1.0 - v0 / denom if denom > 0 else 0.5
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def _cluster_var(cov: np.ndarray, items: np.ndarray) -> float:
    """Inverse-variance-weighted variance of a cluster."""
    sub = cov[np.ix_(items, items)]
    d = np.diag(sub)
    ivp = 1.0 / np.where(d > 0, d, np.inf)
    total = ivp.sum()
    if total <= 0:
        ivp = np.full(len(items), 1.0 / len(items))
    else:
        ivp = ivp / total
    return float(ivp @ sub @ ivp)


def hrp(window: pd.DataFrame, cov_method: str = "ledoit_wolf",
        weights: np.ndarray | None = None) -> pd.Series:
    """Hierarchical Risk Parity weights (Lopez de Prado, 2016)."""
    S = estimate_covariance(window, cov_method, weights)
    cols = list(window.columns)
    # correlation and the HRP distance metric d = sqrt((1 - rho) / 2)
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    np.fill_diagonal(dist, 0.0)

    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _hrp_quasi_diag(link)
    w = _hrp_recursive_bisection(S, sort_ix)
    total = w.sum()
    if total > 0:
        w = w / total
    return pd.Series(w, index=cols)


# ---------------------------------------------------------------------------
# Dispatcher & helpers
# ---------------------------------------------------------------------------
def _clean_weights(w: np.ndarray | None, n: int, tol: float = 1e-6) -> np.ndarray:
    """Clip tiny negatives and renormalise to sum to one."""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    w[np.abs(w) < tol] = 0.0
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


# Base construction methods and their regime-conditional counterparts.
BASE_METHODS = ["mean_variance", "min_variance", "hrp", "cvar", "equilibrium"]
RC_SUFFIX = "_rc"


def factorial_strategies(include_benchmarks: bool = True) -> list[str]:
    """The full strategy list: five methods x two conditions, plus benchmarks."""
    out = []
    for m in BASE_METHODS:
        out.append(m)
        out.append(m + RC_SUFFIX)
    if include_benchmarks:
        out = ["equal_weight", "index_benchmark"] + out
    return out


def parse_strategy(strategy: str) -> tuple[str, bool]:
    """Split a strategy name into (base method, regime_conditional)."""
    if strategy.endswith(RC_SUFFIX):
        return strategy[: -len(RC_SUFFIX)], True
    return strategy, False


def build_weights(
    strategy: str,
    window: pd.DataFrame,
    cfg: Config,
    context: dict | None = None,
) -> pd.Series:
    """
    Dispatch to a named strategy.

    Names are a base method optionally suffixed ``_rc`` for the regime-conditional
    variant, plus ``equal_weight``. ``_rc`` strategies need ``context`` with
    ``regime_posteriors`` and ``regime_posterior``.
    """
    from .portfolios_regime import cvar_min, equilibrium_allocation

    p = cfg.portfolio
    cov_method = p["covariance_estimator"]
    if window.shape[1] == 0:
        return pd.Series(dtype=float)

    if strategy == "equal_weight":
        return equal_weight(window)

    base, conditional = parse_strategy(strategy)

    obs_w = None
    if conditional:
        if not context or context.get("regime_posteriors") is None:
            raise ValueError(
                f"'{strategy}' requires context with 'regime_posteriors' and "
                "'regime_posterior'.")
        obs_w = regime_observation_weights(
            window.index,
            context["regime_posteriors"],
            context["regime_posterior"],
            floor=float(p.get("regime_weight_floor", 0.0)),
        )

    if base == "min_variance":
        return min_variance(window, cov_method, obs_w)
    if base == "mean_variance":
        return mean_variance(window, cov_method,
                             p.get("mean_variance_objective", "max_sharpe"),
                             float(p.get("risk_aversion", 1.0)), obs_w)
    if base == "hrp":
        return hrp(window, cov_method, obs_w)
    if base == "cvar":
        return cvar_min(window, float(p.get("cvar_confidence", 0.95)), obs_w)
    if base == "equilibrium":
        return equilibrium_allocation(window, cfg, obs_w)
    raise NotImplementedError(f"Unknown strategy '{strategy}'.")


if __name__ == "__main__":
    # Demonstration on the Russell universe if the cleaned panel is present.
    cfg = load_config()
    path = cfg.processed_dir / "RUSSELL1000_constituents_clean.csv"
    if not path.exists():
        print("Run Steps 1-2 first to produce cleaned constituent panels.")
    else:
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        end = panel.index[-1]
        lb = int(cfg.portfolio["default_lookback"])
        win = investable_window(panel, end, lb,
                                float(cfg.portfolio["min_window_coverage"]),
                                cfg.portfolio.get("max_universe"))
        print(f"investable universe at {end.date()}: {win.shape[1]} assets, "
              f"{win.shape[0]} days")
        for strat in ["equal_weight", "min_variance", "mean_variance", "hrp"]:
            w = build_weights(strat, win, cfg)
            print(f"{strat:>14}: sum={w.sum():.4f} | nonzero={int((w>1e-6).sum()):>4} "
                  f"| max={w.max():.4f} | top5={w.sort_values(ascending=False).head(5).sum():.3f}")
