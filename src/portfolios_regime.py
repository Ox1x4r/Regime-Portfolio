"""
CVaR minimisation and the equilibrium-anchored allocation.

Two further construction methods, built on the estimation layer in
portfolios.py. Both return long-only, fully invested weights.

* ``cvar``         minimises Conditional Value-at-Risk , 
                   penalising only the loss tail
* ``equilibrium``  blends an equilibrium prior with a view on expected returns;
                   the ``_rc`` variant derives that view from the current
                   regime posterior
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config
from .portfolios import (
    estimate_covariance,
    estimate_expected_returns,
    equal_weight,
    min_variance,
)


# ---------------------------------------------------------------------------
# CVaR minimisation (Rockafellar & Uryasev, 2000)
# ---------------------------------------------------------------------------
def cvar_min(
    window: pd.DataFrame,
    confidence: float = 0.95,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Minimise portfolio CVaR at ``confidence``; long-only, fully invested.

    Solves the Rockafellar-Uryasev linear programme over the window's scenarios.
    ``weights`` supplies regime observation weights as scenario probabilities.
    """
    import cvxpy as cp

    R = window.values                     # (T, N) scenario returns
    T, N = R.shape
    alpha = confidence

    # Scenario probabilities: uniform, or regime observation weights
    # normalised to a probability vector (the regime-conditional variant).
    if weights is None:
        p_scen = np.full(T, 1.0 / T)
    else:
        pw = np.asarray(weights, dtype=float)
        p_scen = pw / pw.sum()

    w = cp.Variable(N)
    zeta = cp.Variable()                  # value-at-risk level
    u = cp.Variable(T, nonneg=True)       # tail-loss auxiliaries

    losses = -(R @ w)
    constraints = [
        u >= losses - zeta,
        cp.sum(w) == 1,
        w >= 0,
    ]
    cvar = zeta + (1.0 / (1.0 - alpha)) * (p_scen @ u)
    prob = cp.Problem(cp.Minimize(cvar), constraints)
    # This is a linear programme, so an LP solver is much faster than the
    # default conic one. Fall back if HIGHS is unavailable.
    try:
        prob.solve(solver="HIGHS")
    except Exception:
        prob.solve()

    if w.value is None:
        return equal_weight(window)
    weights = np.maximum(w.value, 0.0)
    s = weights.sum()
    weights = weights / s if s > 0 else np.full(N, 1.0 / N)
    return pd.Series(weights, index=window.columns)


# ---------------------------------------------------------------------------
# Regime-conditioned Black-Litterman
# ---------------------------------------------------------------------------
def regime_specific_means(
    window: pd.DataFrame,
    states_in_window: pd.Series,
    n_states: int,
    min_days: int,
) -> dict[int, np.ndarray]:
    """
    Mean return per asset within each regime over the window.

    Regimes with fewer than ``min_days`` observations fall back to the overall
    window mean.
    """
    overall = window.mean().values
    out: dict[int, np.ndarray] = {}
    s = states_in_window.reindex(window.index)
    for k in range(n_states):
        mask = (s == k).values
        if mask.sum() >= min_days:
            out[k] = window.loc[mask].mean().values
        else:
            out[k] = overall
    return out


def black_litterman_posterior(
    Sigma: np.ndarray,
    Pi: np.ndarray,
    Q: np.ndarray,
    tau: float,
    view_confidence: float,
) -> np.ndarray:
    """
    Black-Litterman posterior mean with absolute views on every asset (P=I).

    ``tau`` scales prior uncertainty; ``view_confidence`` scales view precision, so
    higher values pull the posterior further from the prior.
    """
    N = Sigma.shape[0]
    tauSig = tau * Sigma
    tauSig_inv = np.linalg.inv(tauSig + 1e-10 * np.eye(N))
    omega_diag = (1.0 / max(view_confidence, 1e-8)) * tau * np.diag(Sigma)
    Omega_inv = np.diag(1.0 / np.maximum(omega_diag, 1e-12))

    A = tauSig_inv + Omega_inv                    # P = I
    b = tauSig_inv @ Pi + Omega_inv @ Q
    return np.linalg.solve(A, b)


def equilibrium_allocation(
    window: pd.DataFrame,
    cfg: Config,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Equilibrium-anchored allocation; long-only, fully invested.

    The classical Black-Litterman prior is reverse-optimised from market-cap
    weights, which this dataset cannot supply. An equal-weight reference portfolio
    is substituted as the anchor, hence the name.

    ``weights=None`` uses unconditional window means as the view; supplying weights
    makes the view regime-conditional.
    """
    import cvxpy as cp

    p = cfg.portfolio
    cov_method = p["covariance_estimator"]
    delta = float(p.get("risk_aversion", 1.0))
    tau = float(p.get("bl_tau", 0.05))
    conf = float(p.get("bl_view_confidence", 1.0))

    Sigma = estimate_covariance(window, cov_method, weights)
    N = Sigma.shape[0]

    # Equilibrium prior from reverse optimisation of the equal-weight reference.
    w_ref = np.full(N, 1.0 / N)
    Pi = delta * Sigma @ w_ref

    # Views: unconditional or regime-weighted expected returns.
    Q = estimate_expected_returns(window, weights)

    mu_bl = black_litterman_posterior(Sigma, Pi, Q, tau, conf)

    w = cp.Variable(N)
    prob = cp.Problem(
        cp.Maximize(mu_bl @ w - 0.5 * delta * cp.quad_form(w, cp.psd_wrap(Sigma))),
        [cp.sum(w) == 1, w >= 0],
    )
    prob.solve()
    if w.value is None:
        return min_variance(window, cov_method, weights)
    out = np.maximum(w.value, 0.0)
    ssum = out.sum()
    out = out / ssum if ssum > 0 else w_ref
    return pd.Series(out, index=window.columns)


# ---------------------------------------------------------------------------
# Regime-context loading (aligns a regime file to a window)
# ---------------------------------------------------------------------------
def load_regime_context(
    name: str, cfg: Config
) -> tuple[pd.Series, pd.DataFrame, int] | None:
    """Load the state path and filtered probabilities for an index."""
    which = cfg.portfolio.get("regime_model", "hmm")
    suffix = "regimes" if which == "hmm" else "msgarch_regimes"
    path = cfg.processed_dir / f"{name}_{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    states = df["state"].astype(int)
    prob_cols = [c for c in df.columns if c.startswith("p_state_")]
    posteriors = df[prob_cols]
    return states, posteriors, len(prob_cols)


def current_posterior(
    posteriors: pd.DataFrame, end_date: pd.Timestamp
) -> np.ndarray:
    """Filtered state-probability vector as of ``end_date``."""
    upto = posteriors.loc[:end_date]
    if upto.empty:
        n = posteriors.shape[1]
        return np.full(n, 1.0 / n)
    return upto.iloc[-1].values


if __name__ == "__main__":
    cfg = load_config()
    pth = cfg.processed_dir / "RUSSELL1000_constituents_clean.csv"
    reg = cfg.processed_dir / "RUSSELL1000_regimes.csv"
    if not pth.exists() or not reg.exists():
        print("Run the preprocessing and regime stages first.")
    else:
        import numpy as np
        from .portfolios import investable_window, regime_observation_weights
        panel = pd.read_csv(pth, index_col=0, parse_dates=True)
        end = panel.index[-1]
        lb = int(cfg.portfolio["default_lookback"])
        win = investable_window(panel, end, lb,
                                float(cfg.portfolio["min_window_coverage"]),
                                cfg.portfolio.get("max_universe"))
        ctx = load_regime_context("RUSSELL1000", cfg)
        states, posteriors, n_states = ctx
        pi = current_posterior(posteriors, end)
        print(f"universe={win.shape[1]} assets | filtered posterior={pi.round(3)}")

        obs_w = regime_observation_weights(win.index, posteriors, pi)
        for tag, w in [
            ("cvar", cvar_min(win, float(cfg.portfolio["cvar_confidence"]))),
            ("cvar_rc", cvar_min(win, float(cfg.portfolio["cvar_confidence"]),
                                 obs_w)),
            ("equilibrium", equilibrium_allocation(win, cfg, None)),
            ("equilibrium_rc", equilibrium_allocation(win, cfg, obs_w)),
        ]:
            print(f"{tag:>15}: sum={w.sum():.4f} "
                  f"nonzero={int((w > 1e-6).sum()):>4} max={w.max():.4f}")
