"""
Cross-sectional features for regime detection.

Reduces the daily cross-section of constituent returns to five statistics:

* ``xs_dispersion``      cross-sectional standard deviation, in percent
* ``frac_negative``      fraction of names with a negative return
* ``xs_skew``            cross-sectional skewness
* ``avg_pairwise_corr``  mean pairwise correlation over a trailing window
* ``eig1_share``         share of correlation variance in the top eigenvalue

Fitting an HMM to the panel directly is not identifiable, hence the summary.
Every feature at date t uses only data up to t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-day dispersion, sign breadth and skewness."""
    out = pd.DataFrame(index=panel.index)
    # Percentage points, so the scale matches the other features.
    out["xs_dispersion"] = panel.std(axis=1, ddof=1) * 100.0
    out["frac_negative"] = (panel < 0).sum(axis=1) / panel.notna().sum(axis=1)
    out["xs_skew"] = panel.skew(axis=1)
    return out


def _avg_pairwise_correlation(
    panel: pd.DataFrame, window: int
) -> pd.Series:
    """
    Trailing mean pairwise correlation, via the variance-ratio identity.
    """
    V = panel.values
    M = ~np.isnan(V)
    X = np.where(M, V, 0.0)

    # Rolling sums from cumulative sums, O(T*N) overall.
    def _roll_sum(A: np.ndarray) -> np.ndarray:
        C = np.cumsum(A, axis=0)
        out = np.full_like(C, np.nan, dtype=float)
        out[window - 1:] = C[window - 1:]
        out[window:] = C[window:] - C[:-window]
        return out

    n_obs = _roll_sum(M.astype(float))          # observations per name
    s1 = _roll_sum(X)                            # sum of returns
    s2 = _roll_sum(X * X)                        # sum of squared returns

    # Per-name variance, restricted to names with full coverage of the window.
    full = n_obs >= window
    with np.errstate(invalid="ignore", divide="ignore"):
        var_i = (s2 - (s1 ** 2) / np.maximum(n_obs, 1)) / np.maximum(n_obs - 1, 1)
    var_i = np.where(full, var_i, np.nan)
    finite = np.isfinite(var_i)
    sums = np.where(finite, var_i, 0.0).sum(axis=1)
    cnts = finite.sum(axis=1)
    v_bar = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    n_eff = np.nansum(full, axis=1).astype(float)

    # Equally weighted portfolio return per day, then its rolling variance.
    counts = M.sum(axis=1)
    ew = np.where(counts > 0, X.sum(axis=1) / np.maximum(counts, 1), np.nan)
    ew0 = np.nan_to_num(ew)
    e1 = _roll_sum(ew0.reshape(-1, 1)).ravel()
    e2 = _roll_sum((ew0 ** 2).reshape(-1, 1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        var_ew = (e2 - (e1 ** 2) / window) / (window - 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (n_eff * var_ew - v_bar) / ((n_eff - 1.0) * v_bar)
    rho = np.clip(rho, -1.0, 1.0)
    return pd.Series(rho, index=panel.index, name="avg_pairwise_corr")


def _eigenvalue_share(
    panel: pd.DataFrame,
    window: int,
    max_names: int,
    seed: int = 42,
) -> pd.Series:
    """
    Trailing share of correlation variance in the leading eigenvalue.

    Taken from the top singular value of the standardised window, so the
    correlation matrix is never formed. Capped at ``max_names`` to bound SVD cost.
    """
    rng = np.random.default_rng(seed)
    V = panel.values
    M = ~np.isnan(V)
    T = V.shape[0]
    values = np.full(T, np.nan)

    for i in range(window - 1, T):
        rows = slice(i - window + 1, i + 1)
        complete = M[rows].all(axis=0)
        n_complete = int(complete.sum())
        if n_complete < 10:
            continue
        cols = np.flatnonzero(complete)
        if n_complete > max_names:
            cols = rng.choice(cols, size=max_names, replace=False)
        X = V[rows][:, cols]
        sd = X.std(axis=0, ddof=1)
        keep = sd > 0
        if keep.sum() < 10:
            continue
        Z = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
        n = Z.shape[1]
        s1 = np.linalg.svd(Z / np.sqrt(Z.shape[0] - 1), compute_uv=False)[0]
        values[i] = float(s1 ** 2 / n)

    return pd.Series(values, index=panel.index, name="eig1_share")


def constituent_features(
    panel: pd.DataFrame,
    window: int = 63,
    max_names_for_eig: int = 200,
    features: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the feature matrix from a constituent return panel.

    ``window`` sets the trailing window for the correlation-structure features.
    Rows with any missing feature are dropped, removing the warm-up period.
    """
    all_feats = ["xs_dispersion", "frac_negative", "xs_skew",
                 "avg_pairwise_corr", "eig1_share"]
    features = features or all_feats

    parts: list[pd.DataFrame | pd.Series] = []
    daily = _daily_cross_sectional(panel)
    parts.append(daily[[f for f in features if f in daily.columns]])

    if "avg_pairwise_corr" in features:
        parts.append(_avg_pairwise_correlation(panel, window))
    if "eig1_share" in features:
        parts.append(_eigenvalue_share(panel, window, max_names_for_eig, seed))

    feats = pd.concat(parts, axis=1)
    feats = feats[[f for f in features]]        # stable column order
    return feats.replace([np.inf, -np.inf], np.nan).dropna(how="any")
