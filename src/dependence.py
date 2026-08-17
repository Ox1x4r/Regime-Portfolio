"""
Cross-market dependence.

Static Pearson and Spearman correlations, a daily-vs-weekly comparison (the
Epps effect under non-synchronous trading), rolling pairwise correlations,
a crisis-versus-calm split, and a two-stage DCC-GARCH model.

Measured on days every market traded, so no filled values enter the estimates.

Writes the ``dependence_*.csv`` tables.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from .config import Config, load_config
from .data_loader import load_index_return_series


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def common_trading_returns(cfg: Config) -> pd.DataFrame:
    """Index returns on days every market traded (list-wise deletion)."""
    df = load_index_return_series(cfg)
    start = pd.Timestamp(cfg.preprocessing["start_date"])
    end = pd.Timestamp(cfg.preprocessing["end_date"])
    df = df.loc[(df.index >= start) & (df.index <= end)]
    common = df.dropna(how="any")
    print(f"[align] {common.shape[0]} common trading days across "
          f"{common.shape[1]} markets "
          f"({common.index.min().date()} -> {common.index.max().date()})")
    return common


# ---------------------------------------------------------------------------
# 1) Static dependence
# ---------------------------------------------------------------------------
def static_correlations(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-period Pearson and Spearman correlation matrices."""
    return returns.corr(method="pearson"), returns.corr(method="spearman")


def weekly_correlations(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix of weekly compounded returns.

    Daily cross-market correlations are biased down when markets trade in
    non-overlapping sessions, since a common shock lands on different calendar
    days. Weekly aggregation spans the offset.
    """
    weekly = (1.0 + returns).resample("W-FRI").prod() - 1.0
    return weekly.dropna(how="any").corr(method="pearson")


def epps_comparison(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-pair daily vs weekly correlation."""
    daily_c = returns.corr(method="pearson")
    weekly_c = weekly_correlations(returns)
    rows = []
    for a, b in combinations(returns.columns, 2):
        d, w = float(daily_c.loc[a, b]), float(weekly_c.loc[a, b])
        rows.append({"pair": f"{a} ~ {b}", "daily_corr": d,
                     "weekly_corr": w, "epps_uplift": w - d})
    return pd.DataFrame(rows).set_index("pair")


# ---------------------------------------------------------------------------
# 2) Rolling dependence
# ---------------------------------------------------------------------------
def rolling_pairwise_correlations(
    returns: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Rolling correlation for every market pair.

    Returns a DataFrame whose columns are ``"A~B"`` pair labels and whose rows
    are dates; each value is the trailing-``window`` correlation of that pair.
    """
    pairs = list(combinations(returns.columns, 2))
    out = {}
    for a, b in pairs:
        out[f"{a} ~ {b}"] = returns[a].rolling(window).corr(returns[b])
    return pd.DataFrame(out, index=returns.index)


def average_rolling_correlation(rolling: pd.DataFrame) -> pd.Series:
    """Cross-pair average rolling correlation (a market-integration series)."""
    return rolling.mean(axis=1).rename("avg_pairwise_corr")


# ---------------------------------------------------------------------------
# 3) Crisis vs calm
# ---------------------------------------------------------------------------
def crisis_vs_calm(
    returns: pd.DataFrame, crisis_periods: list[dict]
) -> pd.DataFrame:
    """Average pairwise correlation inside crisis windows vs the calm rest."""
    def mean_offdiag(sample: pd.DataFrame) -> float:
        if len(sample) < 5:
            return np.nan
        c = sample.corr().values
        iu = np.triu_indices_from(c, k=1)
        return float(np.nanmean(c[iu]))

    rows = []
    crisis_mask = pd.Series(False, index=returns.index)
    for cp in crisis_periods:
        m = (returns.index >= pd.Timestamp(cp["start"])) & \
            (returns.index <= pd.Timestamp(cp["end"]))
        crisis_mask |= m
        rows.append({
            "period": cp["name"],
            "n_days": int(m.sum()),
            "avg_corr": mean_offdiag(returns.loc[m]),
        })
    # Aggregate crisis vs calm.
    rows.append({
        "period": "ALL CRISIS",
        "n_days": int(crisis_mask.sum()),
        "avg_corr": mean_offdiag(returns.loc[crisis_mask]),
    })
    rows.append({
        "period": "CALM (rest)",
        "n_days": int((~crisis_mask).sum()),
        "avg_corr": mean_offdiag(returns.loc[~crisis_mask]),
    })
    df = pd.DataFrame(rows).set_index("period")
    calm = df.loc["CALM (rest)", "avg_corr"]
    df["contagion_vs_calm"] = df["avg_corr"] - calm
    return df


# ---------------------------------------------------------------------------
# 4) DCC-GARCH estimated in two stages
# ---------------------------------------------------------------------------
def _fit_univariate_garch(returns: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Stage 1: fit a GARCH(1,1) to each series; return standardised residuals.

    Returns start as percentage-scaled for numerical stability in ``arch`` and
    are rescaled back internally. Output ``z`` has unit-variance residuals.
    """
    z = {}
    cond_vol = {}
    for col in returns.columns:
        r = returns[col].dropna() * 100.0  # percent scale aids optimisation
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                        dist="normal")
        res = am.fit(disp="off")
        std_resid = res.resid / res.conditional_volatility
        z[col] = pd.Series(std_resid.values, index=r.index)
        cond_vol[col] = pd.Series(res.conditional_volatility.values / 100.0,
                                  index=r.index)
    z_df = pd.DataFrame(z).dropna(how="any")
    return z_df.values, z_df


def _dcc_loglik(params: np.ndarray, z: np.ndarray) -> float:
    """Negative DCC(1,1) quasi log-likelihood (correlation component)."""
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    T, N = z.shape
    Qbar = np.cov(z, rowvar=False)
    Qt = Qbar.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            zt1 = z[t - 1][:, None]
            Qt = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Qt
        dinv = np.diag(1.0 / np.sqrt(np.diag(Qt)))
        Rt = dinv @ Qt @ dinv
        # Guard against numerical non-PD.
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e10
        zt = z[t]  # 1-D vector
        quad = float(zt @ np.linalg.solve(Rt, zt))
        ll += logdet + quad - float(zt @ zt)
    return 0.5 * ll


def fit_dcc(returns: pd.DataFrame) -> dict:
    """Estimate DCC(1,1) and return the time-varying correlation path."""
    _, z_df = _fit_univariate_garch(returns)
    z = z_df.values
    labels = list(z_df.columns)
    T, N = z.shape

    # Estimate (a, b) by QMLE.
    res = minimize(
        _dcc_loglik, x0=np.array([0.02, 0.95]), args=(z,),
        method="L-BFGS-B", bounds=[(1e-6, 0.3), (1e-6, 0.999)],
    )
    a, b = res.x

    # Reconstruct the correlation path at the optimum.
    Qbar = np.cov(z, rowvar=False)
    Qt = Qbar.copy()
    iu = np.triu_indices(N, k=1)
    avg_corr = np.empty(T)
    corr_sum = np.zeros((N, N))
    for t in range(T):
        if t > 0:
            zt1 = z[t - 1][:, None]
            Qt = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Qt
        dinv = np.diag(1.0 / np.sqrt(np.diag(Qt)))
        Rt = dinv @ Qt @ dinv
        avg_corr[t] = np.mean(Rt[iu])
        corr_sum += Rt
    mean_matrix = pd.DataFrame(corr_sum / T, index=labels, columns=labels)

    return {
        "a": float(a),
        "b": float(b),
        "persistence": float(a + b),
        "avg_corr": pd.Series(avg_corr, index=z_df.index, name="dcc_avg_corr"),
        "mean_matrix": mean_matrix,
        "converged": bool(res.success),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_dependence(cfg: Config | None = None, write_csv: bool = True) -> dict:
    """Run the dependence analysis and write the output tables."""
    cfg = cfg or load_config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    window = int(cfg.raw["eda"]["rolling_corr_window"])
    crisis_periods = cfg.raw["eda"]["crisis_periods"]

    returns = common_trading_returns(cfg)

    pearson, spearman = static_correlations(returns)
    print("\n[1] static Pearson correlation:\n", pearson.round(3))

    epps = epps_comparison(returns)
    print("\n[1b] daily vs weekly correlation (Epps-effect check):\n",
          epps.round(3))

    rolling = rolling_pairwise_correlations(returns, window)
    avg_roll = average_rolling_correlation(rolling)
    print(f"\n[2] rolling ({window}d) avg pairwise corr: "
          f"mean={avg_roll.mean():.3f}, min={avg_roll.min():.3f}, "
          f"max={avg_roll.max():.3f}")

    contagion = crisis_vs_calm(returns, crisis_periods)
    print("\n[3] crisis vs calm (correlation contagion):\n",
          contagion.round(3))

    print("\n[4] fitting DCC-GARCH (two-stage)...")
    dcc = fit_dcc(returns)
    print(f"    DCC(1,1): a={dcc['a']:.4f}, b={dcc['b']:.4f}, "
          f"persistence(a+b)={dcc['persistence']:.4f}, "
          f"converged={dcc['converged']}")
    print("    period-average DCC correlation matrix:\n",
          dcc["mean_matrix"].round(3))

    if write_csv:
        td = cfg.tables_dir
        pearson.to_csv(td / "dependence_static_pearson.csv")
        spearman.to_csv(td / "dependence_static_spearman.csv")
        epps.to_csv(td / "dependence_daily_vs_weekly_epps.csv")
        avg_roll.to_frame().to_csv(td / "dependence_rolling_avg.csv")
        contagion.to_csv(td / "dependence_crisis_vs_calm.csv")
        dcc["avg_corr"].to_frame().to_csv(td / "dependence_dcc_avg.csv")
        dcc["mean_matrix"].to_csv(td / "dependence_dcc_mean_matrix.csv")
        print(f"\n[io]   wrote dependence tables to {td}")

    return {
        "returns": returns, "pearson": pearson, "spearman": spearman,
        "epps": epps, "rolling": rolling, "avg_roll": avg_roll,
        "contagion": contagion, "dcc": dcc,
    }


if __name__ == "__main__":
    run_dependence()
