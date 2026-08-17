"""
Statistical testing of backtest results.

Two problems are handled:

* Sampling uncertainty in a Sharpe difference. Returns are serially dependent
  and heavy-tailed, so the studentised circular block bootstrap 
  is used rather than a naive standard error.
* Multiple testing. The Deflated Sharpe Ratio and Benjamini-Hochberg FDR control are both applied.

Both need the number of configurations tried, which is an explicit argument.
Tests use per-period returns; Sharpes are annualised for display only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe ratio and its HAC standard error (Ledoit & Wolf, 2008)
# ---------------------------------------------------------------------------
def _sharpe(x: np.ndarray) -> float:
    """Per-period Sharpe ratio of an excess-return series."""
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else np.nan


def _hac_cov(V: np.ndarray, lag: int | None = None) -> np.ndarray:
    """Heteroskedasticity- and autocorrelation-consistent covariance matrix.

    Newey-West estimator with a Bartlett kernel. ``V`` is (T, k) of centred
    influence terms; the returned matrix estimates the long-run covariance.
    """
    T, k = V.shape
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    S = (V.T @ V) / T
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = (V[l:].T @ V[:-l]) / T
        S += w * (G + G.T)
    return S


def sharpe_diff_se(r1: np.ndarray, r2: np.ndarray) -> float:
    """HAC standard error of a Sharpe difference, via the delta method."""
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1, s2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    if s1 <= 0 or s2 <= 0:
        return np.nan

    grad = np.array([
        g1 / s1 ** 1.5,             # d(SR1)/d(mu1)
        -g2 / s2 ** 1.5,            # d(SR2)/d(mu2)
        -0.5 * mu1 / s1 ** 1.5,     # d(SR1)/d(gamma1)
        0.5 * mu2 / s2 ** 1.5,      # d(SR2)/d(gamma2)
    ])
    V = np.column_stack([
        r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2,
    ])
    S = _hac_cov(V)
    var = float(grad @ S @ grad) / T
    return float(np.sqrt(var)) if var > 0 else np.nan


def _circular_block_indices(
    T: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Index array for one circular block-bootstrap resample of length T."""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([
        (np.arange(s, s + block) % T) for s in starts
    ])
    return idx[:T]


def block_bootstrap_sharpe_test(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Studentised circular block bootstrap test of a Sharpe difference.

    Tests H0: SR(strategy) = SR(benchmark), two-sided. Blocks preserve serial
    dependence; studentisation makes the bootstrap distribution pivotal.

    Returns the observed Sharpes, their difference, the HAC standard error, a
    p-value and a percentile confidence interval.
    """
    r1 = np.asarray(strategy, dtype=float)
    r2 = np.asarray(benchmark, dtype=float)
    ok = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[ok], r2[ok]
    T = len(r1)
    if T < 50:
        return {"n_obs": T, "p_value": np.nan}

    sr1, sr2 = _sharpe(r1), _sharpe(r2)
    d_hat = sr1 - sr2
    se_hat = sharpe_diff_se(r1, r2)
    ann = np.sqrt(TRADING_DAYS)

    if not np.isfinite(se_hat) or se_hat <= 0:
        # Identical or constant series: no sampling variation to bootstrap.
        return {
            "n_obs": T,
            "sharpe_strategy_ann": sr1 * ann,
            "sharpe_benchmark_ann": sr2 * ann,
            "sharpe_diff_ann": d_hat * ann,
            "hac_se_ann": 0.0,
            "t_stat": np.nan,
            "p_value": 1.0 if abs(d_hat) < 1e-12 else np.nan,
            "ci_low_ann": d_hat * ann,
            "ci_high_ann": d_hat * ann,
        }

    rng = np.random.default_rng(seed)
    t_boot = np.empty(n_boot)
    d_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_indices(T, block, rng)
        b1, b2 = r1[idx], r2[idx]
        d_b = _sharpe(b1) - _sharpe(b2)
        se_b = sharpe_diff_se(b1, b2)
        d_boot[b] = d_b
        t_boot[b] = (d_b - d_hat) / se_b if (
            np.isfinite(se_b) and se_b > 0) else np.nan

    t_obs = d_hat / se_hat
    valid = np.isfinite(t_boot)
    p = float(np.mean(np.abs(t_boot[valid]) >= abs(t_obs))) if valid.any() \
        else np.nan
    lo, hi = np.nanpercentile(d_boot, [2.5, 97.5])

    return {
        "n_obs": T,
        "sharpe_strategy_ann": sr1 * ann,
        "sharpe_benchmark_ann": sr2 * ann,
        "sharpe_diff_ann": d_hat * ann,
        "hac_se_ann": se_hat * ann,
        "t_stat": t_obs,
        "p_value": p,
        "ci_low_ann": lo * ann,
        "ci_high_ann": hi * ann,
    }


# ---------------------------------------------------------------------------
# Multiplicity controls
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum Sharpe over ``n_trials`` under the null of no skill."""
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(max(var_sharpe, 0.0))
    a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_sharpe: float | None = None,
) -> dict:
    """
    Deflated Sharpe Ratio

    Probability that the observed Sharpe exceeds the maximum expected from
    ``n_trials`` trials under the null, corrected for skewness and kurtosis. Above
    0.95 is conventionally read as evidence of skill.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 50:
        return {"dsr": np.nan, "n_obs": T}

    sr = _sharpe(r)
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))   # non-excess

    if var_sharpe is None:
        # Variance of the Sharpe estimator under the observed distribution.
        var_sharpe = (1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr ** 2) / (T - 1)
    sr0 = expected_max_sharpe(n_trials, var_sharpe)

    denom = np.sqrt(max(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    ann = np.sqrt(TRADING_DAYS)
    return {
        "n_obs": T,
        "sharpe_ann": sr * ann,
        "sharpe_threshold_ann": sr0 * ann,
        "n_trials": n_trials,
        "skew": skew,
        "kurtosis": kurt,
        "dsr": float(sps.norm.cdf(z)),
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control. Returns rejection flags and q-values."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    reject = np.zeros_like(p, dtype=bool)
    qvals = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return {"reject": reject, "qvalue": qvals, "n_tests": 0}

    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    # step-up q-values
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    qvals[order] = q
    reject[order] = q <= alpha
    return {"reject": reject, "qvalue": qvals, "n_tests": m}


# ---------------------------------------------------------------------------
# Orchestration over a backtest result
# ---------------------------------------------------------------------------
def inference_table(
    returns_by_strategy: dict[str, pd.Series],
    benchmark: str = "equal_weight",
    n_trials: int | None = None,
    block: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    market: str | None = None,
) -> pd.DataFrame:
    """
    Test every strategy against the benchmark and apply both controls.

    ``n_trials`` defaults to the number of strategies, which is a lower bound;
    raise it to include hyper-parameter configurations.
    """
    if benchmark not in returns_by_strategy:
        raise KeyError(f"benchmark '{benchmark}' not among strategies")
    bench = returns_by_strategy[benchmark]
    names = [s for s in returns_by_strategy if s != benchmark]
    n_trials = n_trials or max(len(returns_by_strategy), 2)

    rows = []
    for s in names:
        a, b = returns_by_strategy[s].align(bench, join="inner")
        test = block_bootstrap_sharpe_test(
            a.values, b.values, block=block, n_boot=n_boot, seed=seed)
        dsr = deflated_sharpe_ratio(a.values, n_trials=n_trials)
        row = {"strategy": s, "benchmark": benchmark}
        if market:
            row["market"] = market
        row.update(test)
        row["dsr"] = dsr.get("dsr", np.nan)
        row["sharpe_threshold_ann"] = dsr.get("sharpe_threshold_ann", np.nan)
        row["n_trials"] = n_trials
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "p_value" in df:
        bh = benjamini_hochberg(df["p_value"].values, alpha=alpha)
        df["bh_qvalue"] = bh["qvalue"]
        df["bh_reject"] = bh["reject"]
    return df


def run_inference(
    cfg=None,
    benchmark: str = "equal_weight",
    n_trials: int | None = None,
    write_csv: bool = True,
) -> pd.DataFrame:
    """
    Test every strategy in every market, controlling FDR across the grid.

    Benjamini-Hochberg is applied once across the full strategy-by-market grid
    rather than per market, since that grid is the family selection occurs over.
    """
    from .config import load_config
    cfg = cfg or load_config()
    inf_cfg = cfg.raw.get("inference", {})
    block = int(inf_cfg.get("block_size", 21))
    n_boot = int(inf_cfg.get("n_bootstrap", 2000))
    alpha = float(inf_cfg.get("fdr_alpha", 0.05))

    frames = []
    for spec in cfg.indices:
        name = spec["name"]
        path = cfg.tables_dir / f"backtest_{name}_returns.csv"
        if not path.exists():
            print(f"[skip] {name}: no backtest returns (run the backtest first)")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if benchmark not in df.columns:
            print(f"[skip] {name}: benchmark '{benchmark}' absent")
            continue
        rets = {c: df[c].dropna() for c in df.columns}
        n_t = n_trials or (len(df.columns) * len(cfg.indices))
        frames.append(inference_table(
            rets, benchmark=benchmark, n_trials=n_t, block=block,
            n_boot=n_boot, alpha=alpha, seed=cfg.seed, market=spec["label"]))
        print(f"[ok]   inference: {name}")

    if not frames:
        return pd.DataFrame()

    grid = pd.concat(frames, ignore_index=True)
    # Re-apply BH across the FULL grid (the correct family).
    bh = benjamini_hochberg(grid["p_value"].values, alpha=alpha)
    grid["bh_qvalue"] = bh["qvalue"]
    grid["bh_reject"] = bh["reject"]
    grid.attrs["n_tests"] = bh["n_tests"]

    if write_csv:
        cfg.tables_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(cfg.tables_dir / "inference_sharpe_tests.csv", index=False)
        print(f"[io]   wrote inference_sharpe_tests.csv "
              f"({bh['n_tests']} tests, FDR alpha={alpha})")
    return grid


if __name__ == "__main__":
    out = run_inference()
    if not out.empty:
        cols = ["market", "strategy", "sharpe_diff_ann", "p_value",
                "bh_qvalue", "bh_reject", "dsr"]
        print(out[cols].round(4).to_string(index=False))
