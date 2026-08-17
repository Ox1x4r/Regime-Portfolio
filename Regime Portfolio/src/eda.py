"""
Stylised facts of the return series.

Computes distributional moments, normality tests (Jarque-Bera, KS),
stationarity tests (ADF, KPSS), autocorrelation (Ljung-Box) and volatility
clustering (ARCH-LM), at index and constituent level.

Constituent-level moments use every name; the regression-based tests use a
seeded sample of ``eda.max_constituents_for_tests`` names per market, since
they are expensive per series.

Writes ``stylised_facts_index.csv`` and
``stylised_facts_constituents_summary.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from .config import Config, load_config

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Single-series battery
# ---------------------------------------------------------------------------
def series_stylised_facts(
    r: pd.Series,
    lb_lags: int,
    arch_lags: int,
    include_tests: bool = True,
) -> dict[str, float]:
    """
    Compute the test battery for one return series.

    ``include_tests=False`` returns moment-based statistics only, which is the fast
    path used when screening the full constituent panel.
    """
    x = r.dropna().values
    n = x.size
    out: dict[str, float] = {"n_obs": float(n)}
    if n < 20:  # too short to say anything meaningful
        return out

    # -- moments -----------------------------------------------------------
    out["mean_ann"] = float(np.mean(x) * TRADING_DAYS)
    out["vol_ann"] = float(np.std(x, ddof=1) * np.sqrt(TRADING_DAYS))
    out["skew"] = float(sps.skew(x, bias=False))
    out["excess_kurtosis"] = float(sps.kurtosis(x, fisher=True, bias=False))

    # -- normality ---------------------------------------------------------
    jb_stat, jb_p = sps.jarque_bera(x)
    out["jarque_bera"] = float(jb_stat)
    out["jarque_bera_p"] = float(jb_p)
    # KS against a normal fitted to the sample (EDA diagnostic).
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if sd > 0:
        ks_stat, ks_p = sps.kstest(x, "norm", args=(mu, sd))
        out["ks_stat"] = float(ks_stat)
        out["ks_p"] = float(ks_p)

    if not include_tests:
        return out

    # -- stationarity ------------------------------------------------------
    try:
        adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
        out["adf_stat"] = float(adf_stat)
        out["adf_p"] = float(adf_p)
    except Exception:
        pass
    try:
        # KPSS null = stationary; "c" = level stationarity.
        kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto")[:2]
        out["kpss_stat"] = float(kpss_stat)
        out["kpss_p"] = float(kpss_p)
    except Exception:
        pass

    # -- autocorrelation & volatility clustering ---------------------------
    try:
        lb = acorr_ljungbox(x, lags=[lb_lags], return_df=True)
        out["ljungbox_ret"] = float(lb["lb_stat"].iloc[0])
        out["ljungbox_ret_p"] = float(lb["lb_pvalue"].iloc[0])
        lb2 = acorr_ljungbox(x ** 2, lags=[lb_lags], return_df=True)
        out["ljungbox_sq"] = float(lb2["lb_stat"].iloc[0])
        out["ljungbox_sq_p"] = float(lb2["lb_pvalue"].iloc[0])
    except Exception:
        pass
    try:
        arch_stat, arch_p = het_arch(x, nlags=arch_lags)[:2]
        out["arch_lm"] = float(arch_stat)
        out["arch_lm_p"] = float(arch_p)
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Index-level analysis
# ---------------------------------------------------------------------------
def index_level_facts(cfg: Config) -> pd.DataFrame:
    """Compute the full battery for each index return series."""
    lb = int(cfg.raw["eda"]["ljung_box_lags"])
    arch = int(cfg.raw["eda"]["arch_lm_lags"])
    rows = {}
    for spec in cfg.indices:
        name = spec["name"]
        path = cfg.processed_dir / f"{name}_index_clean.csv"
        if not path.exists():
            print(f"[skip] {name}: cleaned index file not found")
            continue
        s = pd.read_csv(path, index_col=0, parse_dates=True)["index_return"]
        rows[spec["label"]] = series_stylised_facts(s, lb, arch)
        print(f"[ok]   index facts: {name}")
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Constituent-level analysis
# ---------------------------------------------------------------------------
def _panel_moment_facts(panel: pd.DataFrame) -> pd.DataFrame:
    """Vectorised moment-based stats for every constituent column."""
    facts = pd.DataFrame(index=panel.columns)
    facts["n_obs"] = panel.notna().sum()
    facts["mean_ann"] = panel.mean() * TRADING_DAYS
    facts["vol_ann"] = panel.std(ddof=1) * np.sqrt(TRADING_DAYS)
    facts["skew"] = panel.skew()
    facts["excess_kurtosis"] = panel.kurtosis()  # pandas = excess kurtosis
    return facts


def constituent_level_summary(cfg: Config) -> pd.DataFrame:
    """Cross-sectional summary of constituent statistics, per market."""
    lb = int(cfg.raw["eda"]["ljung_box_lags"])
    arch = int(cfg.raw["eda"]["arch_lm_lags"])
    min_hist = int(cfg.preprocessing["min_constituent_history"])
    max_test = int(cfg.raw["eda"]["max_constituents_for_tests"])
    rng = np.random.default_rng(cfg.seed)

    summaries = {}
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        path = cfg.processed_dir / f"{name}_constituents_clean.csv"
        if not path.exists():
            print(f"[skip] {name}: cleaned constituents file not found")
            continue
        panel = pd.read_csv(path, index_col=0, parse_dates=True)

        # Keep constituents with enough history.
        enough = panel.notna().sum() >= min_hist
        panel = panel.loc[:, enough[enough].index]
        n_names = panel.shape[1]

        # Moment-based (all qualifying names, vectorised).
        moments = _panel_moment_facts(panel)

        # Regression-based tests on a seeded sample.
        cols = list(panel.columns)
        if len(cols) > max_test:
            cols = list(rng.choice(cols, size=max_test, replace=False))
        frac_nonstationary = frac_arch = frac_nonnormal = 0.0
        n_tested = 0
        for c in cols:
            f = series_stylised_facts(panel[c], lb, arch, include_tests=True)
            if "adf_p" in f:
                n_tested += 1
                # ADF: p>0.05 => fail to reject unit root => non-stationary.
                frac_nonstationary += (f.get("adf_p", 0) > 0.05)
                frac_arch += (f.get("arch_lm_p", 1) < 0.05)
                frac_nonnormal += (f.get("jarque_bera_p", 1) < 0.05)
        if n_tested:
            frac_nonstationary /= n_tested
            frac_arch /= n_tested
            frac_nonnormal /= n_tested

        summaries[label] = {
            "n_constituents": n_names,
            "median_vol_ann": float(moments["vol_ann"].median()),
            "median_skew": float(moments["skew"].median()),
            "median_excess_kurt": float(moments["excess_kurtosis"].median()),
            "iqr_excess_kurt": float(
                moments["excess_kurtosis"].quantile(0.75)
                - moments["excess_kurtosis"].quantile(0.25)),
            "n_tested": n_tested,
            "frac_nonstationary": round(frac_nonstationary, 3),
            "frac_arch_effects": round(frac_arch, 3),
            "frac_nonnormal": round(frac_nonnormal, 3),
        }
        print(f"[ok]   constituent facts: {name} "
              f"({n_names} names, {n_tested} tested)")
    return pd.DataFrame(summaries).T


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_eda(cfg: Config | None = None, write_csv: bool = True):
    """Compute the stylised facts and optionally write the summary tables."""
    cfg = cfg or load_config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)

    print("--- index-level stylised facts ---")
    idx_facts = index_level_facts(cfg)
    print("\n--- constituent-level stylised facts ---")
    con_summary = constituent_level_summary(cfg)

    if write_csv:
        idx_facts.to_csv(cfg.tables_dir / "stylised_facts_index.csv")
        con_summary.to_csv(
            cfg.tables_dir / "stylised_facts_constituents_summary.csv")
        print(f"\n[io]   wrote stylised-facts tables to {cfg.tables_dir}")
    return idx_facts, con_summary


if __name__ == "__main__":
    idx_facts, con_summary = run_eda()
    with pd.option_context("display.width", 200,
                           "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print("\n=== INDEX-LEVEL STYLISED FACTS ===")
        print(idx_facts)
        print("\n=== CONSTITUENT-LEVEL SUMMARY ===")
        print(con_summary)
