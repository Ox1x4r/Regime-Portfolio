"""
Formatted table generation.

Turns the raw result CSVs into presentation-ready tables, written to
results/paper_tables as Markdown, LaTeX and CSV. Nothing is recomputed.

Strategy and metric names are mapped to display labels (``hrp_rc`` becomes
"HRP (RC)"), percentages formatted and ratios rounded consistently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
STRATEGY_LABELS = {
    "equal_weight": "1/N",
    "index_benchmark": "Index",
    "mean_variance": "MV",
    "mean_variance_rc": "MV (RC)",
    "min_variance": "MinVar",
    "min_variance_rc": "MinVar (RC)",
    "hrp": "HRP",
    "hrp_rc": "HRP (RC)",
    "cvar": "CVaR",
    "cvar_rc": "CVaR (RC)",
    "equilibrium": "Equil.",
    "equilibrium_rc": "Equil. (RC)",
}

METHOD_LABELS = {
    "mean_variance": "Mean-variance",
    "min_variance": "Minimum-variance",
    "hrp": "HRP",
    "cvar": "CVaR",
    "equilibrium": "Equilibrium",
}

# Display order for the strategy tables: benchmarks, then method pairs.
STRATEGY_ORDER = [
    "equal_weight", "index_benchmark",
    "mean_variance", "mean_variance_rc",
    "min_variance", "min_variance_rc",
    "hrp", "hrp_rc",
    "cvar", "cvar_rc",
    "equilibrium", "equilibrium_rc",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _pct(v, dp: int = 1) -> str:
    """Format a decimal fraction as a percentage."""
    if pd.isna(v):
        return "--"
    return f"{100.0 * float(v):.{dp}f}"


def _num(v, dp: int = 2) -> str:
    if pd.isna(v):
        return "--"
    return f"{float(v):.{dp}f}"


def _int(v) -> str:
    if pd.isna(v):
        return "--"
    return f"{int(round(float(v))):,}"


def _write(df: pd.DataFrame, cfg: Config, name: str, caption: str,
           index: bool = False) -> None:
    """Write a formatted table as Markdown, LaTeX and CSV."""
    out = cfg.tables_dir.parent / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / f"{name}.csv", index=index)

    with (out / f"{name}.md").open("w", encoding="utf-8") as fh:
        fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=index))
        fh.write("\n")

    # LaTeX: a complete table environment ready to \input into the template.
    with (out / f"{name}.tex").open("w", encoding="utf-8") as fh:
        fh.write(df.to_latex(index=index, escape=True, caption=caption,
                             label=f"tab:{name}", position="t"))

    print(f"[tab]  {name}  ({df.shape[0]} rows)")


# ---------------------------------------------------------------------------
# Table I -- dataset summary
# ---------------------------------------------------------------------------
def table1_dataset(cfg: Config) -> pd.DataFrame | None:
    p = cfg.tables_dir / "dataset_summary.csv"
    if not p.exists():
        print("[skip] Table I: dataset_summary.csv missing")
        return None
    d = pd.read_csv(p, index_col=0)
    out = pd.DataFrame({
        "Index": d.index,
        "T": [_int(v) for v in d["trading_days_T"]],
        "Unique N": [_int(v) for v in d["unique_constituents"]],
        "N min": [_int(v) for v in d["N_min"]],
        "N med": [_int(v) for v in d["N_median"]],
        "N max": [_int(v) for v in d["N_max"]],
        "N/T": [_num(v, 2) for v in d["N_over_T_window"]],
        "Missing (%)": [_num(v, 1) for v in d["pct_panel_missing"]],
        "First": d["first_date"].values,
        "Last": d["last_date"].values,
    })
    _write(out, cfg, "table1_dataset_summary",
           "Dataset summary after calendar alignment. T is trading days; "
           "N is the daily constituent count; N/T uses median N against the "
           "252-day estimation window; Missing is the non-membership share of "
           "the date-by-constituent panel.")
    return out


# ---------------------------------------------------------------------------
# Table II -- survivorship
# ---------------------------------------------------------------------------
def table2_survivorship(cfg: Config, market_label: str | None = None
                        ) -> pd.DataFrame | None:
    p = cfg.tables_dir / "survivorship_by_year.csv"
    if not p.exists():
        print("[skip] Table II: survivorship_by_year.csv missing")
        return None
    d = pd.read_csv(p)
    if market_label is None:
        # default to the primary market's label
        prim = cfg.raw.get("figures", {}).get("primary_market", "")
        lab = {s["name"]: s["label"] for s in cfg.indices}.get(prim)
        market_label = lab or d["index"].iloc[0]
    sub = d[d["index"] == market_label]
    if sub.empty:
        sub = d[d["index"] == d["index"].iloc[0]]
        market_label = sub["index"].iloc[0]
    out = pd.DataFrame({
        "Year": sub["year"].astype(int).values,
        "Active": [_int(v) for v in sub["active_constituents"]],
        "Entries": [_int(v) for v in sub["entries"]],
        "Exits": [_int(v) for v in sub["exits"]],
        "Net": [_int(v) for v in sub["net"]],
    })
    _write(out, cfg, "table2_survivorship",
           f"Constituent entries and exits by year, {market_label}. Entries "
           "are unobservable in the first sample year and exits in the last, "
           "so both are reported as zero there.")
    # cross-market totals as a compact companion
    tot = d.groupby("index")[["entries", "exits"]].sum().reset_index()
    tot.columns = ["Index", "Total entries", "Total exits"]
    tot["Total entries"] = [_int(v) for v in tot["Total entries"]]
    tot["Total exits"] = [_int(v) for v in tot["Total exits"]]
    _write(tot, cfg, "table2b_survivorship_totals",
           "Total constituent entries and exits over the sample, by index.")
    return out


# ---------------------------------------------------------------------------
# Table III -- stylised facts
# ---------------------------------------------------------------------------
def table3_stylised_facts(cfg: Config) -> pd.DataFrame | None:
    p = cfg.tables_dir / "stylised_facts_index.csv"
    if not p.exists():
        print("[skip] Table III: stylised_facts_index.csv missing")
        return None
    d = pd.read_csv(p, index_col=0)
    out = pd.DataFrame({
        "Index": d.index,
        "Mean (%)": [_pct(v) for v in d.get("mean_ann", np.nan)],
        "Vol (%)": [_pct(v) for v in d.get("vol_ann", np.nan)],
        "Skew": [_num(v) for v in d.get("skew", np.nan)],
        "Ex. kurt.": [_num(v) for v in d.get("excess_kurtosis", np.nan)],
        "JB p": [_num(v, 3) for v in d.get("jarque_bera_p", np.nan)],
        "ADF p": [_num(v, 3) for v in d.get("adf_p", np.nan)],
        "KPSS p": [_num(v, 3) for v in d.get("kpss_p", np.nan)],
        "LB(sq) p": [_num(v, 3) for v in d.get("ljungbox_sq_p", np.nan)],
        "ARCH p": [_num(v, 3) for v in d.get("arch_lm_p", np.nan)],
    })
    _write(out, cfg, "table3_stylised_facts",
           "Stylised facts of daily index returns. Mean and volatility are "
           "annualised. JB is Jarque-Bera; ADF and KPSS are unit-root and "
           "stationarity tests; LB(sq) is Ljung-Box on squared returns; ARCH "
           "is the ARCH-LM test. Lag orders are given in Section III-C.")
    return out


# ---------------------------------------------------------------------------
# Table IV -- regime source comparison
# ---------------------------------------------------------------------------
def table4_regime_source(cfg: Config, market: str | None = None
                         ) -> pd.DataFrame | None:
    market = market or cfg.raw.get("figures", {}).get(
        "primary_market", cfg.indices[0]["name"])
    p = cfg.tables_dir / f"regime_source_comparison_{market}.csv"
    if not p.exists():
        print(f"[skip] Table IV: regime_source_comparison_{market}.csv missing")
        return None
    d = pd.read_csv(p, index_col=0)
    rec = [c for c in d.columns if c.startswith("recall_")]
    out = pd.DataFrame({
        "Feature set": ["Constituent-level" if i == "constituent"
                        else "Index return only" for i in d.index],
        "K": [_int(v) for v in d["K"]],
        "Days": [_int(v) for v in d["n_days"]],
        "Turbulent (%)": [_num(v, 1) for v in d["pct_days_in_crisis_regime"]],
    })
    for c in rec:
        label = c.replace("recall_", "") + " recall (%)"
        out[label] = [_num(v, 1) for v in d[c]]
    _write(out, cfg, "table4_regime_source_comparison",
           "Regime detection from constituent-level cross-sectional features "
           "versus the index return alone. Turbulent is the share of sample "
           "days assigned to the highest-volatility regime; recall is the "
           "share of each crisis window assigned to that regime. Both models "
           "use an identical state grid, selection criterion and seed.")
    return out


# ---------------------------------------------------------------------------
# Table V -- primary-market backtest (Results)
# ---------------------------------------------------------------------------
def _metrics_table(d: pd.DataFrame) -> pd.DataFrame:
    order = [s for s in STRATEGY_ORDER if s in d.index]
    order += [s for s in d.index if s not in order]
    d = d.loc[order]
    return pd.DataFrame({
        "Strategy": [STRATEGY_LABELS.get(s, s) for s in d.index],
        "Return (%)": [_pct(v) for v in d["ann_return"]],
        "Vol (%)": [_pct(v) for v in d["ann_vol"]],
        "Sharpe": [_num(v) for v in d["sharpe"]],
        "Sortino": [_num(v) for v in d["sortino"]],
        "MaxDD (%)": [_pct(v) for v in d["max_drawdown"]],
        "Calmar": [_num(v) for v in d["calmar"]],
        "Turnover": [_num(v) for v in d.get("avg_turnover", np.nan)],
    })


def table5_primary_backtest(cfg: Config, market: str | None = None
                            ) -> pd.DataFrame | None:
    market = market or cfg.raw.get("figures", {}).get(
        "primary_market", cfg.indices[0]["name"])
    label = {s["name"]: s["label"] for s in cfg.indices}.get(market, market)
    p = cfg.tables_dir / f"backtest_{market}_metrics.csv"
    if not p.exists():
        print(f"[skip] Table V: backtest_{market}_metrics.csv missing")
        return None
    d = pd.read_csv(p, index_col=0)
    out = _metrics_table(d)
    _write(out, cfg, "table5_primary_backtest",
           f"Out-of-sample backtest performance, {label} (primary market). "
           "Returns and volatility are annualised and net of transaction "
           "costs. (RC) denotes the regime-conditional variant of the same "
           "method. Turnover is mean one-way turnover per rebalance.")
    return out


# ---------------------------------------------------------------------------
# Table VI -- regime effect across markets (Results)
# ---------------------------------------------------------------------------
def table6_regime_effect(cfg: Config) -> pd.DataFrame | None:
    rows = {}
    for spec in cfg.indices:
        p = cfg.tables_dir / f"backtest_{spec['name']}_regime_effect.csv"
        if p.exists():
            rows[spec["label"]] = pd.read_csv(p, index_col=0)
    if not rows:
        print("[skip] Table VI: no regime-effect tables")
        return None
    methods = [m for m in METHOD_LABELS if any(m in f.index
                                              for f in rows.values())]
    out = pd.DataFrame({"Method": [METHOD_LABELS[m] for m in methods]})
    for label, f in rows.items():
        out[label] = [
            _num(f.loc[m, "d_sharpe"], 3)
            if (m in f.index and "d_sharpe" in f.columns) else "--"
            for m in methods
        ]
    _write(out, cfg, "table6_regime_effect",
           "Within-method regime effect: the change in Sharpe ratio from "
           "conditioning a method's inputs on the prevailing regime "
           "(regime-conditional minus unconditional). Both members of each "
           "pair pass through an identical walk-forward loop, investable "
           "universe and cost model, so the difference is attributable to "
           "regime conditioning alone.")
    return out


# ---------------------------------------------------------------------------
# Table VII -- inference (Results)
# ---------------------------------------------------------------------------
def table7_inference(cfg: Config, market: str | None = None
                     ) -> pd.DataFrame | None:
    p = cfg.tables_dir / "inference_sharpe_tests.csv"
    if not p.exists():
        print("[skip] Table VII: inference_sharpe_tests.csv missing")
        return None
    d = pd.read_csv(p)
    market = market or cfg.raw.get("figures", {}).get(
        "primary_market", cfg.indices[0]["name"])
    label = {s["name"]: s["label"] for s in cfg.indices}.get(market, market)
    if "market" in d.columns and (d["market"] == label).any():
        d = d[d["market"] == label]
    d = d.copy()
    d["_ord"] = d["strategy"].map(
        {s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
    d = d.sort_values("_ord")
    out = pd.DataFrame({
        "Strategy": [STRATEGY_LABELS.get(s, s) for s in d["strategy"]],
        "Sharpe": [_num(v) for v in d.get("sharpe_strategy_ann", np.nan)],
        "vs 1/N": [_num(v, 3) for v in d.get("sharpe_diff_ann", np.nan)],
        "HAC s.e.": [_num(v, 3) for v in d.get("hac_se_ann", np.nan)],
        "p": [_num(v, 3) for v in d.get("p_value", np.nan)],
        "BH q": [_num(v, 3) for v in d.get("bh_qvalue", np.nan)],
        "DSR": [_num(v, 3) for v in d.get("dsr", np.nan)],
    })
    n_trials = int(d["n_trials"].iloc[0]) if "n_trials" in d.columns and len(d) \
        else 0
    _write(out, cfg, "table7_inference",
           f"Tests of the Sharpe-ratio difference against the 1/N benchmark, "
           f"{label}. p-values are from a studentised circular block bootstrap "
           "with a monthly block length; BH q is the Benjamini-Hochberg "
           "adjusted value controlling the false-discovery rate across the "
           f"full strategy-by-market grid; DSR is the Deflated Sharpe Ratio "
           f"with {n_trials} configurations tried.")
    return out


# ---------------------------------------------------------------------------
# Table VIII -- crisis sub-periods (Results)
# ---------------------------------------------------------------------------
def table8_crisis(cfg: Config, market: str | None = None
                  ) -> pd.DataFrame | None:
    market = market or cfg.raw.get("figures", {}).get(
        "primary_market", cfg.indices[0]["name"])
    label = {s["name"]: s["label"] for s in cfg.indices}.get(market, market)
    p = cfg.tables_dir / f"sensitivity_crisis_breakdown_{market}.csv"
    if not p.exists():
        print(f"[skip] Table VIII: crisis breakdown missing for {market}")
        return None
    d = pd.read_csv(p)
    piv = d.pivot_table(index="strategy", columns="period", values="sharpe")
    order = [s for s in STRATEGY_ORDER if s in piv.index]
    piv = piv.loc[order]
    cols = [c for c in piv.columns if c != "FULL"] + \
           (["FULL"] if "FULL" in piv.columns else [])
    out = pd.DataFrame({"Strategy": [STRATEGY_LABELS.get(s, s)
                                     for s in piv.index]})
    for c in cols:
        out["Full sample" if c == "FULL" else c] = [_num(v) for v in piv[c]]
    _write(out, cfg, "table8_crisis_subperiods",
           f"Sharpe ratio within labelled crisis windows and over the full "
           f"sample, {label}. Sub-period Sharpe ratios are computed on the "
           "same daily net return series as the full-sample figures.")
    return out


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------
def tableA1_full_grid(cfg: Config) -> pd.DataFrame | None:
    frames = []
    for spec in cfg.indices:
        p = cfg.tables_dir / f"backtest_{spec['name']}_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, index_col=0)
        t = _metrics_table(d)
        t.insert(0, "Index", spec["label"])
        frames.append(t)
    if not frames:
        print("[skip] Table A1: no metrics tables")
        return None
    out = pd.concat(frames, ignore_index=True)
    _write(out, cfg, "tableA1_full_grid",
           "Full out-of-sample backtest results for every strategy in every "
           "market. Returns and volatility are annualised and net of costs.")
    return out


def tableA2_sensitivity(cfg: Config) -> pd.DataFrame | None:
    frames = []
    specs = [
        ("sensitivity_bl_view.csv", "view_confidence", "View confidence"),
        ("sensitivity_lookback.csv", "lookback", "Lookback (days)"),
        ("sensitivity_txcost.csv", "tc_bps", "Cost (bps)"),
        ("sensitivity_rebalance.csv", "rebalance_freq", "Rebalance"),
    ]
    for fname, col, label in specs:
        p = cfg.tables_dir / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if col not in d.columns:
            continue
        g = d.groupby(["market", col]).agg(
            sharpe=("sharpe", "mean"),
            max_drawdown=("max_drawdown", "mean")).reset_index()
        g.insert(0, "Parameter", label)
        g.columns = ["Parameter", "Index", "Value", "Mean Sharpe", "Mean MaxDD"]
        g["Mean Sharpe"] = [_num(v) for v in g["Mean Sharpe"]]
        g["Mean MaxDD"] = [_pct(v) for v in g["Mean MaxDD"]]
        frames.append(g)
    if not frames:
        print("[skip] Table A2: no sensitivity tables")
        return None
    out = pd.concat(frames, ignore_index=True)
    _write(out, cfg, "tableA2_sensitivity",
           "Hyper-parameter sensitivity. Each row averages across the "
           "strategies run at that parameter value, so the table records the "
           "space explored rather than a preferred configuration.")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_tables(cfg: Config | None = None) -> dict:
    """Generate every formatted table."""
    cfg = cfg or load_config()
    out = {}
    out["table1"] = table1_dataset(cfg)
    out["table2"] = table2_survivorship(cfg)
    out["table3"] = table3_stylised_facts(cfg)
    out["table4"] = table4_regime_source(cfg)
    out["table5"] = table5_primary_backtest(cfg)
    out["table6"] = table6_regime_effect(cfg)
    out["table7"] = table7_inference(cfg)
    out["table8"] = table8_crisis(cfg)
    out["tableA1"] = tableA1_full_grid(cfg)
    out["tableA2"] = tableA2_sensitivity(cfg)
    dest = cfg.tables_dir.parent / "paper_tables"
    print(f"\n[io]   paper tables written to {dest}")
    print("       .md  -> paste into Word")
    print("       .tex -> include in LaTeX")
    print("       .csv -> checking")
    return out


if __name__ == "__main__":
    run_tables()
