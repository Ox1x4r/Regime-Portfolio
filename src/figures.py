"""
Figure generation.

Renders figures from the saved tables and series; nothing is recomputed here.
Each is written to results/figures as both PDF and PNG, sized for a two-column
layout (``COL_WIDTH`` for one column, ``FULL_WIDTH`` to span both).

Figures: regime source comparison, constituent features, cumulative wealth,
regime effect, drawdown, view-confidence sensitivity, cross-market
correlation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .config import Config, load_config

# Two-column layout widths, in inches.
COL_WIDTH = 3.5
FULL_WIDTH = 7.16

# Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]
REGIME_COLOURS = ["#BFD9EC", "#F5C99B", "#E8A0A0"]   # calm -> turbulent


def _style() -> None:
    """Apply the shared plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
    })


def _save(fig, cfg: Config, name: str) -> None:
    out = cfg.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"[fig]  {name}.pdf / .png")


def _shade_crises(ax, cfg: Config, label_once: bool = True,
                  outline: bool = False) -> None:
    """Mark crisis windows. ``outline=True`` draws a hatched band for dense plots."""
    first = True
    for cp in cfg.raw["eda"]["crisis_periods"]:
        lo, hi = pd.Timestamp(cp["start"]), pd.Timestamp(cp["end"])
        lbl = "Crisis window" if (first and label_once) else None
        if outline:
            ax.axvspan(lo, hi, facecolor="none", edgecolor="black",
                       hatch="///", lw=0.9, alpha=0.9, zorder=5, label=lbl)
        else:
            ax.axvspan(lo, hi, color="grey", alpha=0.18, lw=0, label=lbl)
        first = False


def _pct(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


# ---------------------------------------------------------------------------
# Figure 1 -- headline: constituent vs index regime detection
# ---------------------------------------------------------------------------
def fig_regime_source_comparison(cfg: Config, market: str) -> bool:
    con_p = cfg.processed_dir / f"{market}_regimes_constituent.csv"
    idx_p = cfg.processed_dir / f"{market}_regimes_index.csv"
    cmp_p = cfg.tables_dir / f"regime_source_comparison_{market}.csv"
    if not (con_p.exists() and idx_p.exists()):
        print(f"[skip] fig1: regime source files missing for {market}")
        return False

    con = pd.read_csv(con_p, index_col=0, parse_dates=True)
    idx = pd.read_csv(idx_p, index_col=0, parse_dates=True)

    fig = plt.figure(figsize=(FULL_WIDTH, 4.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.45, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    for ax, df, title in (
        (ax1, con, "(a) Regimes from constituent-level features"),
        (ax2, idx, "(b) Regimes from the index return alone"),
    ):
        states = df["state"].astype(int)
        k = int(states.max()) + 1
        # Shade each day by its decoded regime.
        for s in range(k):
            mask = (states == s).values
            ax.fill_between(df.index, 0, 1, where=mask,
                            color=REGIME_COLOURS[min(s, len(REGIME_COLOURS) - 1)],
                            step="mid", lw=0,
                            label=f"Regime {s}" if ax is ax1 else None)
        _shade_crises(ax, cfg, label_once=(ax is ax1), outline=True)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(title, loc="left")
    ax1.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.12))
    ax2.set_xlabel("Date")

    # Crisis recall comparison.
    if cmp_p.exists():
        cmp = pd.read_csv(cmp_p, index_col=0)
        rec_cols = [c for c in cmp.columns if c.startswith("recall_")]
        if rec_cols:
            labels = [c.replace("recall_", "") for c in rec_cols]
            x = np.arange(len(labels))
            w = 0.38
            for i, (src, colour) in enumerate(
                    [("constituent", PALETTE[0]), ("index", PALETTE[1])]):
                if src in cmp.index:
                    ax3.bar(x + (i - 0.5) * w, cmp.loc[src, rec_cols].values,
                            w, label=src, color=colour)
            ax3.set_xticks(x)
            ax3.set_xticklabels([l.replace(" ", "\n") for l in labels])
            ax3.set_ylabel("Crisis days in turbulent regime (%)")
            ax3.set_title("(c) Crisis recall", loc="left")
            ax3.legend()
    _save(fig, cfg, "fig1_regime_source_comparison")
    return True


# ---------------------------------------------------------------------------
# Figure 2 -- constituent cross-sectional features
# ---------------------------------------------------------------------------
def fig_constituent_features(cfg: Config, market: str) -> bool:
    ppath = cfg.processed_dir / f"{market}_constituents_clean.csv"
    if not ppath.exists():
        print(f"[skip] fig2: panel missing for {market}")
        return False
    from .constituent_features import constituent_features

    panel = pd.read_csv(ppath, index_col=0, parse_dates=True)
    reg = cfg.regimes
    feats = constituent_features(
        panel, window=int(reg.get("constituent_window", 63)),
        max_names_for_eig=int(reg.get("max_names_for_eig", 200)),
        features=list(reg.get("constituent_features")), seed=cfg.seed)

    names = {
        "xs_dispersion": "Cross-sectional dispersion (%)",
        "frac_negative": "Fraction with negative return",
        "xs_skew": "Cross-sectional skewness",
        "avg_pairwise_corr": "Average pairwise correlation",
        "eig1_share": "Leading eigenvalue share",
    }
    # Raw series drawn faintly with a rolling mean overlaid. Presentational
    # only; the model is fitted on the raw features.
    smooth = int(cfg.regimes.get("vol_window", 21))
    cols = [c for c in feats.columns if c in names]
    fig, axes = plt.subplots(len(cols), 1,
                             figsize=(FULL_WIDTH, 1.05 * len(cols)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c, colour in zip(axes, cols, PALETTE):
        ax.plot(feats.index, feats[c], color=colour, lw=0.4, alpha=0.30)
        ax.plot(feats.index, feats[c].rolling(smooth).mean(),
                color=colour, lw=1.0)
        _shade_crises(ax, cfg, label_once=(ax is axes[0]))
        # Panel titles avoid the label collisions that stacked y-labels cause.
        ax.set_title(names[c], loc="left", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
    axes[0].legend(loc="upper right", fontsize=6.5)
    axes[-1].set_xlabel("Date")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    _save(fig, cfg, "fig2_constituent_features")
    return True


# ---------------------------------------------------------------------------
# Figure 3 -- cumulative wealth
# ---------------------------------------------------------------------------
def fig_cumulative_wealth(cfg: Config, market: str,
                          strategies: list[str] | None = None) -> bool:
    rpath = cfg.tables_dir / f"backtest_{market}_returns.csv"
    if not rpath.exists():
        print(f"[skip] fig3: returns missing for {market}")
        return False
    rets = pd.read_csv(rpath, index_col=0, parse_dates=True)
    strategies = strategies or [
        c for c in ["equal_weight", "index_benchmark", "hrp", "hrp_rc",
                    "cvar", "cvar_rc", "min_variance_rc"] if c in rets.columns]

    # Fixed colour per strategy, with a method and its _rc variant sharing a
    # hue (solid vs dashed).
    colours = {
        "equal_weight": "#000000", "index_benchmark": "#666666",
        "hrp": "#009E73", "hrp_rc": "#009E73",
        "cvar": "#D55E00", "cvar_rc": "#D55E00",
        "min_variance": "#0072B2", "min_variance_rc": "#0072B2",
        "mean_variance": "#CC79A7", "mean_variance_rc": "#CC79A7",
        "equilibrium": "#E69F00", "equilibrium_rc": "#E69F00",
    }
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    for s in strategies:
        r = rets[s].dropna()
        style = "--" if s.endswith("_rc") else "-"
        ax.plot(r.index, (1 + r).cumprod(), style,
                color=colours.get(s, "#333333"), lw=1.0, label=s)
    _shade_crises(ax, cfg)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_xlabel("Date")
    ax.legend(ncol=4, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, cfg, "fig3_cumulative_wealth")
    return True


# ---------------------------------------------------------------------------
# Figure 4 -- within-method regime effect across markets
# ---------------------------------------------------------------------------
def fig_regime_effect(cfg: Config) -> bool:
    frames = {}
    for spec in cfg.indices:
        p = cfg.tables_dir / f"backtest_{spec['name']}_regime_effect.csv"
        if p.exists():
            frames[spec["label"]] = pd.read_csv(p, index_col=0)
    if not frames:
        print("[skip] fig4: no regime-effect tables")
        return False

    methods = sorted({m for f in frames.values() for m in f.index})
    x = np.arange(len(methods))
    n = len(frames)
    w = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.9))
    for i, (label, f) in enumerate(frames.items()):
        vals = [f.loc[m, "d_sharpe"] if (m in f.index and "d_sharpe" in f.columns)
                else np.nan for m in methods]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=label,
               color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in methods])
    ax.set_ylabel(r"$\Delta$ Sharpe (conditional $-$ uncond.)", fontsize=7)
    ax.legend(ncol=3, fontsize=6.5)
    fig.tight_layout()
    _save(fig, cfg, "fig4_regime_effect")
    return True


# ---------------------------------------------------------------------------
# Figure 5 -- drawdown (underwater) curves
# ---------------------------------------------------------------------------
def fig_drawdown(cfg: Config, market: str,
                 strategies: list[str] | None = None) -> bool:
    rpath = cfg.tables_dir / f"backtest_{market}_returns.csv"
    if not rpath.exists():
        print(f"[skip] fig5: returns missing for {market}")
        return False
    rets = pd.read_csv(rpath, index_col=0, parse_dates=True)
    strategies = strategies or [
        c for c in ["equal_weight", "hrp", "hrp_rc", "cvar", "cvar_rc"]
        if c in rets.columns]

    colours = {
        "equal_weight": "#000000", "index_benchmark": "#666666",
        "hrp": "#009E73", "hrp_rc": "#009E73",
        "cvar": "#D55E00", "cvar_rc": "#D55E00",
        "min_variance": "#0072B2", "min_variance_rc": "#0072B2",
    }
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    for s in strategies:
        r = rets[s].dropna()
        wealth = (1 + r).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        style = "--" if s.endswith("_rc") else "-"
        ax.plot(dd.index, dd, style, color=colours.get(s, "#333333"),
                lw=0.9, label=s)
    _shade_crises(ax, cfg)
    _pct(ax)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(ncol=3)
    _save(fig, cfg, "fig5_drawdown")
    return True


# ---------------------------------------------------------------------------
# Figure 6 -- view-confidence sensitivity
# ---------------------------------------------------------------------------
def fig_view_confidence(cfg: Config) -> bool:
    p = cfg.tables_dir / "sensitivity_bl_view.csv"
    if not p.exists():
        print("[skip] fig6: view sweep missing")
        return False
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    for i, (market, g) in enumerate(df.groupby("market")):
        g = g.sort_values("view_confidence")
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(g["view_confidence"], g["sharpe"], "o-", color=c,
                     label=market, ms=3)
        axes[1].plot(g["view_confidence"], g["max_drawdown"], "o-", color=c,
                     label=market, ms=3)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("View confidence")
    axes[1].set_xlabel("View confidence")
    axes[0].set_ylabel("Sharpe ratio")
    axes[1].set_ylabel("Maximum drawdown")
    _pct(axes[1])
    axes[0].legend(fontsize=6.5)
    _save(fig, cfg, "fig6_view_confidence")
    return True


# ---------------------------------------------------------------------------
# Figure 7 -- cross-market correlation
# ---------------------------------------------------------------------------
def fig_cross_market_correlation(cfg: Config) -> bool:
    roll_p = cfg.tables_dir / "dependence_rolling_avg.csv"
    dcc_p = cfg.tables_dir / "dependence_dcc_avg.csv"
    if not roll_p.exists():
        print("[skip] fig7: dependence series missing")
        return False
    roll = pd.read_csv(roll_p, index_col=0, parse_dates=True).iloc[:, 0]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    ax.plot(roll.index, roll, color=PALETTE[0], lw=0.9,
            label="Rolling average pairwise correlation")
    if dcc_p.exists():
        dcc = pd.read_csv(dcc_p, index_col=0, parse_dates=True).iloc[:, 0]
        ax.plot(dcc.index, dcc, color=PALETTE[1], lw=0.9, alpha=0.85,
                label="DCC-GARCH average correlation")
    _shade_crises(ax, cfg)
    ax.set_ylabel("Cross-market correlation")
    ax.set_xlabel("Date")
    ax.legend()
    _save(fig, cfg, "fig7_cross_market_correlation")
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_figures(cfg: Config | None = None,
                primary_market: str | None = None) -> None:
    """Generate every figure. ``primary_market`` defaults to the first index."""
    cfg = cfg or load_config()
    _style()
    primary_market = primary_market or cfg.raw.get(
        "figures", {}).get("primary_market") or cfg.indices[0]["name"]
    print(f"[fig]  primary market: {primary_market}")

    fig_regime_source_comparison(cfg, primary_market)
    fig_constituent_features(cfg, primary_market)
    fig_cumulative_wealth(cfg, primary_market)
    fig_regime_effect(cfg)
    fig_drawdown(cfg, primary_market)
    fig_view_confidence(cfg)
    fig_cross_market_correlation(cfg)
    print(f"[io]   figures written to {cfg.figures_dir}")


if __name__ == "__main__":
    run_figures()
