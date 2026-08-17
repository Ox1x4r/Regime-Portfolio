"""
Correctness checks on the generated output.

Inspects the actual output files and checks them against invariants that must
hold if the computation is correct: probabilities summing to one, negative
drawdowns, p-values in [0,1], and reported metrics matching a recomputation
from the underlying return series.

Also runs plausibility checks, which are not guaranteed but whose violation
suggests a problem, such as an equally weighted portfolio not tracking its own
index.

    python -m src.validate_results

Exits non-zero if any invariant fails.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .config import Config, load_config
from .metrics import (
    annualised_return, annualised_volatility, sharpe_ratio, max_drawdown,
)


class Report:
    """Collects check outcomes and prints a summary."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes = 0

    def invariant(self, ok: bool, msg: str) -> None:
        if ok:
            self.passes += 1
        else:
            self.failures.append(msg)

    def plausible(self, ok: bool, msg: str) -> None:
        if ok:
            self.passes += 1
        else:
            self.warnings.append(msg)

    def summary(self) -> bool:
        print("\n" + "=" * 68)
        print(" VALIDATION SUMMARY")
        print("=" * 68)
        print(f"  checks passed : {self.passes}")
        print(f"  warnings      : {len(self.warnings)}")
        print(f"  FAILURES      : {len(self.failures)}")
        if self.warnings:
            print("\n  -- warnings (plausibility, not proof of error) --")
            for w in self.warnings:
                print(f"     ! {w}")
        if self.failures:
            print("\n  -- FAILURES (invariant violated) --")
            for f in self.failures:
                print(f"     X {f}")
        print("=" * 68)
        return not self.failures


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_dataset_summary(cfg: Config, rep: Report) -> None:
    path = cfg.tables_dir / "dataset_summary.csv"
    if not path.exists():
        rep.warnings.append("dataset_summary.csv missing")
        return
    df = pd.read_csv(path, index_col=0)
    for idx, r in df.iterrows():
        rep.invariant(r["N_min"] <= r["N_median"] <= r["N_max"],
                      f"{idx}: N_min <= N_median <= N_max violated")
        rep.invariant(r["trading_days_T"] > 0, f"{idx}: T must be positive")
        rep.invariant(0 <= r["pct_panel_missing"] <= 100,
                      f"{idx}: pct_panel_missing outside [0,100]")
        rep.invariant(r["N_max"] <= r["unique_constituents"],
                      f"{idx}: daily N exceeds total unique constituents")
        # High dimensionality is the motivation for shrinkage; flag if absent.
        rep.plausible(r["N_over_T_window"] > 0.5,
                      f"{idx}: N/T={r['N_over_T_window']} unexpectedly low")


def check_survivorship(cfg: Config, rep: Report) -> None:
    path = cfg.tables_dir / "survivorship_by_year.csv"
    if not path.exists():
        rep.warnings.append("survivorship_by_year.csv missing")
        return
    df = pd.read_csv(path)
    rep.invariant((df["entries"] >= 0).all(), "negative entry count")
    rep.invariant((df["exits"] >= 0).all(), "negative exit count")
    rep.invariant((df["net"] == df["entries"] - df["exits"]).all(),
                  "net != entries - exits")
    # Point-in-time membership implies genuine turnover somewhere.
    rep.plausible(df["entries"].sum() > 0 and df["exits"].sum() > 0,
                  "no constituent entries/exits recorded at all")


def check_regime_files(cfg: Config, rep: Report) -> None:
    for spec in cfg.indices:
        name = spec["name"]
        for suffix in ["regimes", "regimes_constituent", "regimes_index"]:
            path = cfg.processed_dir / f"{name}_{suffix}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            pcols = [c for c in df.columns if c.startswith("p_state_")]
            if not pcols:
                continue
            s = df[pcols].sum(axis=1)
            rep.invariant(np.allclose(s.dropna(), 1.0, atol=1e-6),
                          f"{name}/{suffix}: filtered probabilities do not sum to 1")
            rep.invariant((df[pcols].values >= -1e-9).all(),
                          f"{name}/{suffix}: negative state probability")
            rep.invariant(df["state"].between(0, len(pcols) - 1).all(),
                          f"{name}/{suffix}: decoded state out of range")
            # decoded state should usually be the argmax of the filtered vector
            am = df[pcols].values.argmax(axis=1)
            agree = float((am == df["state"].values).mean())
            rep.plausible(agree > 0.5,
                          f"{name}/{suffix}: decoded state matches argmax only "
                          f"{agree:.0%} of the time")


def check_backtest_consistency(cfg: Config, rep: Report) -> None:
    """Recompute every reported metric from the return series and compare."""
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        rpath = cfg.tables_dir / f"backtest_{name}_returns.csv"
        mpath = cfg.tables_dir / f"backtest_{name}_metrics.csv"
        if not (rpath.exists() and mpath.exists()):
            continue
        rets = pd.read_csv(rpath, index_col=0, parse_dates=True)
        mets = pd.read_csv(mpath, index_col=0)
        rf = float(cfg.backtest["risk_free_annual"])

        for strat in mets.index:
            if strat not in rets.columns:
                rep.failures.append(f"{label}/{strat}: metrics without returns")
                continue
            r = rets[strat].dropna()
            if len(r) < 50:
                continue
            # --- invariants: reported metrics must match a recomputation ---
            for col, fn in [
                ("ann_return", lambda x: annualised_return(x)),
                ("ann_vol", lambda x: annualised_volatility(x)),
                ("sharpe", lambda x: sharpe_ratio(x, rf)),
                ("max_drawdown", lambda x: max_drawdown(x)),
            ]:
                if col not in mets.columns:
                    continue
                reported = float(mets.loc[strat, col])
                recomputed = float(fn(r))
                ok = np.isclose(reported, recomputed, rtol=1e-6, atol=1e-9)
                rep.invariant(ok, f"{label}/{strat}: {col} reported "
                                  f"{reported:.6f} != recomputed {recomputed:.6f}")

            # --- invariants: sign and range ---
            rep.invariant(float(mets.loc[strat, "max_drawdown"]) <= 0,
                          f"{label}/{strat}: positive max drawdown")
            rep.invariant(float(mets.loc[strat, "ann_vol"]) >= 0,
                          f"{label}/{strat}: negative volatility")
            if "avg_turnover" in mets.columns:
                t = mets.loc[strat, "avg_turnover"]
                if pd.notna(t):
                    rep.invariant(0.0 <= float(t) <= 1.0 + 1e-9,
                                  f"{label}/{strat}: turnover {t} outside [0,1]")

            # --- plausibility: equity-like magnitudes ---
            vol = float(mets.loc[strat, "ann_vol"])
            shp = float(mets.loc[strat, "sharpe"])
            rep.plausible(0.01 < vol < 1.5,
                          f"{label}/{strat}: implausible annualised vol {vol:.2f}")
            rep.plausible(abs(shp) < 4.0,
                          f"{label}/{strat}: implausible Sharpe {shp:.2f}")

        # --- plausibility: 1/N should track the index it is drawn from ---
        if {"equal_weight", "index_benchmark"} <= set(rets.columns):
            a = rets["equal_weight"].dropna()
            b = rets["index_benchmark"].dropna()
            common = a.index.intersection(b.index)
            if len(common) > 100:
                c = float(np.corrcoef(a.loc[common], b.loc[common])[0, 1])
                rep.plausible(c > 0.7,
                              f"{label}: 1/N vs index correlation only {c:.2f} "
                              "(they hold the same universe, so this should be high)")


def check_inference(cfg: Config, rep: Report) -> None:
    path = cfg.tables_dir / "inference_sharpe_tests.csv"
    if not path.exists():
        rep.warnings.append("inference_sharpe_tests.csv missing")
        return
    df = pd.read_csv(path)
    p = df["p_value"].dropna()
    rep.invariant(((p >= 0) & (p <= 1)).all(), "p-value outside [0,1]")
    if "bh_qvalue" in df.columns:
        both = df[["p_value", "bh_qvalue"]].dropna()
        rep.invariant((both["bh_qvalue"] >= both["p_value"] - 1e-9).all(),
                      "BH q-value smaller than raw p-value")
    if "dsr" in df.columns:
        d = df["dsr"].dropna()
        rep.invariant(((d >= 0) & (d <= 1)).all(), "DSR outside [0,1]")
    if "n_trials" in df.columns:
        rep.invariant((df["n_trials"].dropna() >= 1).all(),
                      "n_trials less than 1")


def check_regime_effect(cfg: Config, rep: Report) -> None:
    """Check the regime-effect table is an exact difference of its two rows."""
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        epath = cfg.tables_dir / f"backtest_{name}_regime_effect.csv"
        mpath = cfg.tables_dir / f"backtest_{name}_metrics.csv"
        if not (epath.exists() and mpath.exists()):
            continue
        eff = pd.read_csv(epath, index_col=0)
        mets = pd.read_csv(mpath, index_col=0)
        for method, row in eff.iterrows():
            for col in ["sharpe", "max_drawdown", "ann_return"]:
                bc, rc, dc = f"{col}_base", f"{col}_rc", f"d_{col}"
                if not {bc, rc, dc} <= set(eff.columns):
                    continue
                rep.invariant(
                    np.isclose(row[dc], row[rc] - row[bc], atol=1e-9),
                    f"{label}/{method}: {dc} != {rc} - {bc}")
                if method in mets.index and col in mets.columns:
                    rep.invariant(
                        np.isclose(row[bc], mets.loc[method, col], atol=1e-9),
                        f"{label}/{method}: {bc} disagrees with metrics table")


def main() -> int:
    cfg = load_config()
    rep = Report()
    print("Validating generated results in", cfg.tables_dir)
    check_dataset_summary(cfg, rep)
    check_survivorship(cfg, rep)
    check_regime_files(cfg, rep)
    check_backtest_consistency(cfg, rep)
    check_regime_effect(cfg, rep)
    check_inference(cfg, rep)
    ok = rep.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
