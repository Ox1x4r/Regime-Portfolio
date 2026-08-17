"""
Pipeline orchestration.

Runs every stage in order. Stages: preprocess, data_summary, eda, dependence,
regimes, msgarch (optional), backtest, inference, sensitivity, figures, tables,
validate.

    python -m src.run_all --jobs 6

A failed stage is logged and skipped rather than aborting the run. Timings are
written to results/run_manifest.txt.
"""
from __future__ import annotations

import argparse
import copy
import time
import traceback
from datetime import datetime
from pathlib import Path

from .config import Config, load_config


# ---------------------------------------------------------------------------
# Stage implementations (thin wrappers around each module's entry point)
# ---------------------------------------------------------------------------
def _stage_preprocess(cfg: Config) -> None:
    from .preprocessing import preprocess
    preprocess(cfg)


def _stage_data_summary(cfg: Config) -> None:
    from .data_summary import run_data_summary
    run_data_summary(cfg)


def _stage_eda(cfg: Config) -> None:
    from .eda import run_eda
    run_eda(cfg)


def _stage_dependence(cfg: Config) -> None:
    from .dependence import run_dependence
    run_dependence(cfg)


def _stage_regimes(cfg: Config, jobs: int = 1) -> None:
    from .regimes import run_regime_detection, run_source_comparison
    run_regime_detection(cfg)
    # Headline comparison: regimes from constituent features vs the index alone.
    run_source_comparison(cfg, workers=jobs)


def _stage_msgarch(cfg: Config) -> None:
    from .msgarch import run_msgarch
    run_msgarch(cfg)


def _stage_backtest(cfg: Config, jobs: int = 1) -> None:
    from .backtest import run_backtest
    run_backtest(cfg, jobs=jobs)


def _stage_inference(cfg: Config) -> None:
    from .inference import run_inference
    run_inference(cfg)


def _stage_figures(cfg: Config) -> None:
    from .figures import run_figures
    run_figures(cfg)


def _stage_tables(cfg: Config) -> None:
    from .tables import run_tables
    run_tables(cfg)


def _stage_validate(cfg: Config) -> None:
    from .validate_results import main as validate_main
    rc = validate_main()
    if rc != 0:
        raise RuntimeError(
            "result validation failed -- see the failures listed above")


def _stage_sensitivity(cfg: Config, full: bool = False,
                       jobs: int = 1) -> None:
    from .sensitivity import (
        sweep_bl_views, crisis_breakdown_for_market,
        sweep_lookback, sweep_transaction_cost, sweep_rebalance,
    )
    # Primary: equilibrium view-confidence sweep on the configured markets.
    sweep_bl_views(cfg, workers=jobs)
    # Crisis sub-period breakdown (re-aggregates the saved backtest returns).
    for spec in cfg.indices:
        panel = cfg.processed_dir / f"{spec['name']}_constituents_clean.csv"
        if panel.exists():
            crisis_breakdown_for_market(spec["name"], cfg)
    # Secondary sweeps (optional; these genuinely re-run the walk-forward loop
    # because they change the weights, so they benefit most from parallelism).
    if full:
        sweep_lookback(cfg, workers=jobs)
        sweep_transaction_cost(cfg)       # analytic: no re-run needed
        sweep_rebalance(cfg, workers=jobs)


# Ordered registry of core stages.
CORE_STAGES = [
    ("preprocess", _stage_preprocess),
    ("data_summary", _stage_data_summary),
    ("eda", _stage_eda),
    ("dependence", _stage_dependence),
    ("regimes", _stage_regimes),
    ("backtest", _stage_backtest),
    ("inference", _stage_inference),
    ("sensitivity", _stage_sensitivity),
    ("figures", _stage_figures),
    ("tables", _stage_tables),
    ("validate", _stage_validate),
]


# ---------------------------------------------------------------------------
# Fast-mode config overrides (for quick orchestration testing)
# ---------------------------------------------------------------------------
def _apply_fast_overrides(cfg: Config) -> Config:
    """Return a lightweight config for a quick smoke run of the whole pipeline."""
    raw = copy.deepcopy(cfg.raw)
    raw["portfolio"]["max_universe"] = 60
    raw["backtest"]["rebalance_frequency"] = "Q"
    raw["backtest"]["regime_refit_n_init"] = 3
    raw["regimes"]["n_init"] = 3
    raw["sensitivity"]["bl_view_confidence_grid"] = [0.1, 1.0]
    return Config(raw=raw)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated stages to run (others skipped)")
    ap.add_argument("--skip", type=str, default=None,
                    help="comma-separated stages to skip")
    ap.add_argument("--with-msgarch", action="store_true",
                    help="also run the MS-GARCH stage (requires R + rpy2)")
    ap.add_argument("--full-sensitivity", action="store_true",
                    help="also run secondary sweeps (lookback, tx-cost, rebalance)")
    ap.add_argument("--fast", action="store_true",
                    help="reduced settings for a quick smoke run")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes for the backtest and the "
                         "sensitivity sweeps (independent jobs; try --jobs 8)")
    args = ap.parse_args()

    cfg = load_config()
    if args.fast:
        cfg = _apply_fast_overrides(cfg)
        print("[run_all] FAST mode: reduced settings (not for final results).")

    # Build the stage list.
    stages = list(CORE_STAGES)
    if args.with_msgarch:
        # insert msgarch right after regimes
        idx = [n for n, _ in stages].index("regimes") + 1
        stages.insert(idx, ("msgarch", _stage_msgarch))

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir = cfg.tables_dir.parent

    print("=" * 68)
    print(" Regime-Aware Portfolio Optimisation -- full pipeline run")
    print(f" started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68)

    timings: list[tuple[str, str, float]] = []
    for name, fn in stages:
        if only is not None and name not in only:
            continue
        if name in skip:
            print(f"\n[skip] stage '{name}' (user-requested)")
            continue
        print(f"\n{'-' * 68}\n[stage] {name}\n{'-' * 68}")
        t0 = time.time()
        try:
            if name == "sensitivity":
                fn(cfg, full=args.full_sensitivity, jobs=args.jobs)
            elif name in ("backtest", "regimes"):
                fn(cfg, jobs=args.jobs)
            else:
                fn(cfg)
            status = "ok"
        except Exception:
            status = "FAILED"
            print(f"[error] stage '{name}' failed:\n{traceback.format_exc()}")
        dt = time.time() - t0
        timings.append((name, status, dt))
        print(f"[done] {name}: {status} in {dt:.1f}s")

    # Manifest
    manifest = results_dir / "run_manifest.txt"
    with manifest.open("w", encoding="utf-8") as fh:
        fh.write("Regime-Aware Portfolio Optimisation -- run manifest\n")
        fh.write(f"timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"fast_mode: {args.fast}\n\n")
        fh.write("stage                status      seconds\n")
        fh.write("-" * 44 + "\n")
        for name, status, dt in timings:
            fh.write(f"{name:<20} {status:<10} {dt:8.1f}\n")

    print("\n" + "=" * 68)
    print(" SUMMARY")
    print("=" * 68)
    for name, status, dt in timings:
        print(f"  {name:<18} {status:<8} {dt:7.1f}s")
    print(f"\n manifest written to {manifest}")
    print("=" * 68)


if __name__ == "__main__":
    main()
