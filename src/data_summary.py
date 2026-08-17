"""
Dataset description tables.

``dataset_summary.csv`` gives one row per index: trading days after alignment,
min/median/max constituents, N/T for the estimation window, date range and the
share of missing panel observations.

``survivorship_by_year.csv`` gives active constituents, entries and exits per
calendar year, showing that the investable set is rebuilt point-in-time.

Returns are simple daily total returns in decimal form, in each constituent's
local currency, with no FX conversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config


def dataset_summary(cfg: Config | None = None,
                    write_csv: bool = True) -> pd.DataFrame:
    """One-row-per-index dataset description table."""
    cfg = cfg or load_config()
    lookback = int(cfg.portfolio["default_lookback"])

    rows = []
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        ppath = cfg.processed_dir / f"{name}_constituents_clean.csv"
        mpath = cfg.processed_dir / f"{name}_membership.csv"
        if not ppath.exists() or not mpath.exists():
            print(f"[skip] {name}: cleaned panel/membership not found")
            continue

        panel = pd.read_csv(ppath, index_col=0, parse_dates=True)
        member = pd.read_csv(mpath, index_col=0, parse_dates=True).astype(bool)

        per_day = member.sum(axis=1)
        active_days = per_day[per_day > 0]
        T = int(len(active_days))
        n_min = int(active_days.min()) if T else 0
        n_med = int(active_days.median()) if T else 0
        n_max = int(active_days.max()) if T else 0

        rows.append({
            "index": label,
            "trading_days_T": T,
            "unique_constituents": int(panel.shape[1]),
            "N_min": n_min,
            "N_median": n_med,
            "N_max": n_max,
            "N_over_T_window": round(n_med / lookback, 2),
            "estimation_window_days": lookback,
            "first_date": str(active_days.index.min().date()) if T else "",
            "last_date": str(active_days.index.max().date()) if T else "",
            "pct_panel_missing": round(
                100.0 * float(1.0 - member.values.mean()), 2),
        })

    df = pd.DataFrame(rows).set_index("index")
    if write_csv and not df.empty:
        cfg.tables_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.tables_dir / "dataset_summary.csv")
        print(f"[io]   wrote dataset_summary.csv")
    return df


def survivorship_by_year(cfg: Config | None = None,
                         write_csv: bool = True) -> pd.DataFrame:
    """
    Per-index, per-year constituent counts with entries and exits.

    Entry year is that of first observed membership, exit year that of last.
    Entries in the first sample year and exits in the last are not counted, since
    neither event is observed.
    """
    cfg = cfg or load_config()
    rows = []
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        mpath = cfg.processed_dir / f"{name}_membership.csv"
        if not mpath.exists():
            print(f"[skip] {name}: membership file not found")
            continue
        member = pd.read_csv(mpath, index_col=0, parse_dates=True).astype(bool)
        if member.empty:
            continue

        years = sorted(set(member.index.year))
        first_year, last_year = years[0], years[-1]

        # First and last membership date per constituent.
        any_member = member.any()
        cols = any_member[any_member].index
        first_seen = {}
        last_seen = {}
        for c in cols:
            s = member[c]
            idx = s[s].index
            first_seen[c] = idx.min()
            last_seen[c] = idx.max()

        for y in years:
            in_year = member.loc[member.index.year == y]
            active = int(in_year.any().sum())
            entries = sum(
                1 for c in cols
                if first_seen[c].year == y and y != first_year)
            exits = sum(
                1 for c in cols
                if last_seen[c].year == y and y != last_year)
            rows.append({
                "index": label, "year": y, "active_constituents": active,
                "entries": entries, "exits": exits, "net": entries - exits,
            })

    df = pd.DataFrame(rows)
    if write_csv and not df.empty:
        cfg.tables_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.tables_dir / "survivorship_by_year.csv", index=False)
        print(f"[io]   wrote survivorship_by_year.csv")
    return df


def run_data_summary(cfg: Config | None = None) -> dict:
    """Produce both dataset description tables."""
    cfg = cfg or load_config()
    summary = dataset_summary(cfg)
    surv = survivorship_by_year(cfg)
    return {"summary": summary, "survivorship": surv}


if __name__ == "__main__":
    out = run_data_summary()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n=== DATASET SUMMARY (III-A) ===")
        print(out["summary"].to_string())
        if not out["survivorship"].empty:
            print("\n=== SURVIVORSHIP BY YEAR (III-B), first rows ===")
            print(out["survivorship"].head(20).to_string(index=False))
