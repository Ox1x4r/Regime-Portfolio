"""
Cleaning and alignment.

Turns the parsed loader output into analysis-ready panels:

* Calendars are aligned across markets (see ``calendar_alignment``).
* Membership is point-in-time: a name is investable on a date only if it has a
  return there.
* Gaps within a membership span are forward-filled up to
  ``max_forward_fill_days`` (0 disables).
* Returns are winsorised per constituent.

Writes ``<INDEX>_index_clean.csv``, ``<INDEX>_constituents_clean.csv``,
``<INDEX>_membership.csv`` and ``preprocessing_summary.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data_loader import load_all_indices


# ---------------------------------------------------------------------------
# Diagnostics container
# ---------------------------------------------------------------------------
@dataclass
class IndexDiagnostics:
    """Per-index preprocessing diagnostics, aggregated into a summary table."""
    index: str
    n_own_trading_days: int
    n_days_lost_to_alignment: int
    n_trading_days: int
    n_unique_constituents: int
    median_daily_membership: int
    min_daily_membership: int
    max_daily_membership: int
    start: str
    end: str
    pct_cells_missing_in_membership: float
    n_gaps_forward_filled: int
    n_returns_winsorized: int


# ---------------------------------------------------------------------------
# Core cleaning steps
# ---------------------------------------------------------------------------
def build_membership_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask of point-in-time membership: True where a return exists."""
    return panel.notna()


def cap_forward_fill(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    max_days: int,
) -> tuple[pd.DataFrame, int]:
    """
    Forward-fill gaps inside a membership span, up to ``max_days``.

    Leading and trailing non-membership is never filled. Returns the filled panel
    and the number of cells filled.
    """
    if max_days <= 0:
        return panel.copy(), 0

    # Active span per column: between first and last observed date (inclusive).
    first_valid = membership.apply(lambda col: col[col].index.min())
    last_valid = membership.apply(lambda col: col[col].index.max())

    filled = panel.copy()
    n_filled = 0
    for col in filled.columns:
        lo, hi = first_valid[col], last_valid[col]
        if pd.isna(lo) or pd.isna(hi):
            continue  # constituent never observed; nothing to fill
        span = (filled.index >= lo) & (filled.index <= hi)
        sub = filled.loc[span, col]
        before = sub.isna().sum()
        # limit=max_days ensures only short gaps are bridged
        sub_filled = sub.ffill(limit=max_days)
        after = sub_filled.isna().sum()
        n_filled += int(before - after)
        filled.loc[span, col] = sub_filled.values
    return filled, n_filled


def winsorize_returns(
    panel: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> tuple[pd.DataFrame, int]:
    """Clip returns to per-constituent quantiles. Returns (panel, n_clipped)."""
    if lower_q <= 0 and upper_q >= 1:
        return panel.copy(), 0

    lo = panel.quantile(lower_q)
    hi = panel.quantile(upper_q)
    clipped = panel.clip(lower=lo, upper=hi, axis=1)
    # Count only cells that are non-null in both and differ. In pandas
    # NaN != NaN is True, so a naive comparison counts every missing cell.
    both_present = panel.notna() & clipped.notna()
    differs = (clipped != panel) & both_present
    n_clipped = int(differs.values.sum())
    return clipped, n_clipped


def clean_index_series(
    index_returns: pd.Series,
    union_calendar: pd.DatetimeIndex,
    max_ffill: int,
) -> pd.Series:
    """Reindex an index series onto the common calendar and cap-fill gaps."""
    s = index_returns.reindex(union_calendar)
    if max_ffill <= 0:
        # No carry-forward: any date the market did not trade stays NaN.
        return s
    lo, hi = index_returns.index.min(), index_returns.index.max()
    span = (s.index >= lo) & (s.index <= hi)
    s.loc[span] = s.loc[span].ffill(limit=max_ffill)
    return s


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def preprocess(
    cfg: Config | None = None,
    write_csv: bool = True,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Clean and align every index. Returns (cleaned, summary)."""
    cfg = cfg or load_config()
    pp = cfg.preprocessing
    lower_q, upper_q = pp["winsorize_quantiles"]
    max_ffill = int(pp["max_forward_fill_days"])
    start, end = pd.Timestamp(pp["start_date"]), pd.Timestamp(pp["end_date"])
    min_active = int(pp.get("min_active_constituents", 0))

    # Parse the raw files. This also caches the unprocessed CSVs.
    parsed = load_all_indices(cfg, write_csv=False)

    # 1) Common trading calendar, clipped to the window. Intersection keeps
    #    only dates every market traded, so nothing is carried forward.
    alignment = pp.get("calendar_alignment", "intersection")
    date_sets = [set(d["index"].index) for d in parsed.values()]
    if alignment == "intersection":
        common = set.intersection(*date_sets)
    else:
        common = set().union(*date_sets)
    union_calendar = pd.DatetimeIndex(sorted(common))
    union_calendar = union_calendar[
        (union_calendar >= start) & (union_calendar <= end)
    ]
    print(f"[cal]  {alignment} calendar: {len(union_calendar)} dates "
          f"({union_calendar.min().date()} -> {union_calendar.max().date()})")

    # Report how many of its own trading days each index loses under alignment.
    for name, d in parsed.items():
        own = d["index"].index
        own = own[(own >= start) & (own <= end)]
        lost = len(own) - len(set(own) & set(union_calendar))
        pct = 100.0 * lost / max(len(own), 1)
        print(f"[cal]  {name}: {len(own)} own trading days, "
              f"{lost} lost to alignment ({pct:.1f}%)")

    cleaned: dict[str, dict] = {}
    diagnostics: list[IndexDiagnostics] = []
    days_lost: dict[str, tuple[int, int]] = {}
    for name, d in parsed.items():
        own = d["index"].index
        own = own[(own >= start) & (own <= end)]
        days_lost[name] = (
            len(own), len(own) - len(set(own) & set(union_calendar)))

    for name, d in parsed.items():
        idx_ret = d["index"]
        panel = d["constituents"]

        # 2) Reindex constituent panel onto the union calendar (extra dates=NaN).
        panel = panel.reindex(union_calendar)

        # 3) Point-in-time membership mask BEFORE any filling.
        membership = build_membership_mask(panel)

        # 3a) Days with implausibly few active names are holiday artefacts.
        #     Treat them as non-trading: clear membership and blank returns.
        daily_active = membership.sum(axis=1)
        sparse_days = daily_active[
            (daily_active > 0) & (daily_active < min_active)
        ].index
        n_sparse = int(len(sparse_days))
        if n_sparse:
            membership.loc[sparse_days, :] = False
            panel.loc[sparse_days, :] = np.nan
            print(f"[cal]  {name}: dropped {n_sparse} sparse day(s) "
                  f"(<{min_active} active names)")

        # 4) Capped forward-fill of isolated within-membership gaps.
        panel_filled, n_filled = cap_forward_fill(panel, membership, max_ffill)

        # 5) Winsorize extreme returns per constituent.
        panel_clean, n_winz = winsorize_returns(panel_filled, lower_q, upper_q)

        # 6) Clean the index series onto the same calendar.
        idx_clean = clean_index_series(idx_ret, union_calendar, max_ffill)

        cleaned[name] = {
            "index": idx_clean,
            "constituents": panel_clean,
            "membership": membership,
        }

        daily_membership = membership.sum(axis=1)
        active = daily_membership[daily_membership > 0]
        diagnostics.append(IndexDiagnostics(
            index=name,
            n_own_trading_days=int(days_lost[name][0]),
            n_days_lost_to_alignment=int(days_lost[name][1]),
            n_trading_days=int((daily_membership > 0).sum()),
            n_unique_constituents=int(panel.shape[1]),
            median_daily_membership=int(active.median()) if len(active) else 0,
            min_daily_membership=int(active.min()) if len(active) else 0,
            max_daily_membership=int(active.max()) if len(active) else 0,
            start=str(union_calendar.min().date()),
            end=str(union_calendar.max().date()),
            pct_cells_missing_in_membership=round(
                100.0 * (1 - membership.values.mean()), 2),
            n_gaps_forward_filled=int(n_filled),
            n_returns_winsorized=int(n_winz),
        ))
        print(f"[ok]   {name}: median {diagnostics[-1].median_daily_membership} "
              f"names/day | {n_filled} gaps filled | {n_winz} returns winsorized")

    summary = pd.DataFrame([asdict(x) for x in diagnostics]).set_index("index")

    if write_csv:
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, c in cleaned.items():
            c["index"].to_frame("index_return").to_csv(
                cfg.processed_dir / f"{name}_index_clean.csv")
            c["constituents"].to_csv(
                cfg.processed_dir / f"{name}_constituents_clean.csv")
            c["membership"].to_csv(
                cfg.processed_dir / f"{name}_membership.csv")
        summary.to_csv(cfg.processed_dir / "preprocessing_summary.csv")
        print(f"[io]   wrote cleaned panels + summary to {cfg.processed_dir}")

    return cleaned, summary


if __name__ == "__main__":
    _, summary = preprocess()
    print("\n=== preprocessing summary ===")
    with pd.option_context("display.width", 160,
                           "display.max_columns", None):
        print(summary)
