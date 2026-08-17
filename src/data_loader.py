"""
Raw data loading.

Each source file is JSON keyed by trading date, where each date holds the
index return and a dict of constituent id -> return. Parsed into an index
return series and a wide constituent panel (dates x id), with NaN where a name
was not a member.

Constituent ids are internal to each index and not comparable across them, so
indices are parsed independently.
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path

import pandas as pd

from .config import Config, load_config


# ---------------------------------------------------------------------------
# Single-file parsing
# ---------------------------------------------------------------------------
def _resolve_raw_path(raw_dir: Path, name: str) -> Path | None:
    """Locate an index's raw file, accepting several common naming layouts.

    The data is distributed as bzip2-compressed JSON but is often decompressed
    first, so this checks, in priority order: ``<name>.json.bz2``,
    ``<name>_json.bz2``, ``<name>.json``, ``<name>_json``. Returns the first
    match, or ``None`` if none exists.
    """
    candidates = [
        raw_dir / f"{name}.json.bz2",
        raw_dir / f"{name}_json.bz2",
        raw_dir / f"{name}.json",
        raw_dir / f"{name}_json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_raw_json(path: Path) -> dict:
    """Parse a raw index file, handling both compressed and plain JSON.

    Files ending in ``.bz2`` are decompressed on the fly; everything else is
    read as plain UTF-8 JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_index_file(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Parse one raw file into (index_returns, constituent_panel)."""
    raw = _read_raw_json(path)

    # Index-level series.
    dates = pd.to_datetime(list(raw.keys()), utc=True).tz_convert(None)
    index_returns = pd.Series(
        data=[raw[k]["return"] for k in raw.keys()],
        index=dates,
        name="index_return",
        dtype="float64",
    ).sort_index()

    # Constituent panel. Building from a dict-of-dicts lets pandas align the
    # union of ids and fill gaps with NaN in one vectorised construction.
    ts_by_date = {
        pd.Timestamp(k).tz_localize(None) if pd.Timestamp(k).tzinfo is None
        else pd.Timestamp(k).tz_convert(None): raw[k]["ts"]
        for k in raw.keys()
    }
    constituent_panel = (
        pd.DataFrame.from_dict(ts_by_date, orient="index")
        .sort_index()
        .astype("float64")
    )
    constituent_panel.index.name = "date"
    # Sort columns numerically rather than lexically, so "10" follows "9".
    constituent_panel = constituent_panel[
        sorted(constituent_panel.columns, key=lambda c: (len(str(c)), str(c)))
    ]
    return index_returns, constituent_panel


# ---------------------------------------------------------------------------
# Loading across all indices
# ---------------------------------------------------------------------------
def load_all_indices(
    cfg: Config | None = None,
    write_csv: bool = True,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """
    Parse every configured index, optionally caching to CSV.

    Returns ``{name: {"index": Series, "constituents": DataFrame}}``.
    """
    cfg = cfg or load_config()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for spec in cfg.indices:
        name = spec["name"]
        raw_path = _resolve_raw_path(cfg.raw_dir, name)
        if raw_path is None:
            # Skip rather than fail, so a partial dataset still runs.
            print(f"[skip] {name}: no raw file found in {cfg.raw_dir}")
            continue

        idx_ret, panel = parse_index_file(raw_path)
        results[name] = {"index": idx_ret, "constituents": panel}
        print(
            f"[ok]   {name}: {len(idx_ret):>5} trading days | "
            f"{panel.shape[1]:>5} unique constituents | "
            f"{idx_ret.index.min().date()} -> {idx_ret.index.max().date()}"
        )

        if write_csv:
            idx_ret.to_frame().to_csv(cfg.processed_dir / f"{name}_index.csv")
            panel.to_csv(cfg.processed_dir / f"{name}_constituents.csv")

    if not results:
        raise RuntimeError(
            "No index files were parsed. Check that raw *_json.bz2 files "
            f"exist in {cfg.raw_dir}."
        )
    return results


def load_index_return_series(cfg: Config | None = None) -> pd.DataFrame:
    """
    Load daily index returns for all indices into one DataFrame.

    Reads only the ``return`` field, skipping the constituent dicts, so this is
    fast. Columns align on the union of dates with NaN where a market did not
    trade. Nothing is forward-filled.
    """
    cfg = cfg or load_config()
    series_by_label: dict[str, pd.Series] = {}
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        path = _resolve_raw_path(cfg.raw_dir, name)
        if path is None:
            print(f"[skip] {name}: no raw file found in {cfg.raw_dir}")
            continue
        raw = _read_raw_json(path)
        dates = pd.to_datetime(list(raw.keys()), utc=True).tz_convert(None)
        s = pd.Series(
            [raw[k]["return"] for k in raw.keys()],
            index=dates, name=label, dtype="float64",
        ).sort_index()
        series_by_label[label] = s
    if not series_by_label:
        raise RuntimeError(f"No raw index files found in {cfg.raw_dir}.")
    return pd.DataFrame(series_by_label).sort_index()


if __name__ == "__main__":
    # ``python -m src.data_loader`` parses whatever is present and prints a
    # one-line summary per index.
    load_all_indices()
