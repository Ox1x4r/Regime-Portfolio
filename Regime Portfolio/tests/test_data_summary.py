"""
Tests for src.data_summary.

Uses a synthetic membership panel with a known entry/exit pattern, so the
survivorship counts can be checked exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_summary import dataset_summary, survivorship_by_year
from src.config import load_config, Config
import copy


@pytest.fixture()
def tiny_project(tmp_path):
    """
    One-index project with a known membership pattern.

    A is present throughout, B enters in year 2, C exits after year 1, and D is
    present only in year 2.
    """
    dates = pd.bdate_range("2010-01-04", "2011-12-30")
    y1 = dates.year == 2010
    y2 = dates.year == 2011

    member = pd.DataFrame(False, index=dates, columns=["A", "B", "C", "D"])
    member["A"] = True
    member.loc[y2, "B"] = True
    member.loc[y1, "C"] = True
    member.loc[y2, "D"] = True

    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 4)), index=dates,
        columns=["A", "B", "C", "D"]).where(member)

    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    panel.to_csv(proc / "TEST_constituents_clean.csv")
    member.to_csv(proc / "TEST_membership.csv")

    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["data"]["indices"] = [{"name": "TEST", "label": "Test Index",
                               "market": "X"}]
    cfg = Config(raw=raw)
    # point the config at the temporary directories
    cfg.__dict__["_proc"] = proc
    type(cfg).processed_dir = property(lambda self: proc)
    type(cfg).tables_dir = property(lambda self: tables)
    return cfg


def test_dataset_summary_counts(tiny_project) -> None:
    df = dataset_summary(tiny_project, write_csv=False)
    row = df.loc["Test Index"]
    assert row["unique_constituents"] == 4
    # 2010 has {A, C} = 2 names/day; 2011 has {A, B, D} = 3 names/day
    assert row["N_min"] == 2
    assert row["N_max"] == 3
    assert row["N_median"] in (2, 3)
    # present cells = 2*n2010 + 3*n2011 out of 4*T -> roughly 37.5% missing
    assert 30.0 < row["pct_panel_missing"] < 45.0


def test_survivorship_entries_and_exits(tiny_project) -> None:
    df = survivorship_by_year(tiny_project, write_csv=False)
    y2010 = df[df["year"] == 2010].iloc[0]
    y2011 = df[df["year"] == 2011].iloc[0]

    # 2010 is the first year: entries are not observable, so must be zero.
    assert y2010["entries"] == 0
    # C's last membership is in 2010 -> one exit that year.
    assert y2010["exits"] == 1
    # B and D first appear in 2011 -> two entries.
    assert y2011["entries"] == 2
    # 2011 is the final year: exits are not observable, so must be zero.
    assert y2011["exits"] == 0
    # active counts: A + C in 2010; A + B + D in 2011
    assert y2010["active_constituents"] == 2
    assert y2011["active_constituents"] == 3


def test_net_change_is_entries_minus_exits(tiny_project) -> None:
    df = survivorship_by_year(tiny_project, write_csv=False)
    assert (df["net"] == df["entries"] - df["exits"]).all()
