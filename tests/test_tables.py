"""
Tests for src.tables.

The formatting helpers can silently corrupt a published number, so they are
tested directly. Table builders are checked for graceful degradation when
inputs are absent.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, Config
from src.tables import (
    _pct, _num, _int, STRATEGY_LABELS, METHOD_LABELS, STRATEGY_ORDER,
    table1_dataset, table5_primary_backtest, table6_regime_effect,
)


def test_pct_formats_decimal_fraction() -> None:
    assert _pct(0.1234) == "12.3"
    assert _pct(-0.3456) == "-34.6"
    assert _pct(np.nan) == "--"


def test_num_and_int() -> None:
    assert _num(0.876543) == "0.88"
    assert _num(0.876543, 3) == "0.877"
    assert _num(np.nan) == "--"
    assert _int(3564) == "3,564"
    assert _int(np.nan) == "--"


def test_strategy_labels_cover_the_factorial() -> None:
    from src.portfolios import factorial_strategies
    for s in factorial_strategies():
        assert s in STRATEGY_LABELS, f"no publication label for {s}"


def test_strategy_order_covers_all_labels() -> None:
    assert set(STRATEGY_ORDER) == set(STRATEGY_LABELS)


def test_rc_labels_are_distinguishable() -> None:
    """A method and its regime-conditional variant must not share a label."""
    for m in METHOD_LABELS:
        assert STRATEGY_LABELS[m] != STRATEGY_LABELS[m + "_rc"]


def test_tables_degrade_gracefully(tmp_path) -> None:
    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)
    empty = Config(raw=raw)
    type(empty).tables_dir = property(lambda self: tmp_path / "tables")
    (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
    assert table1_dataset(empty) is None
    assert table5_primary_backtest(empty) is None
    assert table6_regime_effect(empty) is None
