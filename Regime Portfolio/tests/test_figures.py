"""
Tests for src.figures.

Figures render from saved results, so these check that each function returns
False when its inputs are missing rather than raising, and that the styling
helpers behave.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from src.config import load_config
from src.figures import (
    _style, COL_WIDTH, FULL_WIDTH, PALETTE,
    fig_regime_source_comparison, fig_cumulative_wealth,
    fig_regime_effect, fig_view_confidence,
)


def test_layout_constants_are_two_column_sane() -> None:
    assert COL_WIDTH < FULL_WIDTH
    assert 3.0 < COL_WIDTH < 4.0
    assert 6.5 < FULL_WIDTH < 7.5


def test_palette_is_distinct() -> None:
    assert len(PALETTE) == len(set(PALETTE))
    assert all(c.startswith("#") for c in PALETTE)


def test_style_applies_without_error() -> None:
    _style()
    import matplotlib.pyplot as plt
    assert plt.rcParams["savefig.dpi"] == 300


def test_figures_degrade_gracefully_on_missing_inputs(tmp_path) -> None:
    """A missing input must skip the figure, not crash the pipeline."""
    import copy
    from src.config import Config

    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)
    empty = Config(raw=raw)
    # Point at empty directories so no input file exists.
    type(empty).tables_dir = property(lambda self: tmp_path / "tables")
    type(empty).processed_dir = property(lambda self: tmp_path / "processed")
    type(empty).figures_dir = property(lambda self: tmp_path / "figures")

    assert fig_regime_source_comparison(empty, "NOPE") is False
    assert fig_cumulative_wealth(empty, "NOPE") is False
    assert fig_regime_effect(empty) is False
    assert fig_view_confidence(empty) is False
