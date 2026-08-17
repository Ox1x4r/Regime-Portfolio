"""
Tests for src.inference.

Checks the size and power of the bootstrap test by simulation, the
Benjamini-Hochberg procedure against a known case, and that the Deflated Sharpe
Ratio decreases as the number of trials rises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    _sharpe,
    sharpe_diff_se,
    block_bootstrap_sharpe_test,
    benjamini_hochberg,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    inference_table,
)


def test_sharpe_basic() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 5000)
    # per-period Sharpe ~ 0.05
    assert _sharpe(r) == pytest.approx(0.05, abs=0.03)


def test_hac_se_finite_and_positive() -> None:
    rng = np.random.default_rng(1)
    r1 = rng.normal(0.0006, 0.01, 1500)
    r2 = rng.normal(0.0003, 0.01, 1500)
    se = sharpe_diff_se(r1, r2)
    assert np.isfinite(se) and se > 0


def test_bootstrap_has_power_against_clear_difference() -> None:
    """A large genuine Sharpe difference must be detected."""
    rng = np.random.default_rng(11)
    n = 4000
    # A large effect: per-period SR ~ 0.25 versus ~ 0.00.
    good = rng.normal(0.0025, 0.01, n)
    poor = rng.normal(0.0000, 0.01, n)
    out = block_bootstrap_sharpe_test(good, poor, block=21, n_boot=400, seed=0)
    assert out["sharpe_diff_ann"] > 0
    assert out["p_value"] < 0.10


def test_bootstrap_does_not_reject_identical_series() -> None:
    """With the same series on both sides the difference is exactly zero."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, 1200)
    out = block_bootstrap_sharpe_test(r, r.copy(), block=21, n_boot=300, seed=0)
    assert out["sharpe_diff_ann"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_value"] > 0.20


def test_bootstrap_size_under_null() -> None:
    """
    Rejection rate under a true null should stay modest.

    Two independent draws from the same distribution are compared repeatedly. The
    bound is loose so the test itself is not flaky.
    """
    rejects = 0
    trials = 12
    for i in range(trials):
        rng = np.random.default_rng(100 + i)
        a = rng.normal(0.0004, 0.01, 800)
        b = rng.normal(0.0004, 0.01, 800)
        out = block_bootstrap_sharpe_test(a, b, block=21, n_boot=200, seed=i)
        if np.isfinite(out.get("p_value", np.nan)) and out["p_value"] < 0.05:
            rejects += 1
    assert rejects <= trials // 2


def test_benjamini_hochberg_known_case() -> None:
    # Classic BH example: with m=4 and alpha=0.05
    p = np.array([0.001, 0.008, 0.039, 0.041])
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["n_tests"] == 4
    # q-values must be non-decreasing in p and bounded by 1
    q = out["qvalue"]
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(q <= 1.0)
    # the smallest p must be rejected
    assert out["reject"][0]


def test_benjamini_hochberg_rejects_nothing_when_all_large() -> None:
    out = benjamini_hochberg(np.array([0.4, 0.6, 0.9]), alpha=0.05)
    assert not out["reject"].any()


def test_expected_max_sharpe_increases_with_trials() -> None:
    v = 0.001
    assert expected_max_sharpe(100, v) > expected_max_sharpe(10, v)
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_decreases_with_more_trials() -> None:
    """Trying more configurations must make the same Sharpe less impressive."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.0008, 0.01, 2000)
    few = deflated_sharpe_ratio(r, n_trials=2)
    many = deflated_sharpe_ratio(r, n_trials=500)
    assert few["dsr"] > many["dsr"]
    assert many["sharpe_threshold_ann"] > few["sharpe_threshold_ann"]


def test_inference_table_structure() -> None:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=800)
    rets = {
        "equal_weight": pd.Series(rng.normal(0.0004, 0.01, 800), index=idx),
        "good": pd.Series(rng.normal(0.0010, 0.01, 800), index=idx),
        "poor": pd.Series(rng.normal(0.0000, 0.01, 800), index=idx),
    }
    df = inference_table(rets, benchmark="equal_weight", n_boot=200,
                         market="TEST")
    assert set(df["strategy"]) == {"good", "poor"}
    for col in ["p_value", "bh_qvalue", "bh_reject", "dsr", "n_trials",
                "sharpe_diff_ann", "market"]:
        assert col in df.columns
    # q-values must be at least as large as raw p-values
    ok = df["p_value"].notna() & df["bh_qvalue"].notna()
    assert (df.loc[ok, "bh_qvalue"] >= df.loc[ok, "p_value"] - 1e-12).all()
