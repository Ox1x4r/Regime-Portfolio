"""
Tests for src.parallel.

Parallel execution must change when work happens, never what it produces.
Worker counts are also checked for sensible bounds.
"""
from __future__ import annotations

import os

import pytest

from src.parallel import suggested_workers, parallel_map


def _square(x):
    """Module-level worker (picklable on spawn platforms such as Windows)."""
    return x * x


def test_suggested_workers_never_exceeds_jobs() -> None:
    assert suggested_workers(3, 16) <= 3
    assert suggested_workers(1, 16) == 1
    assert suggested_workers(0, 16) == 1


def test_suggested_workers_never_exceeds_cpus() -> None:
    cpus = os.cpu_count() or 1
    assert suggested_workers(1000, 1000) <= cpus


def test_suggested_workers_respects_request() -> None:
    assert suggested_workers(10, 2) == min(2, os.cpu_count() or 1)


def test_parallel_map_matches_serial() -> None:
    jobs = list(range(12))
    serial = parallel_map(_square, jobs, workers=1)
    par = parallel_map(_square, jobs, workers=4)
    # completion order may differ, so compare as multisets
    assert sorted(serial) == sorted(par)
    assert sorted(serial) == [x * x for x in jobs]


def test_parallel_map_empty_and_single() -> None:
    assert parallel_map(_square, [], workers=4) == []
    assert parallel_map(_square, [5], workers=4) == [25]
