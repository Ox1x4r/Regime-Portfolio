"""
Helpers for running independent jobs in parallel.

Markets and sweep points are independent, so they can run concurrently. All
random draws are seeded from config, so parallel and serial runs give identical
results. Worker functions must be module-level to be picklable on Windows.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence


def suggested_workers(n_jobs: int, requested: int | None = None) -> int:
    """Worker count for ``n_jobs`` tasks, capped by job count and CPU count."""
    cpus = os.cpu_count() or 1
    if requested is not None and requested > 0:
        return max(1, min(int(requested), n_jobs, cpus))
    return max(1, min(n_jobs, cpus))


def parallel_map(
    fn: Callable,
    jobs: Sequence,
    workers: int = 1,
    desc: str = "job",
) -> list:
    """
    Apply ``fn`` to each job, in parallel if ``workers > 1``.

    Falls back to serial execution if a process pool cannot be created. Results
    come back in completion order.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    n = suggested_workers(len(jobs), workers)
    if n <= 1 or len(jobs) == 1:
        return [fn(j) for j in jobs]

    try:
        out = []
        with ProcessPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(fn, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                out.append(fut.result())
                done += 1
                print(f"  [{desc}] {done}/{len(jobs)} complete")
        return out
    except Exception as exc:                     # pragma: no cover
        print(f"  [warn] parallel execution unavailable ({exc}); "
              "falling back to serial")
        return [fn(j) for j in jobs]
