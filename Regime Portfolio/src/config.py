"""
Configuration loading.

Reads config.yaml and resolves paths relative to the project root.

    >>> cfg = load_config()
    >>> cfg.backtest["transaction_cost_bps"]
    10.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# This file lives in ``src/``, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    """Parsed configuration.

    Each top-level YAML section is exposed as a dictionary attribute, and the
    directory properties return absolute paths.
    """

    raw: dict[str, Any] = field(repr=False)

    data: dict[str, Any] = field(init=False)
    preprocessing: dict[str, Any] = field(init=False)
    regimes: dict[str, Any] = field(init=False)
    portfolio: dict[str, Any] = field(init=False)
    backtest: dict[str, Any] = field(init=False)
    seed: int = field(init=False)

    def __post_init__(self) -> None:
        self.data = self.raw["data"]
        self.preprocessing = self.raw["preprocessing"]
        self.regimes = self.raw["regimes"]
        self.portfolio = self.raw["portfolio"]
        self.backtest = self.raw["backtest"]
        self.seed = int(self.raw.get("seed", 42))

    # -- path helpers -------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.data["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.data["processed_dir"]

    @property
    def figures_dir(self) -> Path:
        return PROJECT_ROOT / self.raw.get("figures_dir", "results/figures")

    @property
    def tables_dir(self) -> Path:
        return PROJECT_ROOT / self.raw.get("tables_dir", "results/tables")

    @property
    def indices(self) -> list[dict[str, str]]:
        return self.data["indices"]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read ``config.yaml`` and return a :class:`Config` instance."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
