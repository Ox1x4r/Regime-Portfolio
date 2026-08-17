"""
Markov-switching GARCH regimes (optional).

Fits MS-GARCH per index via the R ``MSGARCH`` package through rpy2, as a
volatility-focused comparison against the HMM. Conditions no portfolio; the
pipeline runs without it.

R is used because there is no maintained Python implementation and the model's
path dependence makes a from-scratch version error-prone.

Parameters live under ``msgarch`` in config.yaml. Writes
``<INDEX>_msgarch_regimes.csv`` and ``msgarch_summary.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# R bridge (kept isolated so the rest of the module stays importable/testable)
# ---------------------------------------------------------------------------
def _get_r_bridge():
    """
    Import rpy2 and MSGARCH.

    Returns (ro, msgarch, localconverter, converter). Raises with an actionable
    message if R or the package is unavailable.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri, default_converter
        from rpy2.robjects.conversion import localconverter
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "rpy2 is required for MS-GARCH (Step 4b) but could not be "
            "imported. Install R, then `pip install rpy2`, and ensure R is on "
            "the system path (set R_HOME if needed)."
        ) from exc
    try:
        msgarch = importr("MSGARCH")
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "The R package 'MSGARCH' could not be loaded. Install it inside R "
            "with install.packages('MSGARCH')."
        ) from exc
    # A conversion context that maps numpy arrays to/from R, used to wrap the
    # R calls locally (replaces the deprecated global activate()).
    converter = default_converter + numpy2ri.converter
    return ro, msgarch, localconverter, converter


# ---------------------------------------------------------------------------
# Post-processing helpers (pure Python -> unit-testable without R)
# ---------------------------------------------------------------------------
def relabel_states_by_vol(
    state_vol: np.ndarray,
) -> np.ndarray:
    """Permutation mapping raw states to volatility-ascending labels."""
    order = np.argsort(state_vol)
    mapping = np.empty(len(state_vol), dtype=int)
    for new_label, old_label in enumerate(order):
        mapping[old_label] = new_label
    return mapping


def decode_from_smoothed(smoothed: np.ndarray) -> np.ndarray:
    """Hard state assignment from smoothed regime probabilities (argmax)."""
    return np.asarray(smoothed).argmax(axis=1)


def summarise_regimes(
    returns: pd.Series, states: np.ndarray, n_states: int
) -> pd.DataFrame:
    """Per-regime descriptive statistics."""
    r = returns.values
    rows = []
    for s in range(n_states):
        mask = states == s
        rs = r[mask]
        rows.append({
            "regime": s,
            "frequency": float(mask.mean()) if mask.size else np.nan,
            "n_days": int(mask.sum()),
            "mean_ann": float(np.mean(rs) * TRADING_DAYS) if rs.size else np.nan,
            "vol_ann": float(np.std(rs, ddof=1) * np.sqrt(TRADING_DAYS))
            if rs.size > 1 else np.nan,
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Core fit for one index
# ---------------------------------------------------------------------------
def fit_msgarch_for_series(
    returns: pd.Series, cfg_ms: dict, ro, localconverter, converter,
) -> dict:
    """Fit MS-GARCH to one series and extract the regime path."""
    scale = float(cfg_ms.get("return_scale", 100.0))
    n_states = int(cfg_ms["n_states"])
    garch = cfg_ms["garch_model"]
    dist = cfg_ms["distribution"]

    r = returns.dropna()
    r_scaled = (r * scale).values.astype(float)

    # All R interaction happens inside the conversion context, which replaces
    # the deprecated global numpy2ri.activate().
    with localconverter(converter):
        ro.globalenv["r_data"] = ro.FloatVector(r_scaled)
        ro.r(f'''
            spec <- CreateSpec(
                variance.spec = list(model = rep("{garch}", {n_states})),
                distribution.spec = list(distribution = rep("{dist}", {n_states})),
                switch.spec = list(do.mix = FALSE)
            )
            fit <- FitML(spec = spec, data = r_data)
            smoothed_mat <- State(fit)$SmoothProb
            condvol_vec <- Volatility(fit)
        ''')
        # Coerce smoothed probabilities to a (T, K) numpy matrix. MSGARCH
        # returns a 3-D array [T (+1 burn-in), 1, K]; drop the middle axis.
        smoothed = np.asarray(ro.r("as.matrix(smoothed_mat[,1,])"), dtype=float)
        cond_vol = np.asarray(ro.r("as.numeric(condvol_vec)"), dtype=float)

    if smoothed.ndim == 1:
        smoothed = smoothed.reshape(-1, n_states)
    # Trim any leading burn-in row so lengths align with the data.
    if smoothed.shape[0] == len(r) + 1:
        smoothed = smoothed[1:, :]

    raw_states = decode_from_smoothed(smoothed)

    # State-conditional volatility per regime (average realised vol per state,
    # in daily return units), used for vol-ascending relabelling.
    state_vol = np.array([
        np.std(r.values[raw_states == s], ddof=1)
        if np.sum(raw_states == s) > 1 else np.inf
        for s in range(n_states)
    ])
    mapping = relabel_states_by_vol(state_vol)
    states = mapping[raw_states]
    smoothed = smoothed[:, np.argsort(mapping)]

    cond_vol = cond_vol / scale

    stats = summarise_regimes(r, states, n_states)

    out_df = pd.DataFrame({"state": states}, index=r.index)
    for j in range(n_states):
        out_df[f"p_state_{j}"] = smoothed[:, j]
    if len(cond_vol) == len(out_df):
        out_df["cond_vol"] = cond_vol

    return {"regimes": out_df, "stats": stats,
            "n_states": n_states, "state_vol": np.sort(state_vol)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_msgarch(cfg: Config | None = None, write_csv: bool = True) -> dict:
    """Fit MS-GARCH for every index. Requires R and MSGARCH via rpy2."""
    cfg = cfg or load_config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg_ms = cfg.raw["msgarch"]

    ro, msgarch, localconverter, converter = _get_r_bridge()

    results, summary_rows = {}, {}
    for spec in cfg.indices:
        name, label = spec["name"], spec["label"]
        path = cfg.processed_dir / f"{name}_index_clean.csv"
        if not path.exists():
            print(f"[skip] {name}: cleaned index file not found")
            continue
        s = pd.read_csv(path, index_col=0, parse_dates=True)["index_return"]
        try:
            out = fit_msgarch_for_series(
                s, cfg_ms, ro, localconverter, converter)
        except Exception as exc:  # pragma: no cover - R-runtime specific
            print(f"[fail] {name}: MS-GARCH fit failed ({exc})")
            continue
        results[name] = out
        for reg, row in out["stats"].iterrows():
            summary_rows[(label, reg)] = row
        print(f"[ok]   {name}: MS-GARCH K={out['n_states']} | "
              f"regime vol_ann={out['stats']['vol_ann'].round(3).tolist()}")

        if write_csv:
            out["regimes"].to_csv(
                cfg.processed_dir / f"{name}_msgarch_regimes.csv")

    if summary_rows and write_csv:
        summary = pd.DataFrame(summary_rows).T
        summary.index.names = ["index", "regime"]
        summary.to_csv(cfg.tables_dir / "msgarch_summary.csv")
        print(f"\n[io]   wrote MS-GARCH outputs to {cfg.processed_dir} "
              f"and {cfg.tables_dir}")
    return results


if __name__ == "__main__":
    run_msgarch()
