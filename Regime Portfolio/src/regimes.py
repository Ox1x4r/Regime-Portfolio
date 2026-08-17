"""
Regime detection.

Fits a Gaussian HMM per market and decodes the state path. Emission features
follow ``regimes.feature_source``: either the index return with trailing
realised volatility, or the constituent cross-sectional feature vector.

K is searched over ``n_states_grid`` with ``n_init`` random restarts per K
(EM converges locally), and chosen by BIC or AIC. States are relabelled by
ascending return volatility, since raw HMM labels are arbitrary and unstable
across refits.

Writes ``regime_model_selection.csv``, ``regime_statistics_<INDEX>.csv`` and
``<INDEX>_regimes.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .config import Config, load_config

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Filtered vs smoothed inference
# ---------------------------------------------------------------------------
def filtered_probabilities(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Filtered state probabilities P(s_t | x_1..x_t) from the forward pass.

    ``predict_proba`` returns smoothed probabilities, which condition on the whole
    sample including the future, so they must not drive a decision at t. The two
    coincide at the final observation.
    """
    from scipy.special import logsumexp

    framelogprob = model._compute_log_likelihood(X)
    log_startprob = np.log(np.maximum(model.startprob_, 1e-300))
    log_transmat = np.log(np.maximum(model.transmat_, 1e-300))
    n_samples, n_components = framelogprob.shape

    fwd = np.zeros((n_samples, n_components))
    fwd[0] = log_startprob + framelogprob[0]
    for t in range(1, n_samples):
        for j in range(n_components):
            fwd[t, j] = logsumexp(fwd[t - 1] + log_transmat[:, j]) \
                + framelogprob[t, j]
    return np.exp(fwd - logsumexp(fwd, axis=1, keepdims=True))


def build_regime_features(
    cfg: Config,
    index_returns: pd.Series,
    panel: pd.DataFrame | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Build HMM emission features from the configured source.

    ``source='index'`` uses the index return and trailing realised volatility;
    ``source='constituent'`` uses the cross-sectional feature vector.
    """
    reg = cfg.regimes
    source = source or reg.get("feature_source", "index")
    if source == "index":
        return build_features(index_returns, reg["features"],
                              int(reg["vol_window"]))
    if source == "constituent":
        if panel is None:
            raise ValueError("constituent features require the panel")
        from .constituent_features import constituent_features
        return constituent_features(
            panel,
            window=int(reg.get("constituent_window", 63)),
            max_names_for_eig=int(reg.get("max_names_for_eig", 200)),
            features=list(reg.get("constituent_features")),
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown regime feature_source: {source}")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_features(
    index_returns: pd.Series,
    feature_names: list[str],
    vol_window: int,
) -> pd.DataFrame:
    """Index-level emission features: return, realised_vol, abs_return."""
    r = index_returns.dropna()
    cols: dict[str, pd.Series] = {}
    if "return" in feature_names:
        cols["return"] = r
    if "realised_vol" in feature_names:
        cols["realised_vol"] = r.rolling(vol_window).std()
    if "abs_return" in feature_names:
        cols["abs_return"] = r.abs()
    feats = pd.DataFrame(cols).dropna(how="any")
    return feats


# ---------------------------------------------------------------------------
# HMM fitting with multiple restarts
# ---------------------------------------------------------------------------
def fit_hmm(
    X: np.ndarray,
    n_states: int,
    cov_type: str,
    n_iter: int,
    n_init: int,
    seed: int,
) -> tuple[GaussianHMM, float]:
    """Fit a Gaussian HMM, keeping the best of ``n_init`` restarts."""
    best_model, best_ll = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=seed + i,
            tol=1e-4,
        )
        try:
            model.fit(X)
            ll = model.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll:
            best_model, best_ll = model, ll
    if best_model is None:
        raise RuntimeError(f"HMM failed to fit for n_states={n_states}.")
    return best_model, best_ll


def _n_params(n_states: int, n_features: int, cov_type: str) -> int:
    """Free-parameter count for a Gaussian HMM (for AIC/BIC)."""
    trans = n_states * (n_states - 1)      # rows sum to 1
    start = n_states - 1
    means = n_states * n_features
    if cov_type == "full":
        cov = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        cov = n_states * n_features
    elif cov_type == "spherical":
        cov = n_states
    elif cov_type == "tied":
        cov = n_features * (n_features + 1) // 2
    else:
        cov = n_states * n_features
    return trans + start + means + cov


def select_model(
    X: np.ndarray,
    k_grid: list[int],
    cfg_regimes: dict,
    seed: int,
) -> tuple[pd.DataFrame, int, GaussianHMM]:
    """Fit the K grid and select by BIC or AIC."""
    n_obs, n_feat = X.shape
    cov_type = cfg_regimes["covariance_type"]
    rows, models = [], {}
    for k in k_grid:
        model, ll = fit_hmm(
            X, k, cov_type,
            int(cfg_regimes["n_iter"]),
            int(cfg_regimes["n_init"]),
            seed,
        )
        p = _n_params(k, n_feat, cov_type)
        aic = -2 * ll + 2 * p
        bic = -2 * ll + p * np.log(n_obs)
        rows.append({"K": k, "logL": ll, "n_params": p,
                     "AIC": aic, "BIC": bic,
                     "converged": bool(model.monitor_.converged)})
        models[k] = model
    table = pd.DataFrame(rows).set_index("K")
    criterion = cfg_regimes.get("model_selection", "bic").upper()
    chosen_k = int(table[criterion].idxmin())
    return table, chosen_k, models[chosen_k]


# ---------------------------------------------------------------------------
# Decoding, relabelling, statistics
# ---------------------------------------------------------------------------
def relabel_by_volatility(
    model: GaussianHMM, states: np.ndarray, returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel states by ascending volatility (0 = calmest)."""
    k = model.n_components
    vol_by_state = np.array([
        np.std(returns[states == s]) if np.any(states == s) else np.inf
        for s in range(k)
    ])
    order = np.argsort(vol_by_state)          # ascending volatility
    mapping = np.empty(k, dtype=int)
    for new_label, old_label in enumerate(order):
        mapping[old_label] = new_label
    return mapping[states], mapping


def regime_statistics(
    states: np.ndarray, returns: np.ndarray, k: int
) -> pd.DataFrame:
    """Per-regime frequency, mean, volatility and average duration."""
    rows = []
    n = len(states)
    for s in range(k):
        mask = states == s
        rs = returns[mask]
        # average duration = avg length of consecutive runs in state s
        runs, cur = [], 0
        for v in mask:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        rows.append({
            "regime": s,
            "frequency": float(mask.mean()),
            "n_days": int(mask.sum()),
            "mean_ann": float(np.mean(rs) * TRADING_DAYS) if rs.size else np.nan,
            "vol_ann": float(np.std(rs, ddof=1) * np.sqrt(TRADING_DAYS))
            if rs.size > 1 else np.nan,
            "avg_duration_days": float(np.mean(runs)) if runs else np.nan,
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def detect_regimes_for_index(
    name: str, label: str, cfg: Config, source: str | None = None
) -> dict | None:
    """
    Fit and decode regimes for one index.

    Returns both filtered and smoothed probabilities: only the filtered series may
    condition an allocation.
    """
    path = cfg.processed_dir / f"{name}_index_clean.csv"
    if not path.exists():
        print(f"[skip] {name}: cleaned index file not found")
        return None

    reg = cfg.regimes
    source = source or reg.get("feature_source", "index")
    s = pd.read_csv(path, index_col=0, parse_dates=True)["index_return"]

    panel = None
    if source == "constituent":
        ppath = cfg.processed_dir / f"{name}_constituents_clean.csv"
        if not ppath.exists():
            print(f"[skip] {name}: cleaned constituent panel not found")
            return None
        panel = pd.read_csv(ppath, index_col=0, parse_dates=True)

    feats = build_regime_features(cfg, s, panel, source)
    X = feats.values
    # Index returns aligned to the feature dates, used to label and describe
    # the regimes on a common economic scale regardless of feature source.
    ret_col = s.reindex(feats.index).values

    table, k, model = select_model(
        X, list(reg["n_states_grid"]), reg, cfg.seed)

    raw_states = model.predict(X)
    smoothed = model.predict_proba(X)
    filtered = filtered_probabilities(model, X)
    states, mapping = relabel_by_volatility(model, raw_states, ret_col)
    # reorder probability columns to match relabelled states
    inv = np.argsort(mapping)
    smoothed = smoothed[:, inv]
    filtered = filtered[:, inv]

    stats = regime_statistics(states, ret_col, k)

    regime_df = pd.DataFrame({"state": states}, index=feats.index)
    for j in range(k):
        regime_df[f"p_state_{j}"] = filtered[:, j]        # filtered (for use)
    for j in range(k):
        regime_df[f"smooth_state_{j}"] = smoothed[:, j]   # descriptive only

    print(f"[ok]   {name} [{source}]: chosen K={k} by "
          f"{reg['model_selection'].upper()} | "
          f"regimes vol_ann={stats['vol_ann'].round(3).tolist()}")

    return {"label": label, "name": name, "source": source,
            "selection": table,
            "chosen_k": k, "stats": stats, "regimes": regime_df,
            "transition": pd.DataFrame(model.transmat_)}


def run_regime_detection(cfg: Config | None = None, write_csv: bool = True):
    """Detect regimes for every configured index."""
    cfg = cfg or load_config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = {}
    results = {}
    for spec in cfg.indices:
        out = detect_regimes_for_index(spec["name"], spec["label"], cfg)
        if out is None:
            continue
        results[spec["name"]] = out
        sel = out["selection"].copy()
        sel.insert(0, "chosen", sel.index == out["chosen_k"])
        for kk, row in sel.iterrows():
            selection_rows[(spec["label"], kk)] = row

        if write_csv:
            out["stats"].to_csv(
                cfg.tables_dir / f"regime_statistics_{spec['name']}.csv")
            out["regimes"].to_csv(
                cfg.processed_dir / f"{spec['name']}_regimes.csv")

    if selection_rows:
        sel_table = pd.DataFrame(selection_rows).T
        sel_table.index.names = ["index", "K"]
        if write_csv:
            sel_table.to_csv(cfg.tables_dir / "regime_model_selection.csv")
        print(f"\n[io]   wrote regime tables to {cfg.tables_dir} "
              f"and state paths to {cfg.processed_dir}")
    return results


def _compare_job(job):
    """Module-level worker: one market's constituent-vs-index comparison."""
    name, label, cfg = job
    return name, compare_feature_sources(name, label, cfg)


def run_source_comparison(cfg: Config | None = None,
                          workers: int = 1) -> dict:
    """Run the constituent-vs-index comparison for every market."""
    from .parallel import parallel_map

    cfg = cfg or load_config()
    jobs = [(spec["name"], spec["label"], cfg) for spec in cfg.indices
            if (cfg.processed_dir / f"{spec['name']}_index_clean.csv").exists()]
    results = parallel_map(_compare_job, jobs, workers, desc="regime-compare")
    return {name: res for name, res in results if res is not None}


if __name__ == "__main__":
    _cfg = load_config()
    res = run_regime_detection(_cfg)
    for name, out in res.items():
        print(f"\n=== {out['label']} (K={out['chosen_k']}) ===")
        with pd.option_context("display.width", 160,
                               "display.float_format", lambda v: f"{v:.4f}"):
            print(out["stats"])
    print("\n\n############ constituent vs index regime comparison ############")
    run_source_comparison(_cfg)


# ---------------------------------------------------------------------------
# Headline comparison: constituent-level vs index-level regime detection
# ---------------------------------------------------------------------------
def compare_feature_sources(
    name: str, label: str, cfg: Config, write_csv: bool = True
) -> dict | None:
    """
    Fit regimes from constituent features and from the index return.

    Both models use the same state grid, criterion and seed, so only the
    information set differs. Returns the comparison summary and per-source detail.
    """
    out = {}
    for src in ("constituent", "index"):
        res = detect_regimes_for_index(name, label, cfg, source=src)
        if res is None:
            return None
        out[src] = res

    crisis_periods = cfg.raw["eda"]["crisis_periods"]
    rows = []
    for src, res in out.items():
        df = res["regimes"]
        k = res["chosen_k"]
        top = k - 1                                   # most volatile regime
        in_top = (df["state"] == top)
        row = {
            "source": src,
            "K": k,
            "n_days": len(df),
            "pct_days_in_crisis_regime": round(100.0 * in_top.mean(), 2),
        }
        for cp in crisis_periods:
            m = (df.index >= pd.Timestamp(cp["start"])) & \
                (df.index <= pd.Timestamp(cp["end"]))
            if m.sum():
                row[f"recall_{cp['name']}"] = round(
                    100.0 * float(in_top[m].mean()), 1)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("source")

    # Agreement between the two decoded paths on shared dates.
    a = out["constituent"]["regimes"]["state"]
    b = out["index"]["regimes"]["state"]
    shared = a.index.intersection(b.index)
    agreement = float((a.loc[shared] == b.loc[shared]).mean()) if len(shared) else np.nan
    summary.attrs["agreement"] = agreement

    print(f"\n=== {label}: constituent vs index regime detection ===")
    print(summary.to_string())
    print(f"decoded-state agreement on {len(shared)} shared days: "
          f"{100 * agreement:.1f}%")

    if write_csv:
        cfg.tables_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(cfg.tables_dir / f"regime_source_comparison_{name}.csv")
        for src, res in out.items():
            res["stats"].to_csv(
                cfg.tables_dir / f"regime_statistics_{name}_{src}.csv")
            res["regimes"].to_csv(
                cfg.processed_dir / f"{name}_regimes_{src}.csv")

    return {"summary": summary, "agreement": agreement, "detail": out}
