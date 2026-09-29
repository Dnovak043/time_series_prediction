# -*- coding: utf-8 -*-
"""
Feature panels: the pipeline's featurized days, stacked into one frame.

A *panel* is a DataFrame with one row per resampled block and a `day`
column. Every lag / lead / shift in the lab is taken *within* a day, so
overnight gaps never leak into a lead-lag estimate.

Raw data is read through pipeline.features.DayFeatureCache, i.e. exactly
the featurization the pipeline uses; parquet cache entries are reused
(and new ones written under the config's featurize.cache_dir in outputs/,
never under data/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import REPO_ROOT

# Price *levels* are non-stationary; dependence tests on them are mostly
# spurious. stationarize() replaces each with its within-day first difference.
LEVEL_COLUMNS = ("log_mid", "mid_price", "micro_price")

# Extra microstructure columns worth having around besides the config's
# predictors (all present in the featurized frame).
EXTRA_FEATURES = ("log_mid_ret", "rel_spread", "log_event_intensity",
                  "micro_mid", "imbalance", "trade_ofi_n", "dvpin")


def load_config(config):
    """RunConfig from a path (relative to the repo root is fine) or pass-through."""
    from pipeline.config import RunConfig
    if isinstance(config, RunConfig):
        return config
    p = Path(config)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return RunConfig.load(p)


def default_features(cfg) -> list[str]:
    """The config's predicted + predictors (lists flattened), then EXTRA_FEATURES."""
    out = [cfg.distributions.predicted]
    for p in cfg.distributions.predictors:
        out.extend(p if isinstance(p, (list, tuple)) else [p])
    out.extend(EXTRA_FEATURES)
    return list(dict.fromkeys(out))


def default_targets(cfg) -> list[str]:
    return [f"log_mid_return_fwd_{k}" for k in cfg.featurize.forward_intervals]


def load(config, dates: int | Sequence[str] | None = None,
         symbol: str | None = None, features: Iterable[str] | None = None,
         targets: Iterable[str] | None = None, stationarize_levels: bool = True,
         verbose: bool = True) -> pd.DataFrame:
    """
    Stack featurized days into a panel.

    config    RunConfig or YAML path (e.g. "configs/april_nvda.yaml")
    dates     None = all config dates, int n = first n, or explicit list
    symbol    override data.symbol (asset_paths resolves its directory)
    features  columns to keep (default: default_features(cfg))
    targets   forward-return columns (default: one per forward_interval)
    """
    from pipeline.features import DayFeatureCache

    cfg = load_config(config)
    if symbol:
        cfg.data.symbol = symbol
    all_dates = list(cfg.data.dates)
    if dates is None:
        dates = all_dates
    elif isinstance(dates, int):
        dates = all_dates[:dates]
    features = list(features) if features is not None else default_features(cfg)
    targets = list(targets) if targets is not None else default_targets(cfg)

    cache = DayFeatureCache(cfg.data, cfg.featurize)
    frames = []
    for i, d in enumerate(dates, 1):
        if verbose:
            cached = cache.path_for(d).exists()
            print(f"[{i}/{len(dates)}] {cfg.data.symbol} {d} "
                  f"({'cache' if cached else 'featurizing'})", flush=True)
        df = cache.get(d)
        frames.append(_select(df, features, targets, d))
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs.update(symbol=cfg.data.symbol, features=features,
                       targets=targets, dates=list(dates))
    if stationarize_levels:
        panel = stationarize(panel)
    return panel


def from_parquet(paths: Iterable[str | Path], features: Iterable[str],
                 targets: Iterable[str] = ("log_mid_return_fwd_1",),
                 stationarize_levels: bool = True) -> pd.DataFrame:
    """Panel straight from feature-cache parquet files (day = file stem part 2)."""
    features, targets = list(features), list(targets)
    frames = []
    for p in sorted(map(Path, paths)):
        day = p.stem.split("_")[1] if "_" in p.stem else p.stem
        frames.append(_select(pd.read_parquet(p), features, targets, day))
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs.update(features=features, targets=targets)
    return stationarize(panel) if stationarize_levels else panel


def _select(df: pd.DataFrame, features, targets, day) -> pd.DataFrame:
    cols = list(dict.fromkeys(list(features) + list(targets)))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{day}: columns not in featurized frame: {missing}")
    out = df[cols].astype(float).reset_index(drop=True)
    out.insert(0, "day", str(day))
    return out


# -- transforms ---------------------------------------------------------------

def stationarize(panel: pd.DataFrame, levels=LEVEL_COLUMNS) -> pd.DataFrame:
    """Replace level columns x by d_x (within-day diff); renames in attrs too."""
    panel = panel.copy()
    renames = {}
    for c in levels:
        if c in panel.columns:
            panel["d_" + c] = panel.groupby("day")[c].diff()
            panel = panel.drop(columns=c)
            renames[c] = "d_" + c
    if "features" in panel.attrs:
        panel.attrs["features"] = [renames.get(c, c) for c in panel.attrs["features"]]
    return panel


def features_of(panel: pd.DataFrame) -> list[str]:
    return list(panel.attrs.get("features", [c for c in panel.columns if c != "day"]))


def targets_of(panel: pd.DataFrame) -> list[str]:
    return list(panel.attrs.get("targets", []))


def clean(panel: pd.DataFrame, cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Drop rows with NaN/inf in `cols` (default all numeric)."""
    cols = list(cols) if cols is not None else [c for c in panel.columns if c != "day"]
    ok = np.isfinite(panel[cols].to_numpy(float)).all(axis=1)
    out = panel.loc[ok].reset_index(drop=True)
    out.attrs = dict(panel.attrs)
    return out


def standardize(panel: pd.DataFrame, cols: Sequence[str] | None = None,
                per_day: bool = False) -> pd.DataFrame:
    cols = list(cols) if cols is not None else [c for c in panel.columns if c != "day"]
    out = panel.copy()
    if per_day:
        g = out.groupby("day")[cols]
        out[cols] = (out[cols] - g.transform("mean")) / g.transform("std")
    else:
        out[cols] = (out[cols] - out[cols].mean()) / out[cols].std()
    out.attrs = dict(panel.attrs)
    return out


def lagged(panel: pd.DataFrame, cols: Sequence[str], lags: Iterable[int],
           keep: Sequence[str] = ()) -> pd.DataFrame:
    """Within-day lagged copies `{col}_L{k}` (k>0 = past); keeps `keep` + day."""
    g = panel.groupby("day")
    out = {"day": panel["day"]}
    for c in keep:
        out[c] = panel[c]
    for k in lags:
        for c in cols:
            out[f"{c}_L{k}"] = g[c].shift(k)
    return pd.DataFrame(out).dropna().reset_index(drop=True)


def subsample(panel: pd.DataFrame, max_rows: int | None, seed: int = 0) -> pd.DataFrame:
    """Keep whole days (contiguous) until max_rows is reached — preserves dynamics."""
    if not max_rows or len(panel) <= max_rows:
        return panel
    days = panel["day"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(days)
    keep, n = [], 0
    sizes = panel.groupby("day").size()
    for d in days:
        keep.append(d)
        n += sizes[d]
        if n >= max_rows:
            break
    out = panel[panel["day"].isin(keep)]
    if len(out) > max_rows:  # single huge day: take its first max_rows blocks
        out = out.iloc[:max_rows]
    out = out.reset_index(drop=True)
    out.attrs = dict(panel.attrs)
    return out


def encode_symbols(panel: pd.DataFrame, cols: Sequence[str], config) -> pd.DataFrame:
    """
    The pipeline's EWMA z-encoding (process_distributions.z_encoding with the
    config's encode params), applied per day exactly like the pipeline does.
    Returns day + `{col}_sym` int columns — the alphabet the Kraus models see.
    Rows with a warm-up/missing symbol (<0) are dropped. For alphabets identical
    to the pipeline's, load the panel with stationarize_levels=False (the
    pipeline encodes log_mid as a level).
    """
    import pipeline  # noqa: F401  (puts TrainingDistributions on sys.path)
    from process_distributions import z_encoding

    e = load_config(config).encode
    parts = []
    for day, g in panel.groupby("day", sort=False):
        ts = g[list(cols)].copy()
        for c in cols:
            z_encoding(ts, c, e.n_symbols, e.alpha, min_periods=e.min_periods,
                       fill_mode=e.fill_mode, bins_mode=e.bins_mode)
        sym = ts[[c + "_sym" for c in cols]]
        sym.insert(0, "day", day)
        parts.append(sym)
    out = pd.concat(parts).dropna()
    sym_cols = [c + "_sym" for c in cols]
    out[sym_cols] = out[sym_cols].astype(int)
    out = out[(out[sym_cols] >= 0).all(axis=1)].reset_index(drop=True)
    out.attrs["n_symbols"] = e.n_symbols
    return out
