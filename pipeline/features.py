# -*- coding: utf-8 -*-
"""
DayFeatureCache: read + featurize each raw day exactly once.

The featurized DataFrame depends only on (symbol, date, featurize params) —
not on which predictor is being processed — so the original driver's
"re-decode the raw file once per predictor" pattern (8x waste) is replaced
by a keyed on-disk cache. Raw files under data/ are only ever read.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import REPO_ROOT
from .config import DataConfig, FeaturizeConfig


def _parse_hhmm(s) -> datetime.time:
    # hand-edited YAML may deliver an unquoted 9:30 as the integer 570
    # (YAML 1.1 base-60); interpret that as minutes since midnight
    if isinstance(s, int):
        return datetime.time(s // 60, s % 60)
    h, m = str(s).split(":")
    return datetime.time(int(h), int(m))


class DayFeatureCache:
    def __init__(self, data_cfg: DataConfig, feat_cfg: FeaturizeConfig,
                 repo_root: Path | None = None):
        self.data_cfg = data_cfg
        self.feat_cfg = feat_cfg
        self.root = Path(repo_root or REPO_ROOT)
        self.cache_dir = self.root / feat_cfg.cache_dir
        self.data_path = str(self.root / data_cfg.data_path)

    # -- cache keying ---------------------------------------------------------
    def params_key(self) -> str:
        """Hash of every parameter that changes the featurized output."""
        d = self.data_cfg
        f = self.feat_cfg
        payload = {
            "symbol": d.symbol,
            "file_pattern": d.file_pattern,
            "session": [d.session_start, d.session_end],
            "resampling": f.resampling,
            "frequency": f.frequency,
            "forward_intervals": list(f.forward_intervals),
            "vol_window": f.vol_window or 3 * f.frequency,
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def path_for(self, date: str) -> Path:
        ext = "parquet" if self.feat_cfg.cache_format == "parquet" else "pkl"
        name = f"{self.data_cfg.symbol}_{date}_{self.params_key()}.{ext}"
        return self.cache_dir / name

    # -- public API -------------------------------------------------------------
    def get(self, date: str, force: bool = False) -> pd.DataFrame:
        path = self.path_for(date)
        if self.feat_cfg.use_cache and not force and path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_pickle(path)

        df = self.build(date)

        if self.feat_cfg.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                if path.suffix == ".parquet":
                    df.to_parquet(tmp)
                else:
                    df.to_pickle(tmp)
                tmp.rename(path)
            except Exception:
                # cache write failures (e.g. exotic dtypes vs parquet) must not
                # break the run — recompute next time instead
                tmp.unlink(missing_ok=True)
        return df

    def build(self, date: str) -> pd.DataFrame:
        """Featurize one day from the raw file (no cache involved)."""
        # legacy modules imported lazily: pulling in process_distributions also
        # imports matplotlib etc., which the UI/config layer shouldn't need
        from process_distributions import (
            generate_timeseries,
            generate_timeseries_time,
            generate_timeseries_volume,
        )

        d, f = self.data_cfg, self.feat_cfg
        fname = d.file_pattern.format(date=date)
        t_start = _parse_hhmm(d.session_start)
        t_end = _parse_hhmm(d.session_end)
        w = f.vol_window or 3 * f.frequency
        k_fwd = list(f.forward_intervals)

        generators = {
            "events": generate_timeseries,
            "seconds": generate_timeseries_time,
            "volume": generate_timeseries_volume,
        }
        if f.resampling not in generators:
            raise ValueError(f"Unknown resampling mode: {f.resampling!r}")
        return generators[f.resampling](
            date, t_start, t_end, f.frequency, k_fwd, w, fname, self.data_path
        )
