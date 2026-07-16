# -*- coding: utf-8 -*-
"""
Ensemble training-data stage: fixed-length multi-channel tables (ENS_TD_*).

Integrates the colleague's ensemble_training_data.py experiment into the
pipeline. The counting math (`prepare_ensemble_training_data`,
`_finalize_table`) is imported VERBATIM from
TrainingDistributions/ensemble_reference.py — this stage only replaces the
data plumbing around it:

    colleague's script                     this stage
    ------------------------------------  -----------------------------------
    get_timeseries_by_date per predictor  DayFeatureCache: one featurize per
    per date per (length, class) combo    (symbol, day), served from cache
    (~6,300 decodes for a month)          (21 decodes)
    get_bivariate_ts per combo            one encode per (day, channel),
                                          reused across all combos
    add_class_label per combo             one class series per (day, class)

Value-equivalence rests on determinism: featurize and encode produce
identical values every time they run (proven byte-for-byte by the parity /
baseline harnesses), so computing them once instead of 25 times cannot
change the counts. tests/verify_ensemble_reference.py (run by the user)
byte-compares this stage's ENS_TD_* pickles against the colleague's own
functions run his way.

Output naming is the colleague's, verbatim:
    ENS_TD_{symbol}_{yyyymm}_SL_{k}_CL_{cls}_{predicted}_ALL_{n_channels}
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig
from .distributions import DistributionBuilder
from .features import DayFeatureCache
from .runner import RUNS_DIR, RunProgress, new_run_id


def run_ensemble(cfg: RunConfig, run_id: str | None = None,
                 repo_root: Path | None = None) -> dict:
    """Build every (seq_length, class_name) ensemble table for the config's
    dates. Returns {(seq_length, class_name): output_path}."""
    from ensemble_reference import prepare_ensemble_training_data
    from process_distributions import add_class_label

    root = Path(repo_root or REPO_ROOT)
    run_id = run_id or new_run_id("ensemble")
    progress = RunProgress(root / RUNS_DIR / run_id)
    cfg.save(progress.run_dir / "config.yaml")   # provenance

    e = cfg.ensemble
    dates = list(cfg.data.dates)
    channels = list(e.predictors)
    class_names = list(e.class_names)
    seq_lengths = [int(k) for k in e.seq_lengths]
    class_values = tuple(int(v) for v in e.class_values)
    predicted = cfg.distributions.predicted

    cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
    builder = DistributionBuilder(cfg.encode, cfg.distributions)

    # ---- per day: one featurize, one encode per channel, one class series
    # per class name (colleague's script recomputes all of this per
    # (length, class) combo; the values are deterministic, so once is enough)
    daily_z: list[list] = []                       # [day][channel] -> z Series
    daily_cls = {c: [] for c in class_names}       # class -> [day] -> Series
    steps_total = len(dates) + len(seq_lengths) * len(class_names)
    for i, date in enumerate(dates):
        progress.update(stage="encode", pct=100.0 * i / steps_total,
                        message=f"{date}: {len(channels)} channels "
                                f"({i + 1}/{len(dates)} days)")
        day_df = cache.get(date)
        # encode_bivariate copies the frame per channel, matching the fresh
        # featurize the colleague's get_bivariate_ts call sites see
        daily_z.append([builder.encode_bivariate(day_df, ch)[1]
                        for ch in channels])
        for cls in class_names:
            daily_cls[cls].append(add_class_label(day_df.copy(), cls))

    # ---- count + save, colleague's function and naming verbatim ----------
    out_dir = root / e.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    month = dates[0][:6]
    plist = "ALL_" + str(len(channels))
    outputs: dict = {}
    done = len(dates)
    for cls in class_names:
        daily_data = [(daily_z[d], daily_cls[cls][d])
                      for d in range(len(dates))]
        for k in seq_lengths:
            t0 = time.time()
            progress.update(stage="count", pct=100.0 * done / steps_total,
                            message=f"SL={k} CL={cls}")
            joint_data, component_data = prepare_ensemble_training_data(
                daily_data=daily_data,
                sequence_length=k,
                class_values=class_values,
                alphabet_size=cfg.alphabet_size,
                smoothing=e.smoothing,
            )
            name = (f"ENS_TD_{cfg.data.symbol}_{month}_SL_{k}_CL_{cls}"
                    f"_{predicted}_{plist}")
            path = out_dir / name
            with open(path, "wb") as fh:
                pickle.dump([joint_data, component_data], fh)
            outputs[(k, cls)] = str(path)
            done += 1
            print(f"Dumped {name}  "
                  f"({joint_data['n_occurrences']} samples, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    progress.done(f"wrote {len(outputs)} ENS_TD files")
    return outputs
