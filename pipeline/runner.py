# -*- coding: utf-8 -*-
"""
Run orchestration: the inverted-loop batch driver plus progress reporting.

Loop order is days-outer / predictors-inner, so each raw day is decoded and
featurized once (via DayFeatureCache) and shared by every predictor —
the original driver's loop nest re-featurized each day once per predictor.
Per-predictor incremental aggregation is otherwise identical to the original
(same integrate_* calls in the same order), so outputs match.

Progress is written to outputs/runs/<run_id>/progress.json so any client
(CLI, ipywidgets panel, plain `cat`) can watch a run, including runs
launched as background subprocesses over SSH.
"""
from __future__ import annotations

import datetime
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig, predictor_key
from .distributions import (
    DistributionBuilder,
    flatten_class_distributions,
    format_counts,
)
from .features import DayFeatureCache

RUNS_DIR = "outputs/runs"


# ---------------------------------------------------------------------------
class RunProgress:
    """Atomic JSON progress file, safe to poll from another process."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "progress.json"
        self.state = {
            "status": "running",
            "pid": os.getpid(),
            "pct": 0.0,
            "stage": "",
            "message": "",
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "updated": "",
            "error": "",
        }

    def update(self, **kw):
        self.state.update(kw)
        self.state["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)

    def done(self, message="completed"):
        self.update(status="completed", pct=100.0, message=message)

    def fail(self, err: str):
        self.update(status="failed", error=err, message="failed")


def read_progress(run_id: str, repo_root: Path | None = None) -> dict:
    p = Path(repo_root or REPO_ROOT) / RUNS_DIR / run_id / "progress.json"
    if not p.exists():
        return {"status": "unknown", "message": f"no progress file at {p}"}
    return json.loads(p.read_text())


def list_runs(repo_root: Path | None = None) -> list[str]:
    d = Path(repo_root or REPO_ROOT) / RUNS_DIR
    if not d.exists():
        return []
    return sorted((p.name for p in d.iterdir() if (p / "progress.json").exists()),
                  reverse=True)


def new_run_id(prefix: str = "run") -> str:
    return prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
def _write_day_distributions(cfg, root, date, builder, day_df, cls_keys):
    """His DAILY (validation) outputs for one day -- verbatim from the
    driver's `if daily_sequence_distributions:` / `if save_daily_class:`
    blocks. The payloads deliberately differ from the monthly ones:

        SEQ  ->  [sequences, seq_probs]
                 (his: sequences = flattened all_subsequences,
                       seq_probs = [i[3] for c in counts for i in c])
        CLS  ->  [[subsequence, class_probs], ...]
                 (his: [[i[0], i[2]] for s in cl_distributions[1] for i in s])

    His daily branch is bivariate only, so list (multivariate) predictors are
    skipped rather than silently written in a format he never produces.
    """
    c = cfg.distributions
    out_dir = root / c.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for predictor in c.predictors:
        if not isinstance(predictor, str):
            continue
        ts, z12 = builder.encode_bivariate(day_df, predictor)
        if c.sequence_calculation:
            all_subsequences, counts = builder.sequence_counts(z12)
            sequences = [s for sub in all_subsequences for s in sub]
            seq_probs = [i[3] for cc in counts for i in cc]
            path = out_dir / cfg.seq_distr_name(predictor, date=date)
            with open(path, "wb") as fh:
                pickle.dump([sequences, seq_probs], fh)
            written.append(str(path))
        if c.class_calculation:
            for k in cls_keys:
                cl = builder.class_counts(ts, z12, cls_name=k)
                cls_distr = [[i[0], i[2]] for sub in cl for i in sub]
                path = out_dir / cfg.cls_distr_name(predictor, cls_name=k,
                                                    date=date)
                with open(path, "wb") as fh:
                    pickle.dump(cls_distr, fh)
                written.append(str(path))
    return written


def _write_day_weights(cfg, root, date, builder, day_df):
    """His SQ_PRB_WT_ artifact for one day, verbatim from the driver's
    `if save_seq_prob_weight:` block:

        all_subsequences, counts = <(predicted, predicted) encoding>
        sequences   = [s for subsequence in all_subsequences for s in subsequence]
        seq_probs   = [i[3] for c in counts for i in c]
        global_weights = compute_global_weights(sequences, seq_probs)
        pickle.dump([sequences, seq_probs, global_weights], ...)

    Two quirks of his are preserved: the encoding pairs `predicted` with
    ITSELF (he passes `predicted` as both arguments), and the file name
    carries no predictor tag -- one file per symbol/day, not per predictor.
    """
    c = cfg.distributions
    if not c.save_seq_prob_weight:
        return None
    from process_distributions_v2 import compute_global_weights

    _ts, z12 = builder.encode_bivariate(day_df, c.predicted)
    all_subsequences, counts = builder.sequence_counts(z12)
    sequences = [s for subsequence in all_subsequences for s in subsequence]
    seq_probs = [i[3] for cc in counts for i in cc]
    weights = compute_global_weights(sequences, seq_probs)

    out_dir = root / c.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cfg.seq_prob_weight_name(date)
    with open(path, "wb") as fh:
        pickle.dump([sequences, seq_probs, weights], fh)
    return str(path)


def _process_day(cfg_dict: dict, date: str, root_str: str,
                 device) -> dict:
    """Worker for one day: featurize (via the shared cache) + encode + count
    for every predictor. Module-level so ProcessPoolExecutor can spawn it.
    Returns {predictor: {"counts": ..., "cls": ...}}."""
    import pipeline  # noqa: F401  (sys.path bootstrap in the child)
    from pipeline.config import RunConfig, predictor_key
    from pipeline.distributions import DistributionBuilder
    from pipeline.features import DayFeatureCache

    cfg = RunConfig.from_dict(cfg_dict)
    root = Path(root_str)
    cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
    builder = DistributionBuilder(cfg.encode, cfg.distributions, device=device)
    c = cfg.distributions

    day_df = cache.get(date)
    cls_keys = list(c.class_names) or [None]   # None = legacy single-class
    if c.output_mode == "daily":
        # his validation branch: per-day files, no monthly aggregation
        _write_day_weights(cfg, root, date, builder, day_df)
        return {"__daily_files__": _write_day_distributions(
            cfg, root, date, builder, day_df, cls_keys)}

    out: dict[str, dict] = {}
    for predictor in c.predictors:
        r = _encode_and_count(builder, c, day_df, predictor, cls_keys)
        out[predictor_key(predictor)] = r
    # per-day artifact: written here (not at the monthly fold) because his
    # driver writes it inside the day loop; one independent file per day, so
    # worker-count does not affect its bytes
    _write_day_weights(cfg, root, date, builder, day_df)
    return out


def _encode_and_count(builder, c, day_df, predictor, cls_keys) -> dict:
    """Encode one predictor spec and count. A string is the bivariate
    encoding (SEQ + CLS); a list is the multivariate joint encoding
    (SEQ only — no colleague-defined multivariate CLS output)."""
    r = {}
    if isinstance(predictor, str):
        ts, z12 = builder.encode_bivariate(day_df, predictor)
        if c.sequence_calculation:
            r["counts"] = builder.sequence_counts(z12)[1]
        if c.class_calculation:
            r["cls"] = {k: builder.class_counts(ts, z12, cls_name=k)
                        for k in cls_keys}
    else:
        z_joint = builder.encode_multivariate(day_df, list(predictor))
        if c.sequence_calculation:
            r["counts"] = builder.sequence_counts(z_joint)[1]
    return r


def _resolve_workers(requested: int, n_dates: int) -> int:
    if requested == 1 or n_dates <= 1:
        return 1
    if requested > 0:
        return min(requested, n_dates)
    return min(n_dates, os.cpu_count() or 1)


def run(cfg: RunConfig, run_id: str | None = None,
        repo_root: Path | None = None) -> dict:
    """
    Build sequence and class distributions for every (date, predictor) in
    the config and pickle the aggregated outputs. Returns {predictor: paths}.

    Days are independent, so they run in featurize.workers parallel worker
    processes (0 = auto: one per core, capped at the day count; 1 = serial).
    Results are folded into the monthly aggregate strictly in date order —
    the exact fold the serial loop does — so outputs are byte-identical
    regardless of worker count or completion order.
    """
    from integrate_day_distributions import (
        integrate_conditional_class_distributions,
        integrate_distributions,
    )

    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid config: " + "; ".join(problems))

    root = Path(repo_root or REPO_ROOT)
    run_id = run_id or new_run_id()
    progress = RunProgress(root / RUNS_DIR / run_id)
    cfg.save(progress.run_dir / "config.yaml")   # provenance

    c = cfg.distributions
    dates = list(cfg.data.dates)
    predictors = list(c.predictors)
    max_len = c.max_seq_length
    workers = _resolve_workers(getattr(cfg.featurize, "workers", 1), len(dates))

    cls_keys = list(c.class_names) or [None]   # None = legacy single-class
    pkeys = [predictor_key(p) for p in predictors]
    # symbol alphabet per predictor: n_symbols^2 (bivariate) or
    # n_symbols^(1+len) (multivariate list entry)
    alphabets = {predictor_key(p): list(range(cfg.alphabet_size_for(p)))
                 for p in predictors}
    state = {pk: {"L": [], "first": None, "last": None, "all": None,
                  "C": {k: [] for k in cls_keys},
                  "all_cls": {k: None for k in cls_keys}} for pk in pkeys}

    daily_files: list[str] = []

    def fold(day_result: dict):
        """Aggregate one day's counts — identical math/order to the
        original incremental loop. In daily mode the worker already wrote
        that day's files, so there is nothing to aggregate."""
        if "__daily_files__" in day_result:
            daily_files.extend(day_result["__daily_files__"])
            return
        for pk in pkeys:
            st = state[pk]
            r = day_result[pk]
            alphabet = alphabets[pk]
            if c.sequence_calculation:
                counts = r["counts"]
                st["L"].append(counts)
                if st["first"] is None:
                    st["first"] = counts
                st["last"] = counts
                st["all"] = integrate_distributions(st["L"], max_len, alphabet)
                st["L"] = [st["all"]]
            if c.class_calculation and "cls" in r:
                for k in cls_keys:
                    st["C"][k].append(r["cls"][k])
                    st["all_cls"][k] = integrate_conditional_class_distributions(
                        st["C"][k], max_len=max_len, n_classes=c.num_classes,
                        alphabet=alphabet)
                    st["C"][k] = [st["all_cls"][k]]

    try:
        if workers == 1:
            cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
            builder = DistributionBuilder(cfg.encode, c)
            for i, date in enumerate(dates):
                progress.update(stage="distributions",
                                pct=100.0 * i / len(dates),
                                message=f"{date} ({i + 1}/{len(dates)})")
                day_df = cache.get(date)
                _write_day_weights(cfg, root, date, builder, day_df)
                if c.output_mode == "daily":
                    daily_files += _write_day_distributions(
                        cfg, root, date, builder, day_df, cls_keys)
                    continue
                day_result = {}
                for predictor in predictors:
                    day_result[predictor_key(predictor)] = _encode_and_count(
                        builder, c, day_df, predictor, cls_keys)
                fold(day_result)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            progress.update(stage="distributions",
                            message=f"{len(dates)} days on {workers} workers")
            cfg_dict = cfg.to_dict()
            pending: dict[str, dict] = {}
            next_i = 0
            with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn")) as ex:
                futures = {ex.submit(_process_day, cfg_dict, d, str(root),
                                     "cpu"): d for d in dates}
                for fut in as_completed(futures):
                    pending[futures[fut]] = fut.result()
                    # fold strictly in date order for byte-identical output
                    while next_i < len(dates) and dates[next_i] in pending:
                        fold(pending.pop(dates[next_i]))
                        next_i += 1
                        progress.update(
                            stage="distributions",
                            pct=100.0 * next_i / len(dates),
                            message=f"{next_i}/{len(dates)} days aggregated")

        if c.output_mode == "daily":
            progress.done(f"wrote {len(daily_files)} daily files")
            return {"daily": sorted(daily_files)}

        # ---- persist aggregated outputs, same names/format as the original ----
        out_dir = root / c.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, dict] = {}
        for predictor in predictors:
            pk = predictor_key(predictor)
            st = state[pk]
            paths = {}
            if c.sequence_calculation and st["all"] is not None:
                distrs_all, samples_all = format_counts(st["all"])
                p = out_dir / cfg.seq_distr_name(predictor)
                with open(p, "wb") as fh:
                    pickle.dump([distrs_all, samples_all], fh)
                paths["seq"] = str(p)
            if c.class_calculation:
                for k in cls_keys:
                    if st["all_cls"][k] is None:
                        continue
                    p = out_dir / cfg.cls_distr_name(predictor, cls_name=k)
                    with open(p, "wb") as fh:
                        pickle.dump(
                            flatten_class_distributions(st["all_cls"][k]), fh)
                    paths["cls" if k is None else f"cls_{k}"] = str(p)
            outputs[pk] = paths

        progress.done(f"wrote {sum(len(v) for v in outputs.values())} files")
        return outputs
    except Exception as e:  # noqa: BLE001 - progress file must reflect failures
        progress.fail(f"{type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
def run_async(config_path: str | Path, run_id: str | None = None,
              command: str = "run", repo_root: Path | None = None) -> str:
    """
    Launch `python -m pipeline <command> --config <path>` as a detached
    subprocess (stdout/stderr -> the run's run.log). Returns the run_id;
    poll with read_progress(run_id).
    """
    root = Path(repo_root or REPO_ROOT)
    prefix = (command if command.startswith(("train", "ensemble"))
              else "run")
    run_id = run_id or new_run_id(prefix)
    run_dir = root / RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "run.log", "ab")

    subprocess.Popen(
        [sys.executable, "-m", "pipeline", command,
         "--config", str(config_path), "--run-id", run_id],
        cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return run_id


def stop_run(run_id: str, repo_root: Path | None = None) -> bool:
    """Politely SIGTERM the recorded pid of a running run."""
    import signal

    st = read_progress(run_id, repo_root)
    pid = st.get("pid")
    if st.get("status") == "running" and pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return False
        return True
    return False
