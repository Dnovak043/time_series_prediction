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
from .config import RunConfig
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
def _process_day(cfg_dict: dict, date: str, root_str: str,
                 device) -> dict:
    """Worker for one day: featurize (via the shared cache) + encode + count
    for every predictor. Module-level so ProcessPoolExecutor can spawn it.
    Returns {predictor: {"counts": ..., "cls": ...}}."""
    import pipeline  # noqa: F401  (sys.path bootstrap in the child)
    from pipeline.config import RunConfig
    from pipeline.distributions import DistributionBuilder
    from pipeline.features import DayFeatureCache

    cfg = RunConfig.from_dict(cfg_dict)
    root = Path(root_str)
    cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
    builder = DistributionBuilder(cfg.encode, cfg.distributions, device=device)
    c = cfg.distributions

    day_df = cache.get(date)
    out: dict[str, dict] = {}
    for predictor in c.predictors:
        ts, z12 = builder.encode_bivariate(day_df, predictor)
        r = {}
        if c.sequence_calculation:
            r["counts"] = builder.sequence_counts(z12)[1]
        if c.class_calculation:
            r["cls"] = builder.class_counts(ts, z12)
        out[predictor] = r
    return out


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
    alphabet = list(range(cfg.alphabet_size))
    max_len = c.max_seq_length
    workers = _resolve_workers(getattr(cfg.featurize, "workers", 1), len(dates))

    state = {p: {"L": [], "C": [], "first": None, "last": None,
                 "all": None, "all_cls": None} for p in predictors}

    def fold(day_result: dict):
        """Aggregate one day's counts — identical math/order to the
        original incremental loop."""
        for predictor in predictors:
            st = state[predictor]
            r = day_result[predictor]
            if c.sequence_calculation:
                counts = r["counts"]
                st["L"].append(counts)
                if st["first"] is None:
                    st["first"] = counts
                st["last"] = counts
                st["all"] = integrate_distributions(st["L"], max_len, alphabet)
                st["L"] = [st["all"]]
            if c.class_calculation:
                st["C"].append(r["cls"])
                st["all_cls"] = integrate_conditional_class_distributions(
                    st["C"], max_len=max_len, n_classes=c.num_classes,
                    alphabet=alphabet)
                st["C"] = [st["all_cls"]]

    try:
        if workers == 1:
            cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
            builder = DistributionBuilder(cfg.encode, c)
            for i, date in enumerate(dates):
                progress.update(stage="distributions",
                                pct=100.0 * i / len(dates),
                                message=f"{date} ({i + 1}/{len(dates)})")
                day_df = cache.get(date)
                day_result = {}
                for predictor in predictors:
                    ts, z12 = builder.encode_bivariate(day_df, predictor)
                    r = {}
                    if c.sequence_calculation:
                        r["counts"] = builder.sequence_counts(z12)[1]
                    if c.class_calculation:
                        r["cls"] = builder.class_counts(ts, z12)
                    day_result[predictor] = r
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

        # ---- persist aggregated outputs, same names/format as the original ----
        out_dir = root / c.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, dict] = {}
        for predictor in predictors:
            st = state[predictor]
            paths = {}
            if c.sequence_calculation and st["all"] is not None:
                distrs_all, samples_all = format_counts(st["all"])
                p = out_dir / cfg.seq_distr_name(predictor)
                with open(p, "wb") as fh:
                    pickle.dump([distrs_all, samples_all], fh)
                paths["seq"] = str(p)
            if c.class_calculation and st["all_cls"] is not None:
                p = out_dir / cfg.cls_distr_name(predictor)
                with open(p, "wb") as fh:
                    pickle.dump(flatten_class_distributions(st["all_cls"]), fh)
                paths["cls"] = str(p)
            outputs[predictor] = paths

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
