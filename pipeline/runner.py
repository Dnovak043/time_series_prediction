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
def run(cfg: RunConfig, run_id: str | None = None,
        repo_root: Path | None = None) -> dict:
    """
    Build sequence and class distributions for every (date, predictor) in
    the config and pickle the aggregated outputs. Returns {predictor: paths}.
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

    cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=root)
    builder = DistributionBuilder(cfg.encode, c)

    state = {p: {"L": [], "C": [], "first": None, "last": None,
                 "all": None, "all_cls": None} for p in predictors}

    # one featurize + N_predictor encodes per day
    steps_total = len(dates) * (1 + len(predictors))
    steps_done = 0

    try:
        for date in dates:
            progress.update(stage="featurize", pct=100 * steps_done / steps_total,
                            message=f"featurizing {date}")
            day_df = cache.get(date)
            steps_done += 1

            for predictor in predictors:
                progress.update(stage="distributions",
                                pct=100 * steps_done / steps_total,
                                message=f"{date}: {predictor} -> {c.predicted}")
                ts, z12 = builder.encode_bivariate(day_df, predictor)
                st = state[predictor]

                if c.sequence_calculation:
                    _, counts = builder.sequence_counts(z12)
                    st["L"].append(counts)
                    if st["first"] is None:
                        st["first"] = counts
                    st["last"] = counts
                    st["all"] = integrate_distributions(st["L"], max_len, alphabet)
                    st["L"] = [st["all"]]

                if c.class_calculation:
                    cl = builder.class_counts(ts, z12)
                    st["C"].append(cl)
                    st["all_cls"] = integrate_conditional_class_distributions(
                        st["C"], max_len=max_len, n_classes=c.num_classes,
                        alphabet=alphabet)
                    st["C"] = [st["all_cls"]]

                steps_done += 1

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
    run_id = run_id or new_run_id("train" if command == "train" else "run")
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
