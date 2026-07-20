# -*- coding: utf-8 -*-
"""
Train every predictor's model concurrently — one process per model, one GPU
per process when GPUs exist (ARCHITECTURE_AND_PERFORMANCE.md §7.5).

Hardware-agnostic scheduling:

    8 GPUs, 8 predictors  -> all 8 train at once (CUDA_VISIBLE_DEVICES=0..7)
    2 GPUs, 8 predictors  -> rolling queue, 2 at a time, GPUs reused
    no GPUs (Mac/laptop)  -> training.max_parallel CPU processes (default 1)

Each child is a plain `python -m pipeline train` subprocess with its own run
dir, log, and progress.json (watchable individually); the supervisor writes
an aggregate progress.json so the control panel / `pipeline status` can
follow the whole sweep. Children inherit the config except for
training.predictor, which is overridden per child — the exact child config
is saved into the sweep's run dir for provenance.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig, predictor_key
from .runner import RUNS_DIR, RunProgress, new_run_id

POLL_SECONDS = 2.0


def visible_gpus(spec: str = "auto") -> list[str]:
    """GPU ids to schedule on. 'auto' = every CUDA device torch can see
    (this respects an externally set CUDA_VISIBLE_DEVICES); '0,2,5' = those
    ids; 'none'/'cpu' = force CPU."""
    spec = (spec or "auto").strip().lower()
    if spec in ("none", "cpu"):
        return []
    if spec != "auto":
        return [s.strip() for s in spec.split(",") if s.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:  # torch missing/broken -> CPU scheduling
        return []


def train_all(cfg: RunConfig, run_id: str | None = None,
              repo_root: Path | None = None) -> dict:
    """Run one training per predictor, in parallel across available GPUs.

    Returns {predictor: {"status", "run_id", "device", "log"}}. The sweep's
    own progress.json reports pct = finished/total.
    """
    root = Path(repo_root or REPO_ROOT)
    run_id = run_id or new_run_id("train-all")
    run_dir = root / RUNS_DIR / run_id
    progress = RunProgress(run_dir)

    predictors = list(cfg.training.predictors) or list(
        cfg.distributions.predictors)
    gpus = visible_gpus(cfg.training.gpus)
    if cfg.training.max_parallel > 0:
        workers = cfg.training.max_parallel
    else:
        workers = len(gpus) if gpus else 1
    workers = max(1, min(workers, len(predictors)))

    progress.update(stage="train-all",
                    message=f"{len(predictors)} predictors, "
                            f"{len(gpus) or 'no'} GPU(s), {workers} at a time")

    queue = list(predictors)
    running: dict[str, dict] = {}      # predictor -> {proc, slot, ...}
    results: dict[str, dict] = {}
    free_slots = list(range(workers))  # slot i uses gpus[i % len(gpus)]

    def launch(predictor, slot: int):
        pk = predictor_key(predictor)   # lists get a '+'-joined tag
        child_cfg = RunConfig.from_dict(cfg.to_dict())
        child_cfg.training.predictor = predictor
        cfg_path = run_dir / f"config_{pk}.yaml"
        child_cfg.save(cfg_path)

        child_id = f"{run_id}-{pk}"
        log_path = run_dir / f"train_{pk}.log"
        env = dict(os.environ)
        device = "cpu"
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
            device = f"cuda:{gpus[slot % len(gpus)]}"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "pipeline", "train",
             "--config", str(cfg_path), "--run-id", child_id],
            cwd=root, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        running[pk] = {"proc": proc, "slot": slot, "fh": log_fh,
                       "run_id": child_id, "device": device,
                       "log": str(log_path)}
        print(f"[train-all] {pk}: started on {device} "
              f"(run {child_id}, log {log_path.name})", flush=True)

    try:
        while queue or running:
            while queue and free_slots:
                launch(queue.pop(0), free_slots.pop(0))
            time.sleep(POLL_SECONDS)
            for predictor in list(running):
                info = running[predictor]
                rc = info["proc"].poll()
                if rc is None:
                    continue
                info["fh"].close()
                free_slots.append(running.pop(predictor)["slot"])
                status = "completed" if rc == 0 else "failed"
                results[predictor] = {"status": status,
                                      "run_id": info["run_id"],
                                      "device": info["device"],
                                      "log": info["log"]}
                print(f"[train-all] {predictor}: {status} (exit {rc})",
                      flush=True)
            done = len(results)
            progress.update(
                pct=100.0 * done / len(predictors),
                message=f"{done}/{len(predictors)} done; "
                        f"running: {', '.join(sorted(running)) or '-'}")
    except BaseException:
        for info in running.values():   # don't orphan children on Ctrl-C
            info["proc"].terminate()
        progress.fail("interrupted")
        raise

    failed = sorted(p for p, r in results.items() if r["status"] != "completed")
    if failed:
        progress.fail(f"failed: {', '.join(failed)}")
    else:
        progress.done(f"{len(predictors)} models trained")
    return results
