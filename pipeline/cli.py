# -*- coding: utf-8 -*-
"""
CLI: python -m pipeline <command>

  init-config <path>          write a default run.yaml to edit
  validate  --config <path>   check a config without running anything
  featurize --config <path>   build/refresh the day feature cache only
  run       --config <path>   distribution estimation (SEQ_/CLS_DISTR outputs)
  train     --config <path>   train the configured model on a SEQ_DISTR file
  status    [--run-id <id>]   show progress of the latest (or given) run
  runs                        list known run ids
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import RunConfig


def _load(args) -> RunConfig:
    cfg = RunConfig.load(args.config)
    problems = cfg.validate()
    if problems:
        print("Config problems:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        sys.exit(2)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-config", help="write a default config file")
    p.add_argument("path", nargs="?", default="run.yaml")

    for name in ("validate", "featurize", "run", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name in ("run", "train"):
            p.add_argument("--run-id", default=None)

    p = sub.add_parser("status", help="show run progress")
    p.add_argument("--run-id", default=None)

    sub.add_parser("runs", help="list run ids")

    args = ap.parse_args(argv)

    if args.cmd == "init-config":
        path = RunConfig().save(args.path)
        print(f"Wrote default config to {path}")
        return

    if args.cmd == "validate":
        _load(args)
        print("Config OK")
        return

    if args.cmd == "featurize":
        from .features import DayFeatureCache
        cfg = _load(args)
        cache = DayFeatureCache(cfg.data, cfg.featurize)
        for date in cfg.data.dates:
            df = cache.get(date)
            print(f"{date}: {len(df)} rows -> {cache.path_for(date)}")
        return

    if args.cmd == "run":
        from .runner import run
        cfg = _load(args)
        outputs = run(cfg, run_id=args.run_id)
        for predictor, paths in outputs.items():
            print(predictor, "->", paths)
        return

    if args.cmd == "train":
        from .models import train_model
        cfg = _load(args)
        result = train_model(cfg, run_id=args.run_id)
        print(json.dumps(result, indent=1))
        return

    if args.cmd == "status":
        from .runner import list_runs, read_progress
        run_id = args.run_id or next(iter(list_runs()), None)
        if not run_id:
            print("No runs found")
            return
        print(run_id)
        print(json.dumps(read_progress(run_id), indent=1))
        return

    if args.cmd == "runs":
        from .runner import list_runs
        for rid in list_runs():
            print(rid)
        return
