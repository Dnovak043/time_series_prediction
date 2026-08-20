# -*- coding: utf-8 -*-
"""
Generate the SQ_PRB_ / CLS_DISTR_ set for his 2026-08 process_distributions
scheme: a monthly TRAINING aggregate plus per-day VALIDATION files.

Outputs per security (defaults below), all under
``outputs/sqprb/{SYMBOL}/``:

  training  (one file per predictor, tagged with the month)
      SQ_PRB_{sym}_log_mid-micro_price_202504
      SQ_PRB_{sym}_log_mid-vpin_202504
      SQ_PRB_{sym}_log_mid-ofi_L3_norm_n_202504
      SQ_PRB_{sym}_log_mid-sigma_W_202504
      CLS_DISTR_{sym}_log_mid-micro_price_202504_ca4
      CLS_DISTR_{sym}_log_mid-vpin_202504_ca4
      CLS_DISTR_{sym}_log_mid-ofi_L3_norm_n_202504_ca4
      CLS_DISTR_{sym}_log_mid-sigma_W_202504_ca4

  validation  (one file per predictor PER DAY, 20250501 / 02 / 05)
      SQ_PRB_{sym}_log_mid-{predictor}_{yyyymmdd}
      CLS_DISTR_{sym}_log_mid-{predictor}_ca4_{yyyymmdd}

Payloads follow his two branches, which differ:
  monthly SEQ [distrs, samples]   monthly CLS  4-field rows
  daily   SEQ [sequences, probs]  daily   CLS  [[subsequence, class_probs], ...]

NOTE on the class tag: the CLS names carry the class exactly as his driver
writes it, and he orders the two fields DIFFERENTLY in the two branches --
monthly ``..._{month}_{cls}`` but daily ``..._{cls}_{date}``. That is his
convention, reproduced verbatim, not a typo here. Set
``CLASS_TAG_IN_NAME = False`` to drop the tag (safe only while a single
class is swept).

Run:
    .env/bin/python scripts/run_sqprb.py                     # both stages
    .env/bin/python scripts/run_sqprb.py --only training
    .env/bin/python scripts/run_sqprb.py --symbols NVDA --workers 4
    .env/bin/python scripts/run_sqprb.py --validation-dates 20250501 20250502
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))   # run_april helpers

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401  (sys.path bootstrap)
from pipeline.config import RunConfig  # noqa: E402
from pipeline.runner import run  # noqa: E402
from run_april import find_data_dir  # noqa: E402

# ---- the experiment, as requested -------------------------------------------
SYMBOLS = ["NVDA", "AAPL", "INTC", "IBM"]
PREDICTED = "log_mid"
PREDICTORS = ["micro_price", "vpin", "ofi_L3_norm_n", "sigma_W"]
CLASS_NAME = "ca4"
TRAIN_MONTH = "202504"
# 20250503 is a SATURDAY - no session, no raw file. The first three May
# trading days are the 1st, 2nd and 5th.
VALIDATION_DATES = ["20250501", "20250502", "20250505"]
CLASS_TAG_IN_NAME = True         # his driver's convention (see the CLS names above)
N_QUBITS = 3                     # bivariate alphabet 4^2 = 16 -> d = 8
EPOCHS = 5000                    # his (1).py driver: epochs=5000 (the schema
                                 # default is 3000, from his OLDER driver)
PRINT_EVERY = 100                # his (1).py: `if ep % 100 == 0`
NUM_WORKERS = 8                  # his (1).py passes num_workers=8. NOTE: his
                                 # own DataLoader hardcodes 0, so his run used
                                 # 0 workers -- the parameter is dead in his
                                 # code and live in ours ([vendoring fix 1]).
                                 # Matching his STATED value; results are
                                 # identical either way (batch composition is
                                 # set by the parent sampler).
MIN_SEQ_PROB = 0.0001            # his (1).py bivariate branch; the schema
                                 # default is 0.0, which trains on every
                                 # observed sequence instead of dropping the
                                 # rare tail -- a different training SET, not
                                 # just a different runtime
ENS_SEQ_LENGTHS = [1, 2, 3, 4]   # ENS_TD lengths; the ensemble model reads these
OUTPUT_ROOT = "outputs/sqprb"


def detect_pattern(data_dir, prefix: str) -> str:
    """Which raw filename variant this machine has (.dbn.zst or plain .dbn),
    probed with the dates THIS run needs.

    run_april.detect_pattern probes a hardcoded 202504, so it raises on a
    directory that holds only the month a given run wants -- e.g. a March
    training scope.
    """
    for pattern in ("xnas-itch-{date}.mbp-10.dbn.zst",
                    "xnas-itch-{date}.mbp-10.dbn"):
        head, tail = pattern.split("{date}")
        if any(data_dir.glob(head + prefix + "*" + tail)):
            return pattern
    raise SystemExit(f"no {prefix}* .dbn/.dbn.zst files in {data_dir}")


def days_on_disk(data_dir: Path, pattern: str, prefix: str) -> list[str]:
    """Every yyyymmdd present in `data_dir` whose date starts with `prefix`."""
    head, tail = pattern.split("{date}")
    return sorted(p.name[len(head):-len(tail)]
                  for p in data_dir.glob(head + prefix + "*" + tail)
                  if p.name.endswith(tail))


def make_config(symbol: str, data_dir: Path, pattern: str, dates: list[str],
                mode: str, workers: int) -> RunConfig:
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(data_dir)
    cfg.data.file_pattern = pattern
    cfg.data.dates = list(dates)
    cfg.data.instrument_filter = True          # his generate_timeseries always filters

    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = list(PREDICTORS)
    cfg.distributions.class_names = [CLASS_NAME]
    cfg.distributions.class_values = [-1, 0, 1]
    cfg.distributions.class_tag_in_name = CLASS_TAG_IN_NAME
    cfg.distributions.output_mode = mode       # "monthly" | "daily"
    cfg.distributions.sequence_calculation = True
    cfg.distributions.class_calculation = True
    cfg.distributions.output_dir = f"{OUTPUT_ROOT}/{symbol}"

    # the distribution stage never reads training.predictor, but
    # validate() requires it to name one of the configured predictors
    cfg.training.predictor = list(PREDICTORS)[0]
    cfg.training.predictors = list(PREDICTORS)

    # ---- downstream stages, kept coherent with the above --------------
    # ENS_TD_ tables: one channel per predictor, swept over the same class
    # and the lengths the ensemble model reads back.
    cfg.ensemble.reference = "v2"
    cfg.ensemble.predictors = list(PREDICTORS)
    cfg.ensemble.class_names = [CLASS_NAME]
    cfg.ensemble.class_values = [-1, 0, 1]
    cfg.ensemble.seq_lengths = list(ENS_SEQ_LENGTHS)
    cfg.ensemble.output_dir = f"{OUTPUT_ROOT}/{symbol}/ensemble"

    # encoders: one Kraus model per predictor, named so the ensemble stage
    # finds them (training.weights_scheme owns both sides of that join)
    cfg.training.n_qubits = N_QUBITS
    cfg.training.epochs = EPOCHS
    cfg.training.min_seq_prob = MIN_SEQ_PROB
    cfg.training.print_every = PRINT_EVERY
    cfg.training.num_workers = NUM_WORKERS
    cfg.training.model_dir = f"{OUTPUT_ROOT}/{symbol}/models"

    # ensemble model: channels are exactly the predictors we train encoders
    # for. exclude_last_channel is his default, which drops the LAST channel
    # because in his run that is the multivariate one; every channel here is
    # bivariate, so keeping it would silently discard a real feature.
    cfg.ensemble_model.channels = list(PREDICTORS)
    cfg.ensemble_model.channel_names = list(PREDICTORS)
    cfg.ensemble_model.channel_qubits = [N_QUBITS] * len(PREDICTORS)
    cfg.ensemble_model.class_names = [CLASS_NAME]
    cfg.ensemble_model.seq_lens = list(ENS_SEQ_LENGTHS)
    cfg.ensemble_model.exclude_last_channel = False
    cfg.ensemble_model.model_dir = f"{OUTPUT_ROOT}/{symbol}/ensemble_models"

    cfg.featurize.workers = workers
    cfg.featurize.cache_dir = f"{OUTPUT_ROOT}/{symbol}/feature_cache"
    return cfg


def expected_names(cfg: RunConfig, mode: str, dates: list[str]) -> list[str]:
    """Derived from the CONFIG, so this listing cannot drift from the run."""
    names = []
    if mode == "monthly":
        for p in cfg.distributions.predictors:
            names.append(cfg.seq_distr_name(p))
            names.append(cfg.cls_distr_name(p, cls_name=CLASS_NAME))
    else:
        for d in dates:
            for p in cfg.distributions.predictors:
                names.append(cfg.seq_distr_name(p, date=d))
                names.append(cfg.cls_distr_name(p, cls_name=CLASS_NAME, date=d))
    return names


def run_stage(symbol: str, mode: str, dates: list[str], workers: int) -> None:
    data_dir = find_data_dir(symbol)
    if not data_dir.is_dir():
        sys.exit(f"data directory not found for {symbol}: {data_dir}")
    # probe with the month/date range this stage actually needs
    pattern = detect_pattern(
        data_dir, TRAIN_MONTH if mode == "monthly" else dates[0][:6])

    if mode == "monthly":
        dates = days_on_disk(data_dir, pattern, TRAIN_MONTH)
        if not dates:
            sys.exit(f"no {TRAIN_MONTH} raw files in {data_dir}")
    else:
        head, tail = pattern.split("{date}")
        missing = [d for d in dates
                   if not (data_dir / (head + d + tail)).exists()]
        if missing:
            have = days_on_disk(data_dir, pattern, missing[0][:6])
            sys.exit(f"no raw file for {missing} in {data_dir}\n"
                     f"  available that month: {have}")

    cfg = make_config(symbol, data_dir, pattern, dates, mode, workers)
    want = expected_names(cfg, mode, dates)

    print(f"\n=== {symbol} | {mode} | {len(dates)} day(s) "
          f"({dates[0]}..{dates[-1]}) x {len(PREDICTORS)} predictors "
          f"x class {CLASS_NAME} ===")
    problems = cfg.validate()
    if problems:
        sys.exit("config problems:\n  - " + "\n  - ".join(problems))

    run(cfg, run_id=f"sqprb-{mode}-{symbol}")

    out_dir = ROOT / cfg.distributions.output_dir
    missing = [n for n in want if not (out_dir / n).exists()]
    for n in want:
        mark = " " if (out_dir / n).exists() else "MISSING"
        print(f"  {mark:7s} {n}")
    if missing:
        sys.exit(f"{len(missing)} expected file(s) not written")
    print(f"  -> {len(want)} files in {out_dir}")

def write_configs(symbols, workers) -> dict:
    """One YAML per symbol. The GPU fan-out schedulers read configs from
    disk, and writing them also makes the run reproducible from the files
    alone (they record training.gpus / max_parallel)."""
    cfg_dir = ROOT / OUTPUT_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for symbol in symbols:
        data_dir = find_data_dir(symbol)
        if not data_dir.is_dir():
            sys.exit(f"data directory not found for {symbol}: {data_dir}")
        pattern = detect_pattern(data_dir, TRAIN_MONTH)
        dates = days_on_disk(data_dir, pattern, TRAIN_MONTH)
        if not dates:
            sys.exit(f"no {TRAIN_MONTH} raw files in {data_dir}")
        cfg = make_config(symbol, data_dir, pattern, dates, "monthly", workers)
        problems = cfg.validate()
        if problems:
            sys.exit(f"{symbol} config problems:\n  - "
                     + "\n  - ".join(problems))
        paths[symbol] = cfg.save(cfg_dir / f"{symbol.lower()}.yaml")
    return paths


def run_ensemble_tables(symbol: str, workers: int) -> None:
    from pipeline.ensemble import run_ensemble
    data_dir = find_data_dir(symbol)
    pattern = detect_pattern(data_dir, TRAIN_MONTH)
    dates = days_on_disk(data_dir, pattern, TRAIN_MONTH)
    cfg = make_config(symbol, data_dir, pattern, dates, "monthly", workers)
    print(f"\n=== {symbol} | ENS_TD tables | "
          f"{len(cfg.ensemble.predictors)} channels x "
          f"{len(cfg.ensemble.seq_lengths)} lengths x "
          f"{len(cfg.ensemble.class_names)} class ===")
    out = run_ensemble(cfg, run_id=f"sqprb-ens-{symbol}")
    print(f"  -> {len(out)} files in {cfg.ensemble.output_dir}")


def fan_out(stage: str, cfg_paths: dict, dry_run: bool = False) -> None:
    """Stages 3 and 4 across every symbol AND every GPU at once, using the
    repo's existing scheduler (one job per device, workers exit after each
    job so the GPU is released immediately)."""
    from run_april import (plan_ensemble_models, plan_training,
                           run_training_jobs, _train_one_ensemble_model)

    if stage == "train":
        jobs, n_par = plan_training(cfg_paths)
        worker, label = None, "encoder trainings"
    else:
        jobs, n_par = plan_ensemble_models(cfg_paths)
        worker, label = _train_one_ensemble_model, "ensemble models"

    devices = sorted({j["device"] for j in jobs})
    print(f"\n=== {len(jobs)} {label} | {len(devices)} device(s) | "
          f"{n_par} at a time ===")
    if n_par == 1 and len(jobs) > 1:
        # the schedule is one-per-visible-GPU, so this means torch sees at
        # most one device. Say so here rather than leaving a silent hours-long
        # serial run that looks like it is simply slow.
        try:
            import torch
            avail, count = torch.cuda.is_available(), torch.cuda.device_count()
        except Exception as e:                              # noqa: BLE001
            avail, count = f"torch import failed: {e}", 0
        print(f"    WARNING: running ONE AT A TIME. torch.cuda.is_available()"
              f"={avail}, device_count={count}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
              f"\n    Fan-out schedules one job per visible GPU, so a single "
              f"visible device serialises the stage. Unset "
              f"CUDA_VISIBLE_DEVICES, or install a CUDA build of torch, or "
              f"force concurrency with training.max_parallel.")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        print(f"NOTE: CUDA_VISIBLE_DEVICES={cvd!r} restricts this run — "
              f"torch sees only these GPUs, renumbered from cuda:0")
    for j in jobs:
        print(f"    {j['symbol']:6s} x {j['label']:16s} -> {j['device']}")

    # how many jobs land on each device, and how many devices go unused --
    # with fewer jobs than GPUs the round-robin simply cannot reach them all
    per_device = {}
    for j in jobs:
        per_device[j["device"]] = per_device.get(j["device"], 0) + 1
    print(f"    jobs per device: "
          + ", ".join(f"{d}={n}" for d, n in sorted(per_device.items())))
    try:
        import torch
        total = torch.cuda.device_count()
        if total > len(per_device):
            print(f"    NOTE: {total} GPU(s) present but only "
                  f"{len(per_device)} receive work — there are only "
                  f"{len(jobs)} job(s). More symbols or predictors would "
                  f"fill the rest.")
    except Exception:                                       # noqa: BLE001
        pass

    if dry_run:
        print("    (--dry-run: plan only, nothing trained)")
        return

    for i, r in enumerate(run_training_jobs(jobs, n_par, worker=worker), 1):
        tag = r.get("predictor") or r.get("class_name")
        extra = (f"loss={r['loss']:.3e}" if "loss" in r else
                 f"agreement={r.get('agreement_pct', float('nan')):.2f}%")
        out = pathlib.Path(r.get("weights_file") or r["model_file"]).name
        print(f"  [{i}/{len(jobs)}] {r['symbol']:6s} {str(tag):16s} "
              f"{r['device']:8s} {extra}  -> {out}")


def spawn_per_symbol(stage: str, symbols: list, args, n_par: int) -> None:
    """Run a CPU stage for several symbols at once, as separate processes.

    The day loop inside one symbol is capped at the number of trading days
    (~21), so on a many-core box one symbol cannot saturate it; running
    symbols side by side is the remaining axis. Subprocesses rather than a
    pool because each child starts its own day-worker pool.
    """
    import subprocess
    import time

    def cmd(sym):
        c = [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--symbols", sym, "--stages", stage,
             "--workers", str(args.workers), "--symbol-parallel", "1",
             "--validation-dates", *args.validation_dates]
        return c + (["--only", args.only] if args.only else [])

    queue, running, failed = list(symbols), [], []
    while queue or running:
        while queue and len(running) < n_par:
            sym = queue.pop(0)
            print(f"  launching {stage} for {sym}")
            running.append((sym, subprocess.Popen(cmd(sym))))
        for pair in running[:]:
            sym, proc = pair
            rc = proc.poll()
            if rc is not None:
                running.remove(pair)
                print(f"  {sym} {stage}: {'ok' if rc == 0 else f'FAILED rc={rc}'}")
                if rc != 0:
                    failed.append(sym)
        time.sleep(1)
    if failed:
        sys.exit(f"{stage} failed for: {', '.join(failed)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--validation-dates", nargs="+", default=VALIDATION_DATES)
    ap.add_argument("--only", choices=["training", "validation"], default=None,
                    help="within the distributions stage, run just one "
                         "(default: both)")
    ap.add_argument("--stages", nargs="+", default=["distributions"],
                    choices=["distributions", "ensemble-tables", "train",
                             "ensemble-model", "all"],
                    help="which pipeline stages to run, in order. "
                         "'all' = the full chain: distributions -> ENS_TD "
                         "tables -> encoders -> ensemble model.")
    ap.add_argument("--workers", type=int, default=1,
                    help="day-parallel featurize workers WITHIN one symbol "
                         "(0 = one per core, capped at the day count)")
    ap.add_argument("--dry-run", action="store_true",
                    help="for the GPU stages, print the schedule (jobs, "
                         "devices, concurrency) and exit without training")
    ap.add_argument("--symbol-parallel", type=int, default=0,
                    help="how many symbols to process at once in the CPU "
                         "stages. 0 = all of them (the day loop alone cannot "
                         "saturate a many-core box); 1 = one at a time. GPU "
                         "stages always fan out across every visible GPU.")
    args = ap.parse_args(argv)

    stages = args.stages
    if "all" in stages:
        stages = ["distributions", "ensemble-tables", "train",
                  "ensemble-model"]

    substages = [args.only] if args.only else ["training", "validation"]
    n_sym = args.symbol_parallel or len(args.symbols)

    for stage in stages:
        if stage in ("distributions", "ensemble-tables"):
            if n_sym > 1 and len(args.symbols) > 1:
                spawn_per_symbol(stage, args.symbols, args, n_sym)
                continue
            for symbol in args.symbols:
                if stage == "distributions":
                    for sub in substages:
                        run_stage(symbol,
                                  "monthly" if sub == "training" else "daily",
                                  list(args.validation_dates), args.workers)
                else:
                    run_ensemble_tables(symbol, args.workers)
        else:
            # GPU stages: every symbol x predictor/class scheduled together
            fan_out(stage, write_configs(args.symbols, args.workers),
                    dry_run=args.dry_run)


if __name__ == "__main__":
    main()
