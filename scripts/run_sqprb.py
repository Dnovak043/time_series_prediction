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
from run_april import detect_pattern, find_data_dir  # noqa: E402

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
OUTPUT_ROOT = "outputs/sqprb"


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
    pattern = detect_pattern(data_dir)

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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--validation-dates", nargs="+", default=VALIDATION_DATES)
    ap.add_argument("--only", choices=["training", "validation"], default=None,
                    help="run just one stage (default: both)")
    ap.add_argument("--workers", type=int, default=1,
                    help="day-parallel featurize workers (0 = one per core)")
    args = ap.parse_args(argv)

    stages = [args.only] if args.only else ["training", "validation"]
    for symbol in args.symbols:
        for stage in stages:
            run_stage(symbol,
                      "monthly" if stage == "training" else "daily",
                      list(args.validation_dates), args.workers)


if __name__ == "__main__":
    main()
