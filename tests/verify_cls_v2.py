# -*- coding: utf-8 -*-
"""
Byte-equivalence test: pipeline distributions stage in V2 multi-class mode
vs the colleague's new process_distributions.py, on real data.

Reference side: the colleague's functions from the vendored copy
(TrainingDistributions/cls_reference.py — his file plus the four
author-authorized/required fixes documented in its header), executed the
way his new driver executes them: his get_timeseries_by_date (his featurize
copies), his get_distribution_by_ts for the SEQ side, his get_bivariate_ts
+ add_class_label + rewritten estimate_subsequence_class_probabilities
(class_values=(-1,0,1)) for the CLS side, his integrate calls, his file
naming (including the double-underscore CLS name). Deviations, both
value-neutral and documented: identical featurize inputs computed once per
day instead of once per (class, predictor, date) — determinism proven by
the parity/baseline harnesses — and each use gets a fresh copy of the day
frame, matching the fresh-featurize isolation his loop has.

Pipeline side: `pipeline run` with distributions.class_names=[...] (v2
mode: per-class CLS files, (-1,0,1) column order, his naming) on the same
scope. instrument_filter OFF on both sides (his program has no filter).

Default scope: 2 days x 2 predictors x classes [c1, ca2]
    -> 2 SEQ_DISTR_* + 4 CLS_DISTR_*_{cls} files, byte-compared.

Run:  .env/bin/python tests/verify_cls_v2.py [--data-dir DIR]
      (~15-25 min; the reference side featurizes 2 days the slow way once)

Only READS data/. Outputs under outputs/cls_v2_check/.
"""
import argparse
import datetime
import hashlib
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "TrainingDistributions"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from parity_check import find_data_dir  # noqa: E402
from pipeline.config import RunConfig  # noqa: E402

SYMBOL = "INTC"          # label; both sides read the same (unfiltered) stream
PREDICTED = "log_mid"
PREDICTORS = ["tvi_n", "obi_L1"]
CLASSES = ["c1", "ca2"]
DATES = ["20250401", "20250402"]
MAX_SEQ_LENGTH = 6
ALPHA, N_SYMBOLS = 0.05, 4

SCRATCH = ROOT / "outputs" / "cls_v2_check"
REF_DIR = SCRATCH / "reference"
NEW_DIR = SCRATCH / "pipeline"


def run_reference(data_dir: Path):
    """The colleague's new driver flow, his functions, reduced scope."""
    import numpy as np
    import cls_reference as ref
    from integrate_day_distributions import (
        integrate_conditional_class_distributions, integrate_distributions)

    t_start, t_end = datetime.time(9, 30), datetime.time(15, 30)
    alphabet = list(range(N_SYMBOLS * N_SYMBOLS))
    month = DATES[0][:6]
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # one featurize per day (identical values to his per-combo recomputation)
    frames = {}
    for date in DATES:
        print(f"[reference] featurize {date} (his functions)...", flush=True)
        frames[date] = ref.get_timeseries_by_date(
            SYMBOL, str(data_dir), date, "events", 100, [1, 2, 3, 4],
            t_start, t_end)

    calculated = {p: False for p in PREDICTORS}
    saved = {p: False for p in PREDICTORS}
    for cls_name in CLASSES:                      # his loop order
        for predictor in PREDICTORS:
            C, L, i = [], [], 0
            allcounts = firstcounts = counts = None
            for date in DATES:
                time_series = frames[date].copy()   # fresh frame per use
                if not calculated[predictor]:
                    all_subsequences, counts, z12 = ref.get_distribution_by_ts(
                        time_series, "bivariate", PREDICTED, predictor,
                        ALPHA, N_SYMBOLS, MAX_SEQ_LENGTH)
                    L.append(counts)
                    if i == 0:
                        firstcounts = counts
                    allcounts = integrate_distributions(
                        L, MAX_SEQ_LENGTH, alphabet)
                    L = [allcounts]

                z_series12 = ref.get_bivariate_ts(
                    time_series, PREDICTED, predictor, ALPHA, N_SYMBOLS)
                cls_ts = ref.add_class_label(time_series, cls_name)
                cl = ref.estimate_subsequence_class_probabilities(
                    z_series12, cls_ts,
                    max_subsequence_length=MAX_SEQ_LENGTH,
                    class_values=(-1, 0, 1))
                C.append(cl[1])
                all_cls_distr = integrate_conditional_class_distributions(
                    C, max_len=MAX_SEQ_LENGTH, n_classes=3, alphabet=alphabet)
                C = [all_cls_distr]
                i += 1
            if not calculated[predictor]:
                calculated[predictor] = True
                # his SEQ save block, verbatim math
                cntsall = [[np.array(t[0]).astype(int).tolist(), t[1], t[2]]
                           for sublist in allcounts for t in sublist]
                distrsall = [[x[0], x[1] / x[2]] for x in cntsall]
                samplesall = [x[0] for x in cntsall]
                if not saved[predictor]:
                    outfname = ("SEQ_DISTR_" + SYMBOL + "_bivariate_"
                                + PREDICTED + "-" + predictor + "_" + month)
                    with open(REF_DIR / outfname, "wb") as fh:
                        pickle.dump([distrsall, samplesall], fh)
                    print("[reference] Dumped", outfname, flush=True)
                    saved[predictor] = True
            # his CLS save block + naming, verbatim
            outfname = ("CLS_DISTR_" + SYMBOL + "_" + "_" + PREDICTED + "-"
                        + predictor + "_" + month + "_" + cls_name)
            cls_distr = [item for sublist in all_cls_distr for item in sublist]
            with open(REF_DIR / outfname, "wb") as fh:
                pickle.dump(cls_distr, fh)
            print("[reference] Dumped", outfname, flush=True)


def run_pipeline_stage(data_dir: Path):
    from pipeline.runner import run

    cfg = RunConfig()
    cfg.data.symbol = SYMBOL
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = list(DATES)
    cfg.data.instrument_filter = False
    cfg.featurize.cache_dir = str((SCRATCH / "feature_cache").relative_to(ROOT))
    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = list(PREDICTORS)
    cfg.distributions.class_names = list(CLASSES)      # v2 mode
    cfg.distributions.class_values = [-1, 0, 1]
    cfg.distributions.output_dir = str(NEW_DIR.relative_to(ROOT))
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    run(cfg, run_id="cls-v2-check")


def compare() -> bool:
    ref_files = {p.name for p in REF_DIR.glob("*_DISTR_*")}
    new_files = {p.name for p in NEW_DIR.glob("*_DISTR_*")}
    ok = bool(ref_files)
    if not ref_files:
        print("FAIL: reference produced no files")
    for name in sorted(ref_files | new_files):
        if name not in ref_files or name not in new_files:
            print(f"  FAIL  {name}: missing on one side")
            ok = False
            continue
        ba, bb = (REF_DIR / name).read_bytes(), (NEW_DIR / name).read_bytes()
        ha = hashlib.sha256(ba).hexdigest()[:16]
        if ba == bb:
            print(f"  IDENTICAL  {name}  sha256:{ha}")
        else:
            print(f"  FAIL  {name}: bytes differ")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="directory with the raw files (the NVDA_INTC "
                         "folder); default: auto-discover")
    args = ap.parse_args()
    data_dir = (Path(args.data_dir).resolve() if args.data_dir
                else find_data_dir())

    for d in (REF_DIR, NEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*_DISTR_*"):
            p.unlink()

    print(f"=== 1/2 reference (colleague's new code) — {' '.join(DATES)}, "
          f"{len(PREDICTORS)} predictors x {CLASSES} ===")
    t0 = time.time()
    run_reference(data_dir)
    print(f"reference done in {time.time()-t0:.0f}s")

    print("\n=== 2/2 pipeline distributions, v2 multi-class mode ===")
    t0 = time.time()
    run_pipeline_stage(data_dir)
    print(f"pipeline done in {time.time()-t0:.0f}s")

    print("\n=== comparison ===")
    if compare():
        print("\nCLS-V2 EQUIVALENCE PASSED: pipeline v2 mode reproduces the "
              "colleague's new code byte-for-byte")
    else:
        print(f"\nCLS-V2 EQUIVALENCE FAILED — outputs kept in {SCRATCH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
