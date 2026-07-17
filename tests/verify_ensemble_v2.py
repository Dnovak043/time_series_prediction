# -*- coding: utf-8 -*-
"""
Byte-equivalence test: pipeline `ensemble` stage (reference: v2) vs the
colleague's ensemble_training_data_2.py, on real data.

Reference side: the colleague's own functions, imported from the verbatim
vendored copy (TrainingDistributions/ensemble_reference_2.py), executed the
way his v2 driver executes them — his get_timeseries_by_date, his get_z_ts
(single-predictor bivariate channels AND the joint multivariate channel),
his add_class_label, his prepare_ensemble_training_data (no alphabet_size),
his 'ALL' file naming. His v2 driver already computes date_ts once per day
and reuses it across channels; the only deviations here are pure dedup of
deterministic values: (a) his driver re-featurizes the same day for the
class label and again for every (length, class) combo — we featurize each
day once and hand add_class_label a fresh copy; (b) his z-encodings are
likewise recomputed per combo — we compute each once.

Pipeline side: `pipeline ensemble` with ensemble.reference = "v2" on the
same scope. instrument_filter is OFF on this side to match the colleague's
program exactly (his code has no filter) — this test proves CODE
equivalence; production per-symbol runs then turn the filter on.

Default scope: 1 day (20250401), symbol label INTC, his v2 channels
(3 bivariate + 1 joint ['ofi_L10_norm_n','micro_price','vpin']), all
5 lengths x 4 classes = 20 ENS_TD files compared byte-for-byte.

Run:  .env/bin/python tests/verify_ensemble_v2.py           (~30-45 min,
      dominated by the reference side's one legacy featurize pass)
      .env/bin/python tests/verify_ensemble_v2.py --quick   (k in {1,2} x
      {c1, ca2} = 4 files, same featurize cost)

Only READS data/. Outputs under outputs/ensemble_v2_check/.
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
# his v2 driver's channels, verbatim: 3 bivariate + 1 joint multivariate
CHANNELS = ["ofi_L10_norm_n", "micro_price", "vpin",
            ["ofi_L10_norm_n", "micro_price", "vpin"]]
ALL_LENGTHS = [1, 2, 3, 4, 5]
ALL_CLASSES = ["c1", "c2", "ca2", "ca4"]   # his v2 cls_Names (c4 dropped)

SCRATCH = ROOT / "outputs" / "ensemble_v2_check"
REF_DIR = SCRATCH / "reference"
NEW_DIR = SCRATCH / "pipeline"


def run_reference(dates: list, seq_lengths, class_names, data_dir: Path):
    """The colleague's v2 driver steps, his functions, reduced-scope."""
    import ensemble_reference_2 as ref

    t_start, t_end = datetime.time(9, 30), datetime.time(15, 30)

    daily_z = []                      # his loop order: day-by-day
    daily_cls = {cls: [] for cls in class_names}
    for date in dates:
        args = (str(data_dir), date, "events", 100, [1, 2, 3, 4],
                t_start, t_end)
        t0 = time.time()
        print(f"[reference] {date}: featurize (his get_timeseries_by_date)"
              "...", flush=True)
        date_ts = ref.get_timeseries_by_date(SYMBOL, *args)
        print(f"[reference]   featurized in {time.time()-t0:.0f}s", flush=True)
        z_list = []
        for predictor in CHANNELS:
            t0 = time.time()
            # his v2 driver: same date_ts for every channel (get_z_ts copies)
            z_list.append(ref.get_z_ts(date_ts, PREDICTED, predictor,
                                       0.05, 4))
            print(f"[reference]   {predictor}: {time.time()-t0:.0f}s",
                  flush=True)
        daily_z.append(z_list)
        for cls in class_names:
            # his driver refetches the day for the label; identical values,
            # so a fresh copy of the same featurized frame suffices
            daily_cls[cls].append(ref.add_class_label(date_ts.copy(), cls))
            print(f"[reference]   class {cls} labeled", flush=True)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        daily_data = [(daily_z[d], daily_cls[cls][d])
                      for d in range(len(dates))]
        for k in seq_lengths:
            joint, comp = ref.prepare_ensemble_training_data(
                daily_data=daily_data, sequence_length=k,
                class_values=(-1, 0, 1), smoothing=0.0)
            # his v2 naming, verbatim (predictors_list = 'ALL')
            name = ("ENS_TD_" + SYMBOL + "_" + dates[0][0:6] + "_SL_" + str(k)
                    + "_CL_" + cls + "_" + PREDICTED + "_ALL")
            with open(REF_DIR / name, "wb") as fh:
                pickle.dump([joint, comp], fh)
            print(f"[reference] dumped {name}", flush=True)


def run_pipeline_stage(dates: list, seq_lengths, class_names, data_dir: Path):
    from pipeline.ensemble import run_ensemble

    cfg = RunConfig()
    cfg.data.symbol = SYMBOL
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = list(dates)
    cfg.data.instrument_filter = False        # match his program exactly
    cfg.featurize.cache_dir = str((SCRATCH / "feature_cache")
                                  .relative_to(ROOT))
    cfg.distributions.predicted = PREDICTED
    cfg.ensemble.reference = "v2"
    cfg.ensemble.predictors = list(CHANNELS)
    cfg.ensemble.seq_lengths = list(seq_lengths)
    cfg.ensemble.class_names = list(class_names)
    cfg.ensemble.output_dir = str(NEW_DIR.relative_to(ROOT))
    problems = cfg.validate()
    if problems:
        raise SystemExit(f"config invalid: {problems}")
    run_ensemble(cfg, run_id="ensemble-v2-check")


def compare() -> bool:
    ref_files = {p.name for p in REF_DIR.glob("ENS_TD_*")}
    new_files = {p.name for p in NEW_DIR.glob("ENS_TD_*")}
    ok = bool(ref_files)
    if not ref_files:
        print("FAIL: reference produced no files")
    for name in sorted(ref_files | new_files):
        if name not in ref_files or name not in new_files:
            print(f"  FAIL  {name}: missing on one side")
            ok = False
            continue
        ba = (REF_DIR / name).read_bytes()
        bb = (NEW_DIR / name).read_bytes()
        ha = hashlib.sha256(ba).hexdigest()[:16]
        if ba == bb:
            print(f"  IDENTICAL  {name}  sha256:{ha}")
        else:
            print(f"  FAIL  {name}: bytes differ")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["20250401"],
                    help="trading days (default: 1 day; pass 2+ to also "
                         "exercise the cross-day aggregation path)")
    ap.add_argument("--quick", action="store_true",
                    help="k in {1,2} x {c1, ca2} only (4 files)")
    ap.add_argument("--data-dir", default=None,
                    help="directory with the raw files (the NVDA_INTC "
                         "folder); default: auto-discover")
    args = ap.parse_args()

    seq_lengths = [1, 2] if args.quick else ALL_LENGTHS
    class_names = ["c1", "ca2"] if args.quick else ALL_CLASSES
    data_dir = (Path(args.data_dir).resolve() if args.data_dir
                else find_data_dir())

    for d in (REF_DIR, NEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("ENS_TD_*"):
            p.unlink()

    print(f"=== 1/2 reference (colleague's v2 code) — {' '.join(args.dates)}, "
          f"{len(seq_lengths)}x{len(class_names)} combos ===")
    t0 = time.time()
    run_reference(args.dates, seq_lengths, class_names, data_dir)
    print(f"reference done in {time.time()-t0:.0f}s")

    print("\n=== 2/2 pipeline ensemble stage (reference: v2), same scope ===")
    t0 = time.time()
    run_pipeline_stage(args.dates, seq_lengths, class_names, data_dir)
    print(f"pipeline done in {time.time()-t0:.0f}s")

    print("\n=== comparison ===")
    if compare():
        print("\nENSEMBLE V2 EQUIVALENCE PASSED: pipeline stage reproduces "
              "the colleague's ensemble_training_data_2.py byte-for-byte")
    else:
        print(f"\nENSEMBLE V2 EQUIVALENCE FAILED — outputs kept in {SCRATCH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
