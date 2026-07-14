# -*- coding: utf-8 -*-
"""
Byte-equivalence test: pipeline `ensemble` stage vs the colleague's
ensemble_training_data.py, on real data.

Reference side: the colleague's own functions, imported from the verbatim
vendored copy (TrainingDistributions/ensemble_reference.py), executed the
way his driver executes them — his get_timeseries_by_date (his featurize
copies, not ours), his get_bivariate_ts, his add_class_label, his
prepare_ensemble_training_data, his file naming. One documented deviation:
his driver re-featurizes identical inputs for every (length, class) combo
(600 decodes at the default scope); since featurize/encode are
deterministic (proven byte-for-byte by the parity and baseline harnesses),
we compute each distinct input once (32 decodes) — pure dedup of
identical values.

Pipeline side: `pipeline ensemble` on the same scope. instrument_filter is
OFF on this side to match the colleague's program exactly (his code has no
filter) — this test proves CODE equivalence; production per-symbol runs
then turn the filter on.

Default scope: 2 days (20250401-02, exercises the cross-day aggregation
path), symbol label INTC, his full 11 channels, all 5 lengths x 5 classes
= 25 ENS_TD files compared byte-for-byte.

Run:  .env/bin/python tests/verify_ensemble_reference.py           (~1.5-2h,
      mostly the reference side's 32 slow legacy featurize passes)
      .env/bin/python tests/verify_ensemble_reference.py --quick   (4 files,
      ~1h; add --dates 20250401 for a single-day ~30-40min pass)

Only READS data/. Outputs under outputs/ensemble_check/.
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
CHANNELS = ["log_mid", "tvi_n", "obi_L1", "ofi_L1_n", "ofi_L1_n_norm",
            "ofi_L1_norm_n", "ofi_L3_norm_n", "ofi_L10_norm_n",
            "micro_price", "vpin", "sigma_W"]
ALL_LENGTHS = [1, 2, 3, 4, 5]
ALL_CLASSES = ["c1", "c2", "c4", "ca2", "ca4"]

SCRATCH = ROOT / "outputs" / "ensemble_check"
REF_DIR = SCRATCH / "reference"
NEW_DIR = SCRATCH / "pipeline"


def run_reference(dates: list, seq_lengths, class_names, data_dir: Path):
    """The colleague's driver steps, his functions, reduced-scope."""
    import ensemble_reference as ref

    t_start, t_end = datetime.time(9, 30), datetime.time(15, 30)

    daily_z = []                      # his loop order: day-by-day
    daily_cls = {cls: [] for cls in class_names}
    for date in dates:
        args = (str(data_dir), date, "events", 100, [1, 2, 3, 4],
                t_start, t_end)
        print(f"[reference] {date}: featurize+encode {len(CHANNELS)} "
              "channels (his functions, fresh featurize per channel)...")
        z_list = []
        for predictor in CHANNELS:
            t0 = time.time()
            ts = ref.get_timeseries_by_date(SYMBOL, *args)
            z_list.append(ref.get_bivariate_ts(ts, PREDICTED, predictor,
                                               0.05, 4))
            print(f"[reference]   {predictor}: {time.time()-t0:.0f}s",
                  flush=True)
        daily_z.append(z_list)
        for cls in class_names:
            ts = ref.get_timeseries_by_date(SYMBOL, *args)
            daily_cls[cls].append(ref.add_class_label(ts, cls))
            print(f"[reference]   class {cls} labeled", flush=True)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        daily_data = [(daily_z[d], daily_cls[cls][d])
                      for d in range(len(dates))]
        for k in seq_lengths:
            joint, comp = ref.prepare_ensemble_training_data(
                daily_data=daily_data, sequence_length=k,
                class_values=(-1, 0, 1), alphabet_size=16, smoothing=0.0)
            # his naming, verbatim
            name = ("ENS_TD_" + SYMBOL + "_" + dates[0][0:6] + "_SL_" + str(k)
                    + "_CL_" + cls + "_" + PREDICTED + "_ALL_"
                    + str(len(CHANNELS)))
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
    cfg.ensemble.predictors = list(CHANNELS)
    cfg.ensemble.seq_lengths = list(seq_lengths)
    cfg.ensemble.class_names = list(class_names)
    cfg.ensemble.output_dir = str(NEW_DIR.relative_to(ROOT))
    run_ensemble(cfg, run_id="ensemble-check")


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
    ap.add_argument("--dates", nargs="+", default=["20250401", "20250402"],
                    help="trading days (default: 2 days, exercises the "
                         "cross-day aggregation path)")
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

    print(f"=== 1/2 reference (colleague's code) — {' '.join(args.dates)}, "
          f"{len(seq_lengths)}x{len(class_names)} combos ===")
    t0 = time.time()
    run_reference(args.dates, seq_lengths, class_names, data_dir)
    print(f"reference done in {time.time()-t0:.0f}s")

    print("\n=== 2/2 pipeline ensemble stage, same scope ===")
    t0 = time.time()
    run_pipeline_stage(args.dates, seq_lengths, class_names, data_dir)
    print(f"pipeline done in {time.time()-t0:.0f}s")

    print("\n=== comparison ===")
    if compare():
        print("\nENSEMBLE EQUIVALENCE PASSED: pipeline stage reproduces the "
              "colleague's code byte-for-byte")
    else:
        print(f"\nENSEMBLE EQUIVALENCE FAILED — outputs kept in {SCRATCH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
