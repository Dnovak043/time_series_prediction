# -*- coding: utf-8 -*-
"""
Equivalence test: our distributions stage in his 2026-08 SQ_PRB_ scheme vs
HIS OWN OUTPUT FILES, for INTC.

Reference side: the files his process_distributions run already produced,
read from --reference-dir (default
/home/ilu671742/vanio/experiments/kraus_models/data). Nothing is executed
on that side -- it is his committed output, used as the oracle.

Pipeline side: `pipeline run` with distributions.output_mode set per stage,
SQ_PRB_ naming and the class tag in his per-branch field order. Scope is
his experiment:

    symbol      INTC
    predicted   log_mid
    predictors  micro_price, vpin, ofi_L3_norm_n, sigma_W
    class       ca4
    training    monthly aggregate over every 202503 day on disk
    validation  daily: 20250401, 20250402, 20250403

    -> training    4 SQ_PRB_ +  4 CLS_DISTR_   (month-tagged)
       validation 12 SQ_PRB_ + 12 CLS_DISTR_   (date-tagged)
       32 files compared in total.

Comparison is sha256 byte-for-byte, the project standard. When bytes
differ, both pickles are loaded and the FIRST structural/numeric
difference is printed with its path (e.g. "distrs[418][1]: 0.0031 vs
0.0029"), so a failure says what diverged rather than just that it did.

Run (on the compute box, where the reference files live):

    .env/bin/python tests/verify_sqprb_vs_reference.py
    .env/bin/python tests/verify_sqprb_vs_reference.py --workers 0
    .env/bin/python tests/verify_sqprb_vs_reference.py --skip-run   # compare only
    .env/bin/python tests/verify_sqprb_vs_reference.py \\
        --reference-dir /path/to/his/data --only validation

Only READS data/ and the reference directory. Our outputs go to
outputs/sqprb_verify/{symbol}/. Exit 0 = every compared file identical.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from pipeline.config import RunConfig  # noqa: E402
from pipeline.runner import run  # noqa: E402

REFERENCE_DIR = "/home/ilu671742/vanio/experiments/kraus_models/data"
SYMBOL = "INTC"
PREDICTED = "log_mid"
PREDICTORS = ["micro_price", "vpin", "ofi_L3_norm_n", "sigma_W"]
CLASS_NAME = "ca4"
TRAIN_MONTH = "202503"
VALIDATION_DATES = ["20250401", "20250402", "20250403"]
OUT_ROOT = "outputs/sqprb_verify"


# ---------------------------------------------------------------- comparison
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_difference(a, b, path="") -> str | None:
    """Deep-compare two loaded pickles; return a description of the first
    difference found, or None if they are equal. Floats are compared
    exactly: the project's standard is byte-identical output, so any
    numeric drift at all is a finding."""
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = list(np.asarray(a).tolist()), list(np.asarray(b).tolist())
    except Exception:                                    # numpy optional here
        pass

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path or 'root'}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            only_a = sorted(set(a) - set(b))[:3]
            only_b = sorted(set(b) - set(a))[:3]
            return f"{path or 'root'}: keys differ (ours {only_a}, ref {only_b})"
        for k in a:
            d = first_difference(a[k], b[k], f"{path}[{k!r}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path or 'root'}: {a!r} vs {b!r}"
    return None


def compare(ours: Path, ref: Path) -> tuple[bool, str]:
    if not ref.exists():
        return False, "reference file not found"
    if not ours.exists():
        return False, "our file was not produced"
    if sha256(ours) == sha256(ref):
        return True, "identical (sha256)"
    try:
        with open(ours, "rb") as fh:
            a = pickle.load(fh)
        with open(ref, "rb") as fh:
            b = pickle.load(fh)
    except Exception as e:                               # noqa: BLE001
        return False, f"bytes differ; could not unpickle to diff ({e})"
    d = first_difference(a, b)
    if d is None:
        return False, ("bytes differ but contents are equal — pickle "
                       "protocol/encoding difference only")
    return False, f"bytes differ; {d}"


# ---------------------------------------------------------------- pipeline
def make_config(symbol, data_dir, pattern, dates, mode, workers, predictors):
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(data_dir)
    cfg.data.file_pattern = pattern
    cfg.data.dates = list(dates)
    cfg.data.instrument_filter = True

    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = list(predictors)
    cfg.distributions.class_names = [CLASS_NAME]
    cfg.distributions.class_values = [-1, 0, 1]
    cfg.distributions.class_tag_in_name = True          # his convention
    cfg.distributions.output_mode = mode
    cfg.distributions.output_dir = f"{OUT_ROOT}/{symbol}"

    # the distribution stage never reads training.predictor, but
    # validate() requires it to name one of the configured predictors
    cfg.training.predictor = list(predictors)[0]
    cfg.training.predictors = list(predictors)

    cfg.featurize.workers = workers
    cfg.featurize.cache_dir = f"{OUT_ROOT}/{symbol}/feature_cache"
    return cfg


def expected(cfg, mode, dates, predictors):
    names = []
    if mode == "monthly":
        for p in predictors:
            names.append(cfg.seq_distr_name(p))
            names.append(cfg.cls_distr_name(p, cls_name=CLASS_NAME))
    else:
        for d in dates:
            for p in predictors:
                names.append(cfg.seq_distr_name(p, date=d))
                names.append(cfg.cls_distr_name(p, cls_name=CLASS_NAME,
                                                date=d))
    return names


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
    head, tail = pattern.split("{date}")
    return sorted(p.name[len(head):-len(tail)]
                  for p in data_dir.glob(head + prefix + "*" + tail)
                  if p.name.endswith(tail))


def stage(args, mode: str) -> list[tuple[str, bool, str]]:
    from run_april import find_data_dir

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir(args.symbol)
    if not data_dir.is_dir():
        sys.exit(f"raw data directory not found: {data_dir}")
    # probe with the month/date range this stage actually needs
    pattern = detect_pattern(
        data_dir,
        args.train_month if mode == "monthly" else args.validation_dates[0][:6])

    if mode == "monthly":
        dates = days_on_disk(data_dir, pattern, args.train_month)
        if not dates:
            sys.exit(f"no {args.train_month} raw files in {data_dir}")
    else:
        dates = list(args.validation_dates)
        head, tail = pattern.split("{date}")
        missing = [d for d in dates
                   if not (data_dir / (head + d + tail)).exists()]
        if missing:
            sys.exit(f"no raw file for {missing} in {data_dir}\n"
                     f"  available: {days_on_disk(data_dir, pattern, missing[0][:6])}")

    cfg = make_config(args.symbol, data_dir, pattern, dates, mode,
                      args.workers, args.predictors)
    names = expected(cfg, mode, dates, args.predictors)

    label = "training" if mode == "monthly" else "validation"
    print(f"\n=== {label}: {args.symbol} | {len(dates)} day(s) "
          f"({dates[0]}..{dates[-1]}) x {len(args.predictors)} predictors "
          f"| class {CLASS_NAME} ===")
    problems = cfg.validate()
    if problems:
        sys.exit("config problems:\n  - " + "\n  - ".join(problems))

    if not args.skip_run:
        run(cfg, run_id=f"sqprb-verify-{mode}-{args.symbol}")
    else:
        print("  (--skip-run: comparing previously produced outputs)")

    out_dir = ROOT / cfg.distributions.output_dir
    ref_dir = Path(args.reference_dir)
    results = []
    for n in names:
        ok, why = compare(out_dir / n, ref_dir / n)
        results.append((n, ok, why))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}\n        {why}")
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--reference-dir", default=REFERENCE_DIR)
    ap.add_argument("--data-dir", default=None,
                    help="raw .dbn.zst directory (default: resolve by symbol)")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--predictors", nargs="+", default=PREDICTORS)
    ap.add_argument("--train-month", default=TRAIN_MONTH)
    ap.add_argument("--validation-dates", nargs="+", default=VALIDATION_DATES)
    ap.add_argument("--only", choices=["training", "validation"], default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-run", action="store_true",
                    help="do not re-run the pipeline; compare what is on disk")
    args = ap.parse_args(argv)

    ref = Path(args.reference_dir)
    if not ref.is_dir():
        sys.exit(f"reference directory not found: {ref}\n"
                 "  pass --reference-dir if his outputs live elsewhere")

    modes = ["monthly", "daily"]
    if args.only:
        modes = ["monthly"] if args.only == "training" else ["daily"]

    results = []
    for m in modes:
        results += stage(args, m)

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 68)
    print(f"{len(results) - len(failed)}/{len(results)} files identical")
    if failed:
        print("\nFAILURES:")
        for n, _, why in failed:
            print(f"  {n}\n      {why}")
        sys.exit(1)
    print("PASS — every compared file matches his reference byte for byte.")


if __name__ == "__main__":
    main()
