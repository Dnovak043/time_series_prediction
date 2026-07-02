# -*- coding: utf-8 -*-
"""
Byte-identity test for the day-parallel runner.

Runs the full distribution pipeline on real data (2 days x 2 predictors)
twice — serial (featurize.workers=1) and parallel (workers=2) — and
requires every SEQ_DISTR_*/CLS_DISTR_* output to be byte-for-byte identical.
The parallel path completes days in nondeterministic order; the runner must
fold them into the monthly aggregate in date order, and this test is what
holds it to that.

Run:  .env/bin/python tests/test_parallel_run.py       (~a few minutes;
      the two runs share the feature cache, so day featurization happens
      once)

Only READS data/. Outputs go to outputs/parallel_check/.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from parity_check import find_data_dir  # noqa: E402  (data/ locator)
from pipeline.config import RunConfig  # noqa: E402
from pipeline.runner import run  # noqa: E402

DATES = ["20250401", "20250402"]      # 2 days so fold order matters
PREDICTORS = ["tvi_n", "obi_L1"]
SCRATCH = ROOT / "outputs" / "parallel_check"


def make_config(out: Path, workers: int) -> RunConfig:
    cfg = RunConfig()
    cfg.data.data_path = str(find_data_dir())
    cfg.data.dates = list(DATES)
    cfg.distributions.predictors = list(PREDICTORS)
    cfg.distributions.output_dir = str(out.relative_to(ROOT))
    cfg.featurize.workers = workers
    cfg.featurize.cache_dir = str((SCRATCH / "feature_cache").relative_to(ROOT))
    return cfg


def main():
    serial_out = SCRATCH / "serial"
    parallel_out = SCRATCH / "parallel"
    for d in (serial_out, parallel_out):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*_DISTR_*"):
            p.unlink()

    print("=== serial run (workers=1) ===")
    t0 = time.time()
    run(make_config(serial_out, workers=1), run_id="parallel-check-serial")
    print(f"serial done in {time.time()-t0:.0f}s")

    print("=== parallel run (workers=2) ===")
    t0 = time.time()
    run(make_config(parallel_out, workers=2), run_id="parallel-check-parallel")
    print(f"parallel done in {time.time()-t0:.0f}s")

    names = sorted(p.name for p in serial_out.glob("*_DISTR_*"))
    ok = bool(names)
    if not names:
        print("FAIL: serial run produced no outputs")
    for name in names:
        pb = parallel_out / name
        if not pb.exists():
            print(f"  FAIL  {name}: missing from parallel run")
            ok = False
        elif (serial_out / name).read_bytes() == pb.read_bytes():
            print(f"  IDENTICAL (byte-for-byte)  {name}")
        else:
            print(f"  FAIL  {name}: bytes differ")
            ok = False
    extra = {p.name for p in parallel_out.glob('*_DISTR_*')} - set(names)
    if extra:
        print(f"  FAIL  extra outputs in parallel run: {sorted(extra)}")
        ok = False

    if not ok:
        print("\nPARALLEL RUN CHECK FAILED")
        sys.exit(1)
    print("\nPARALLEL RUN CHECK PASSED (serial == parallel, byte-for-byte)")


if __name__ == "__main__":
    main()
