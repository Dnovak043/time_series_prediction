# -*- coding: utf-8 -*-
"""
Parity check: legacy driver vs new pipeline runner, on real data.

Runs both paths on a reduced scope (1 date, 2 predictors) and compares the
SEQ_DISTR_* / CLS_DISTR_* outputs structurally and byte-wise. Also runs the
new pipeline a second time (served from the feature cache) to prove the
cache roundtrip changes nothing.

Only READS from data/. All outputs go to a scratch directory.

Run:  .env/bin/python tests/parity_check.py        (takes several minutes)
"""
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from pipeline.config import RunConfig  # noqa: E402
from pipeline.runner import run  # noqa: E402

DATES = ["20250401"]
PREDICTED = "log_mid"
PREDICTORS = ["tvi_n", "obi_L1"]


def find_data_dir() -> Path:
    """data/ is untracked (10GB raw); in a git worktree it only exists in the
    main checkout, so fall back to locating it via the common git dir."""
    local = ROOT / "data/NVDA_INTC"
    if local.exists():
        return local
    import subprocess
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    main_root = Path(common).parent
    candidate = main_root / "data/NVDA_INTC"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"raw data not found at {local} or {candidate}")


DATA_DIR = find_data_dir()

SCRATCH = ROOT / "outputs" / "parity_check"
LEGACY_OUT = SCRATCH / "legacy"
NEW_OUT = SCRATCH / "new"
NEW_OUT2 = SCRATCH / "new_cached"


def deep_equal(a, b, path="root"):
    """Structural equality with a helpful path on first mismatch."""
    import numpy as np
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = deep_equal(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        if float(a) != float(b):          # exact, not approximate
            return f"{path}: {a!r} != {b!r}"
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def run_legacy():
    import process_distributions as pdst
    # the guarded driver reads module globals at call time, so the test can
    # shrink its scope without editing source
    pdst.dates = list(DATES)
    pdst.features = [PREDICTED] + PREDICTORS
    pdst.predicted = PREDICTED
    pdst.fPath = str(DATA_DIR)

    LEGACY_OUT.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(LEGACY_OUT)                  # legacy dumps into CWD
    try:
        t0 = time.time()
        pdst.run_legacy_driver()
        print(f"legacy driver done in {time.time()-t0:.0f}s")
    finally:
        os.chdir(cwd)


def make_config(output_dir: Path, use_cache: bool) -> RunConfig:
    cfg = RunConfig()
    cfg.data.data_path = str(DATA_DIR)   # absolute paths are honored as-is
    cfg.data.dates = list(DATES)
    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = list(PREDICTORS)
    cfg.distributions.output_dir = str(output_dir.relative_to(ROOT))
    cfg.training.predictor = PREDICTORS[0]
    cfg.featurize.use_cache = use_cache
    cfg.featurize.cache_dir = str((SCRATCH / "feature_cache").relative_to(ROOT))
    return cfg


def compare(dir_a: Path, dir_b: Path, label: str) -> bool:
    ok = True
    for name in sorted(p.name for p in dir_a.glob("*_DISTR_*")):
        pa, pb = dir_a / name, dir_b / name
        if not pb.exists():
            print(f"  MISSING in {dir_b}: {name}")
            ok = False
            continue
        ba, bb = pa.read_bytes(), pb.read_bytes()
        if ba == bb:
            print(f"  IDENTICAL (byte-for-byte)  {name}")
            continue
        mismatch = deep_equal(pickle.loads(ba), pickle.loads(bb))
        if mismatch is None:
            print(f"  IDENTICAL (structurally; pickle bytes differ)  {name}")
        else:
            print(f"  MISMATCH  {name}: {mismatch}")
            ok = False
    print(f"{label}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    for d in (LEGACY_OUT, NEW_OUT, NEW_OUT2):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*_DISTR_*"):
            p.unlink()

    print("=== 1/3 legacy driver (scope: 1 date x 2 predictors) ===")
    run_legacy()

    print("=== 2/3 new pipeline, cold (featurizes + fills cache) ===")
    t0 = time.time()
    run(make_config(NEW_OUT, use_cache=True), run_id="parity-new")
    print(f"new runner done in {time.time()-t0:.0f}s")

    print("=== 3/3 new pipeline again, warm (from feature cache) ===")
    t0 = time.time()
    run(make_config(NEW_OUT2, use_cache=True), run_id="parity-cached")
    print(f"cached runner done in {time.time()-t0:.0f}s")

    ok1 = compare(LEGACY_OUT, NEW_OUT, "legacy vs new")
    ok2 = compare(NEW_OUT, NEW_OUT2, "cold vs cache-served")
    if ok1 and ok2:
        print("PARITY CHECK PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
