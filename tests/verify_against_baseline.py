# -*- coding: utf-8 -*-
"""
Ground-truth verification against the frozen `baseline` git tag
(= main @ 0182a85, locked on GitHub).

What it does — fully transparent, step by step:

  1. Checks out the `baseline` tag into a temporary git worktree
     (outputs/baseline_verify/src). That code is byte-identical to main;
     nothing in it is modified, wrapped, or monkeypatched.
  2. Runs the ORIGINAL TrainingDistributions/process_distributions.py from
     that checkout as a plain subprocess, exactly as committed: its own
     hardcoded dates (2 days), all 8 predictors, absolute data path.
     Its SEQ_DISTR_* / CLS_DISTR_* pickles land in
     outputs/baseline_verify/baseline/.
  3. Reads the scope variables (dates, feature list, symbol, data path,
     session times, resampling, frequency, ...) out of the BASELINE SOURCE
     via ast parsing — no hand-copied values — and builds the matching
     pipeline config from them (saved to outputs/baseline_verify/new.yaml
     so you can inspect exactly what the new code was asked to do).
  4. Runs `python -m pipeline run --config new.yaml` (the code under test,
     i.e. whatever branch this repo is currently on). Outputs land in
     outputs/baseline_verify/new/.
  5. Compares the two output directories file by file, byte for byte, in
     both directions (missing files fail too), printing sha256 of each.

Exit code 0 = every output identical. Anything else = FAIL.

Run it yourself (this script is meant to be run by YOU, not by automation):

    cd <repo root>
    .env/bin/python tests/verify_against_baseline.py          # full (~1h)
    .env/bin/python tests/verify_against_baseline.py --fast   # ~4 min

Full mode re-derives the ground truth: checks out the baseline tag, runs the
untouched original driver, byte-compares, and on PASS writes
tests/baseline_manifest.json — full sha256 of every baseline output plus the
parsed scope, the tag commit, and the environment (platform/numpy/pandas).
COMMIT that file: it is the golden reference minted by your own passing run.

--fast skips the 55-minute baseline re-run entirely: the frozen code + frozen
data are deterministic, so the manifest hashes ARE the baseline. It runs only
the new pipeline and compares its outputs to the manifest. Use it for routine
regression after changes; re-run full mode only if the manifest's environment
no longer matches (float bit-patterns can differ across numpy builds — the
script warns) or for a from-scratch audit.

Expect roughly 45-60 minutes on the Mac: the original driver re-featurizes
every day once per predictor (2 dates x 8 predictors), by design — that is
the untouched reference behavior. Its console output streams live so you
can watch it.

Notes:
  * Mac only: the baseline code hardcodes the absolute data path
    /Users/ilk085528/time_series_prediction/data/NVDA_INTC.
  * data/ is only ever read, never written.
  * Re-running is safe: previous outputs and the temporary worktree are
    removed first.
"""
import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_REF = "baseline"

SCRATCH = ROOT / "outputs" / "baseline_verify"
SRC = SCRATCH / "src"            # temporary worktree of the baseline tag
BASELINE_OUT = SCRATCH / "baseline"
NEW_OUT = SCRATCH / "new"
CONFIG_PATH = SCRATCH / "new.yaml"

PY = sys.executable


# ---------------------------------------------------------------------------
# step 1: pristine checkout of the baseline tag
# ---------------------------------------------------------------------------
def checkout_baseline():
    subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                    str(SRC)], capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach",
                    str(SRC), BASELINE_REF], check=True)
    head = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True
                          ).stdout.strip()
    print(f"baseline checkout at commit {head}")
    # prove the checked-out driver is bit-identical to the tag
    diff = subprocess.run(["git", "-C", str(SRC), "diff", BASELINE_REF, "--"],
                          capture_output=True, text=True, check=True).stdout
    if diff:
        sys.exit("FATAL: baseline worktree differs from the tag:\n" + diff)


# ---------------------------------------------------------------------------
# step 3 helper: read the driver's scope out of the baseline source itself
# ---------------------------------------------------------------------------
def parse_baseline_globals(path: Path) -> dict:
    """Extract module-level assignments from the pristine script.

    Only literals, `datetime.time(H, M)` calls, and name aliases are
    resolved; the LAST top-level assignment wins (matching execution order).
    Everything extracted is printed so you can eyeball it against the
    source.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    g: dict = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        v = node.value
        try:
            g[name] = ast.literal_eval(v)
            continue
        except (ValueError, SyntaxError):
            pass
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr == "time"
                and all(isinstance(a, ast.Constant) for a in v.args)):
            h, m = (a.value for a in v.args[:2])
            g[name] = f"{h:02d}:{m:02d}"
        elif isinstance(v, ast.Name) and v.id in g:
            g[name] = g[v.id]
        elif (isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name)
                and v.value.id in g and isinstance(v.slice, ast.Constant)):
            g[name] = g[v.value.id][v.slice.value]   # e.g. features[0]
    required = ["fPath", "symbol", "dates", "features", "predicted",
                "resampling", "frequency", "frw_intervals", "tStart", "tEnd",
                "alpha", "n_symbols", "max_seq_length", "clsName",
                "num_classes"]
    missing = [k for k in required if k not in g]
    if missing:
        sys.exit(f"FATAL: could not parse {missing} from baseline source")
    print("scope parsed from baseline source:")
    for k in required:
        print(f"  {k} = {g[k]!r}")
    return g


# ---------------------------------------------------------------------------
# step 2: run the untouched original driver
# ---------------------------------------------------------------------------
def run_baseline():
    script = SRC / "TrainingDistributions" / "process_distributions.py"
    BASELINE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== running UNTOUCHED baseline driver: {script}")
    print("=== (its own scope: hardcoded dates x all predictors; "
          "output streams below) ===\n")
    t0 = time.time()
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([PY, str(script)], cwd=BASELINE_OUT, env=env)
    print(f"\nbaseline driver finished in {time.time()-t0:.0f}s "
          f"(exit {r.returncode})")
    if r.returncode != 0:
        sys.exit("FATAL: baseline driver failed")


# ---------------------------------------------------------------------------
# steps 3+4: run the new pipeline with the scope parsed from the baseline
# ---------------------------------------------------------------------------
def run_new(g: dict):
    sys.path.insert(0, str(ROOT))
    from pipeline.config import RunConfig

    cfg = RunConfig()
    cfg.data.symbol = g["symbol"]
    cfg.data.data_path = g["fPath"]                 # absolute, read-only
    cfg.data.dates = list(g["dates"])
    cfg.data.session_start = g["tStart"]
    cfg.data.session_end = g["tEnd"]
    cfg.featurize.resampling = g["resampling"]
    cfg.featurize.frequency = g["frequency"]
    cfg.featurize.forward_intervals = list(g["frw_intervals"])
    cfg.featurize.use_cache = False                 # no cache: straight run
    cfg.encode.n_symbols = g["n_symbols"]
    cfg.encode.alpha = g["alpha"]
    cfg.distributions.predicted = g["predicted"]
    cfg.distributions.predictors = [f for f in g["features"]
                                    if f != g["predicted"]]
    cfg.distributions.max_seq_length = g["max_seq_length"]
    cfg.distributions.class_name = g["clsName"]
    cfg.distributions.num_classes = g["num_classes"]
    cfg.distributions.output_dir = str(NEW_OUT.relative_to(ROOT))
    NEW_OUT.mkdir(parents=True, exist_ok=True)
    cfg.save(CONFIG_PATH)
    print(f"\nconfig for the new pipeline written to {CONFIG_PATH}")

    print("\n=== running the NEW pipeline (code on the current branch) ===\n")
    t0 = time.time()
    r = subprocess.run([PY, "-m", "pipeline", "run",
                        "--config", str(CONFIG_PATH)], cwd=ROOT)
    print(f"\nnew pipeline finished in {time.time()-t0:.0f}s "
          f"(exit {r.returncode})")
    if r.returncode != 0:
        sys.exit("FATAL: new pipeline failed")


# ---------------------------------------------------------------------------
# step 5: byte-for-byte comparison, both directions
# ---------------------------------------------------------------------------
def compare() -> bool:
    a_files = {p.name for p in BASELINE_OUT.glob("*_DISTR_*")}
    b_files = {p.name for p in NEW_OUT.glob("*_DISTR_*")}
    ok = True
    if not a_files:
        print("FAIL: baseline produced no output files")
        return False
    for name in sorted(a_files | b_files):
        if name not in a_files:
            print(f"  FAIL  {name}: only produced by the NEW pipeline")
            ok = False
            continue
        if name not in b_files:
            print(f"  FAIL  {name}: only produced by the BASELINE")
            ok = False
            continue
        ba = (BASELINE_OUT / name).read_bytes()
        bb = (NEW_OUT / name).read_bytes()
        ha = hashlib.sha256(ba).hexdigest()[:16]
        hb = hashlib.sha256(bb).hexdigest()[:16]
        if ba == bb:
            print(f"  IDENTICAL  {name}  sha256:{ha}")
        else:
            print(f"  FAIL  {name}: bytes differ "
                  f"(baseline sha256:{ha} vs new sha256:{hb})")
            ok = False
    return ok


MANIFEST = ROOT / "tests" / "baseline_manifest.json"


def env_info() -> dict:
    import numpy, pandas
    return {"platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": numpy.__version__, "pandas": pandas.__version__}


def write_manifest(g: dict):
    tag_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", BASELINE_REF],
        capture_output=True, text=True, check=True).stdout.strip()
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "bytes": p.stat().st_size}
             for p in sorted(BASELINE_OUT.glob("*_DISTR_*"))}
    MANIFEST.write_text(json.dumps({
        "baseline_commit": tag_commit,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "environment": env_info(),
        "scope": g,
        "files": files,
    }, indent=1))
    print(f"\ngolden manifest written: {MANIFEST}")
    print("COMMIT IT — future --fast runs verify against these hashes "
          "without re-running the baseline.")


def compare_manifest(manifest: dict) -> bool:
    new_files = {p.name for p in NEW_OUT.glob("*_DISTR_*")}
    golden = manifest["files"]
    ok = True
    for name in sorted(set(golden) | new_files):
        if name not in new_files:
            print(f"  FAIL  {name}: in manifest but not produced")
            ok = False
            continue
        if name not in golden:
            print(f"  FAIL  {name}: produced but not in manifest")
            ok = False
            continue
        h = hashlib.sha256((NEW_OUT / name).read_bytes()).hexdigest()
        if h == golden[name]["sha256"]:
            print(f"  IDENTICAL  {name}  sha256:{h[:16]}")
        else:
            print(f"  FAIL  {name}: sha256 {h[:16]} != manifest "
                  f"{golden[name]['sha256'][:16]}")
            ok = False
    return ok


def cleanup_worktree():
    subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                    str(SRC)], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="verify against tests/baseline_manifest.json "
                         "instead of re-running the frozen baseline (~4 min "
                         "vs ~1h). Requires a committed manifest from a "
                         "previous full PASS.")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    for d in (BASELINE_OUT, NEW_OUT):
        d.mkdir(parents=True, exist_ok=True)
        if d is BASELINE_OUT and args.fast:
            continue
        for f in d.glob("*_DISTR_*"):
            f.unlink()

    if args.fast:
        if not MANIFEST.exists():
            sys.exit("No tests/baseline_manifest.json — run full mode once "
                     "to mint it (your own passing run writes it).")
        manifest = json.loads(MANIFEST.read_text())
        env_now, env_then = env_info(), manifest["environment"]
        if env_now != env_then:
            print("WARNING: environment differs from the manifest's — float "
                  "bit-patterns are not guaranteed identical across builds.")
            print(f"  manifest: {env_then}\n  now:      {env_now}")
            print("  A mismatch below may be environmental, not a code bug; "
                  "re-run full mode on this machine to re-mint if needed.")
        print(f"golden manifest: baseline {manifest['baseline_commit'][:9]}, "
              f"generated {manifest['generated']}")
        run_new(manifest["scope"])
        print("\n=== comparison vs golden manifest ===")
        ok = compare_manifest(manifest)
        if ok:
            print("\nVERIFICATION PASSED (fast): new pipeline matches the "
                  "golden baseline hashes")
        else:
            print(f"\nVERIFICATION FAILED — outputs kept in {SCRATCH}")
            sys.exit(1)
        return

    checkout_baseline()
    g = parse_baseline_globals(
        SRC / "TrainingDistributions" / "process_distributions.py")
    try:
        run_baseline()
        run_new(g)
        print(f"\n=== comparison (baseline={BASELINE_OUT}) ===")
        ok = compare()
    finally:
        cleanup_worktree()
    if ok:
        print("\nVERIFICATION PASSED: new pipeline reproduces the frozen "
              "baseline byte-for-byte")
        write_manifest(g)
    else:
        print("\nVERIFICATION FAILED — outputs kept in "
              f"{SCRATCH} for inspection")
        sys.exit(1)


if __name__ == "__main__":
    main()
