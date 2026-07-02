# -*- coding: utf-8 -*-
"""
Smoke test for `python -m pipeline train-all` (multi-GPU training sweep).

Builds tiny SYNTHETIC SEQ_DISTR_* pickles for three fake predictors (no
market data needed), then runs the real train-all path — real subprocesses,
real scheduler, real progress files — with a 20-epoch toy model, and checks
that every predictor produced trained weights.

Run anywhere:
    .env/bin/python tests/test_train_all_smoke.py      (~1-2 min)

On the A100 box this also exercises GPU assignment: the report prints which
device each child was scheduled on (expect cuda:0/1/2 with >=3 GPUs, all
cuda:0 with one, cpu on the Mac).
"""
import pickle
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402,F401
from pipeline.config import RunConfig  # noqa: E402
from pipeline.parallel import train_all, visible_gpus  # noqa: E402

SCRATCH = ROOT / "outputs" / "train_all_smoke"
PREDICTORS = ["fakeA", "fakeB", "fakeC"]


def synthetic_seq_distr(rng: random.Random):
    """Same shape as a real SEQ_DISTR_* payload: ([[seq, prob], ...], samples)
    over the 16-symbol bivariate alphabet, lengths 1-3, normalized per length."""
    distrs = []
    for length in (1, 2, 3):
        seqs = {tuple(rng.randrange(16) for _ in range(length))
                for _ in range(30)}
        weights = [rng.random() for _ in seqs]
        total = sum(weights)
        distrs += [[list(s), w / total] for s, w in zip(sorted(seqs), weights)]
    return distrs, [d[0] for d in distrs]


def main():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)

    cfg = RunConfig()
    cfg.distributions.predictors = list(PREDICTORS)
    cfg.distributions.output_dir = str(SCRATCH.relative_to(ROOT))
    cfg.training.model_dir = str(SCRATCH.relative_to(ROOT))
    cfg.training.epochs = 20
    cfg.training.batch_size = 64
    cfg.training.n_qubits = 2

    rng = random.Random(7)
    for p in PREDICTORS:
        name = cfg.seq_distr_name(p)       # same naming rule the trainer uses
        with open(SCRATCH / name, "wb") as fh:
            pickle.dump(synthetic_seq_distr(rng), fh)

    print(f"visible GPUs: {visible_gpus('auto') or 'none (CPU mode)'}")
    results = train_all(cfg, run_id="train-all-smoke")

    ok = True
    for p in PREDICTORS:
        r = results.get(p, {})
        weights = list(SCRATCH.glob(f"WGHTS_*{p}*"))
        good = r.get("status") == "completed" and weights
        print(f"  {'PASS' if good else 'FAIL'}  {p}: status={r.get('status')} "
              f"device={r.get('device')} weights={[w.name for w in weights]}")
        if not good:
            ok = False
            log = r.get("log")
            if log and Path(log).exists():
                print("  --- last log lines ---")
                print("\n".join(Path(log).read_text().splitlines()[-15:]))
    if not ok:
        sys.exit(1)
    print("\nTRAIN-ALL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
