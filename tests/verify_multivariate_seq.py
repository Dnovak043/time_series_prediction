# -*- coding: utf-8 -*-
"""
Byte-equivalence test: pipeline multivariate SEQ_DISTR output (list entry in
distributions.predictors) vs the colleague's code composed his way, on real
data. This is the deterministic input to LearningKraus_multivariate.py — the
training itself is stochastic (random init + shuffled batches), so the model
files are not byte-comparable; the SEQ_DISTR file and the load-time filtering
are, and both are covered here.

Reference side: the colleague's own functions, verbatim from the vendored
copies. He supplied no multivariate process_distributions driver — his
LearningKraus_multivariate.py *consumes* SEQ_DISTR_{sym}_multivariate_*
files — so the reference composes his pieces exactly the way his bivariate
driver does: his get_timeseries_by_date (featurize), his get_z_ts from
ensemble_reference_2.py (the only multivariate encoding he has defined,
alpha=0.05, n_symbols=4), then the original process_distributions SEQ loop
verbatim — estimate_observed_subsequence_counts (the pure-Python original
from read_databento_new, for maximal independence from the pipeline's torch
call site) with the driver's arguments (sample_size=1, sample_after_length=30,
random_state=42, sort="lexicographic", include_prob=True), the
L/integrate_distributions incremental fold with alphabet
list(range(n_symbols**n_sequences)), the cntsall/distrsall/samplesall
formatting lines, and his multivariate file naming from
LearningKraus_multivariate.py:
    SEQ_DISTR_{sym}_multivariate_{predicted}-{predictors[0]}-{predictors[-1]}_{month}

Pipeline side: `pipeline run` with a single list entry in
distributions.predictors on the same scope. instrument_filter is OFF to
match the colleague's program exactly (his code has no filter) — this test
proves CODE equivalence; production per-symbol runs then turn the filter on.

After the byte comparison, the training input path is checked too: the
colleague's encoder_data_load filtering (LearningKraus_multivariate.py,
max_seq_len/min_seq_prob) is applied to the reference file and compared
element-for-element against pipeline.models.load_sequence_distribution on
the pipeline file.

Default scope: 1 day (20250401), symbol label INTC, his multivariate driver's
predictors ['ofi_L10_norm_n','micro_price','vpin'], max_seq_length=4 (his
multivariate max_seq_len).

Run:  .env/bin/python tests/verify_multivariate_seq.py
      (dominated by the reference side's one legacy featurize pass)

Only READS data/. Outputs under outputs/multivariate_seq_check/.
"""
import argparse
import datetime
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

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
# his multivariate driver's predictors, verbatim
PREDICTORS = ["ofi_L10_norm_n", "micro_price", "vpin"]
N_SYMBOLS = 4
ALPHA = 0.05
MAX_SEQ_LENGTH = 4       # his multivariate driver's max_seq_len
TRAIN_MIN_SEQ_PROB = 0.0  # his encoder_data_load default

SCRATCH = ROOT / "outputs" / "multivariate_seq_check"
REF_DIR = SCRATCH / "reference"
NEW_DIR = SCRATCH / "pipeline"


def ref_name(dates: list) -> str:
    # verbatim from LearningKraus_multivariate.py's driver
    variate = "multivariate"
    return ("SEQ_DISTR_" + SYMBOL + "_" + variate + "_" + PREDICTED + "-"
            + PREDICTORS[0] + "-" + PREDICTORS[-1] + "_" + dates[0][:6])


def run_reference(dates: list, data_dir: Path):
    """His featurize + his get_z_ts + the original SEQ loop, reduced-scope."""
    import ensemble_reference_2 as ref
    from integrate_day_distributions import integrate_distributions
    from read_databento_new import estimate_observed_subsequence_counts

    t_start, t_end = datetime.time(9, 30), datetime.time(15, 30)
    alphabet = list(range(N_SYMBOLS ** (1 + len(PREDICTORS))))

    L = []
    for date in dates:
        args = (str(data_dir), date, "events", 100, [1, 2, 3, 4],
                t_start, t_end)
        t0 = time.time()
        print(f"[reference] {date}: featurize (his get_timeseries_by_date)"
              "...", flush=True)
        date_ts = ref.get_timeseries_by_date(SYMBOL, *args)
        print(f"[reference]   featurized in {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        z_joint = ref.get_z_ts(date_ts, PREDICTED, PREDICTORS,
                               ALPHA, N_SYMBOLS)
        print(f"[reference]   get_z_ts (joint {PREDICTORS}): "
              f"{time.time()-t0:.0f}s", flush=True)

        # original process_distributions SEQ path, verbatim arguments
        bi_series = list(z_joint.astype(int))
        t0 = time.time()
        _, counts = estimate_observed_subsequence_counts(
            seq=bi_series,
            max_subsequence_length=MAX_SEQ_LENGTH,
            sample_size=1,
            sample_after_length=30,
            random_state=42,
            sort="lexicographic",
            include_prob=True,
        )
        print(f"[reference]   counts (pure-Python original): "
              f"{time.time()-t0:.0f}s", flush=True)
        L.append(counts)
        allcounts = integrate_distributions(L, MAX_SEQ_LENGTH, alphabet)
        L = [allcounts]

    # formatting lines, verbatim from the original driver
    cntsall = [[np.array(t[0]).astype(int).tolist(), t[1], t[2]]
               for sublist in allcounts for t in sublist]
    distrsall = [[c[0], c[1] / c[2]] for c in cntsall]
    samplesall = [s[0] for s in cntsall]

    REF_DIR.mkdir(parents=True, exist_ok=True)
    name = ref_name(dates)
    with open(REF_DIR / name, "wb") as fh:
        pickle.dump([distrsall, samplesall], fh)
    print(f"[reference] dumped {name}", flush=True)


def make_pipeline_config(dates: list, data_dir: Path) -> RunConfig:
    cfg = RunConfig()
    cfg.data.symbol = SYMBOL
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = list(dates)
    cfg.data.instrument_filter = False        # match his program exactly
    cfg.featurize.cache_dir = str((SCRATCH / "feature_cache")
                                  .relative_to(ROOT))
    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = [list(PREDICTORS)]  # one list entry
    cfg.distributions.max_seq_length = MAX_SEQ_LENGTH
    cfg.distributions.sequence_calculation = True
    cfg.distributions.class_calculation = False
    cfg.distributions.output_dir = str(NEW_DIR.relative_to(ROOT))
    cfg.training.predictor = list(PREDICTORS)
    cfg.training.max_seq_len = MAX_SEQ_LENGTH
    cfg.training.min_seq_prob = TRAIN_MIN_SEQ_PROB
    problems = cfg.validate()
    if problems:
        raise SystemExit(f"config invalid: {problems}")
    return cfg


def run_pipeline_stage(cfg: RunConfig):
    from pipeline.runner import run

    run(cfg, run_id="multivariate-seq-check")


def compare(dates: list) -> bool:
    name = ref_name(dates)
    ref_p, new_p = REF_DIR / name, NEW_DIR / name
    ok = True
    for side, p in (("reference", ref_p), ("pipeline", new_p)):
        if not p.exists():
            print(f"  FAIL  {name}: missing on {side} side ({p})")
            ok = False
    if not ok:
        return False
    ba, bb = ref_p.read_bytes(), new_p.read_bytes()
    ha = hashlib.sha256(ba).hexdigest()[:16]
    if ba == bb:
        print(f"  IDENTICAL  {name}  sha256:{ha}")
        return True
    print(f"  FAIL  {name}: bytes differ")
    return False


def compare_training_load(cfg: RunConfig, dates: list) -> bool:
    """His encoder_data_load filtering vs pipeline.models'
    load_sequence_distribution — the exact lists handed to train()."""
    from pipeline.models import load_sequence_distribution

    # encoder_data_load body, verbatim from LearningKraus_multivariate.py
    infname = str(REF_DIR / ref_name(dates))
    distrs_samples = pickle.load(open(infname, "rb"))
    sequences_all = distrs_samples[1]
    emp_probs_all = [i[1] for i in distrs_samples[0]]
    sequences = []
    emp_probs = []
    for i in range(len(sequences_all)):
        if (len(sequences_all[i]) <= MAX_SEQ_LENGTH
                and emp_probs_all[i] > TRAIN_MIN_SEQ_PROB):
            sequences.append(sequences_all[i])
            emp_probs.append(emp_probs_all[i])

    pipe_seqs, pipe_probs, path = load_sequence_distribution(cfg, ROOT)
    if sequences == pipe_seqs and emp_probs == pipe_probs:
        print(f"  IDENTICAL  training load: {len(sequences)} sequences "
              f"(his encoder_data_load == load_sequence_distribution on "
              f"{path.name})")
        return True
    print(f"  FAIL  training load differs: reference {len(sequences)} vs "
          f"pipeline {len(pipe_seqs)} sequences")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["20250401"],
                    help="trading days (default: 1 day; pass 2+ to also "
                         "exercise the cross-day aggregation path)")
    ap.add_argument("--data-dir", default=None,
                    help="directory with the raw files (the NVDA_INTC "
                         "folder); default: auto-discover")
    args = ap.parse_args()

    data_dir = (Path(args.data_dir).resolve() if args.data_dir
                else find_data_dir())

    for d in (REF_DIR, NEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("SEQ_DISTR_*"):
            p.unlink()

    print(f"=== 1/2 reference (colleague's code, his composition) — "
          f"{' '.join(args.dates)} ===")
    t0 = time.time()
    run_reference(args.dates, data_dir)
    print(f"reference done in {time.time()-t0:.0f}s")

    print("\n=== 2/2 pipeline run (multivariate predictor), same scope ===")
    cfg = make_pipeline_config(args.dates, data_dir)
    t0 = time.time()
    run_pipeline_stage(cfg)
    print(f"pipeline done in {time.time()-t0:.0f}s")

    print("\n=== comparison ===")
    ok = compare(args.dates)
    ok = compare_training_load(cfg, args.dates) and ok

    if ok:
        print("\nMULTIVARIATE SEQ EQUIVALENCE PASSED: pipeline reproduces "
              "the colleague's multivariate SEQ_DISTR file (and its "
              "training-load filtering) byte-for-byte")
    else:
        print(f"\nMULTIVARIATE SEQ EQUIVALENCE FAILED — outputs kept in "
              f"{SCRATCH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
