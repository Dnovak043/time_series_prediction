# -*- coding: utf-8 -*-
"""
Baseline Kraus training for three predictors — LearningKraus.main() run
verbatim, three times, changing ONLY the `predictor` selection the way you
would by editing `predictor = features_list[k]` in the script:

    features_list[1] = "tvi_n"
    features_list[2] = "obi_L1"
    features_list[3] = "ofi_L1_n_norm"

Everything else is the original main() body, step for step: same pickle
loading, same max_seq_len/min_seq_prob filtering, same hyperparameters
(epochs=3000, batch_size=6*512, lr=1e-3, adam, nll_seq, learn_rho0=True),
same post-training predict_probs + weighted-MSE cost report, same
plotDistributions call (rendered to a PNG instead of a window), same output
names (including the original's 'MOD'+title[8:] naming quirk).

Two things HAD to be parameterized because the committed main() cannot run
unedited on this machine (it reads a Windows path '..\\Data\\NVDA_INTC\\'
and builds the input name with an underscore where the distribution stage
writes a dash):
  - the input directory: defaults to outputs/baseline_verify/baseline/
    (SEQ_DISTR files produced by the UNTOUCHED main code and byte-verified);
    override with --distr-dir
  - outputs (MOD_*/WGHTS_*/PNG/summary.json) go to outputs/kraus_baseline/

Run (from anywhere):

    .env/bin/python tests/train_kraus_baseline.py                # full 3000 epochs each
    .env/bin/python tests/train_kraus_baseline.py --epochs 50    # quick smoke first
    .env/bin/python tests/train_kraus_baseline.py --seed 0       # reproducible init/shuffle

Notes:
  - default (no --seed) is the original behavior: unseeded random init and
    shuffling, so losses vary run to run. For before/after-optimization
    comparisons pass --seed so both runs start identically.
  - device is chosen exactly like the original: cuda if available else cpu.
    On this Mac that's cpu — expect hours per predictor at 3000 epochs; on
    the A100 box it picks cuda automatically.
  - wall-clock per epoch and final cost land in summary.json: that is the
    baseline any future LearningKraus optimization must be compared against.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainingDistributions"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")             # plotDistributions renders to file, not a window
import matplotlib.pyplot as plt  # noqa: E402

import LearningKraus as lk  # noqa: E402  (guarded: no side effects on import)

PREDICTOR_CHOICES = ["tvi_n", "obi_L1", "ofi_L1_n_norm"]  # features_list[1..3]

# --- constants exactly as in main()'s bivariate branch ---
SYMBOL = "NVDA"
VARIATE = "bivariate"
DATE = "202504"
PREDICTED = "log_mid"          # features_list[0]
MAX_SEQ_LEN = 6
MIN_SEQ_PROB = 0.000000
M = 16                         # observable symbols
N_QUBITS = 3


def find_distr_dir(cli_value: str | None, symbol: str = SYMBOL) -> Path:
    candidates = ([Path(cli_value)] if cli_value else []) + [
        ROOT / "outputs" / "april" / symbol,
        ROOT / "outputs" / "baseline_verify" / "baseline",
        ROOT / "outputs" / "baseline_verify" / "new",
        ROOT,
    ]
    for d in candidates:
        if d.is_dir() and list(d.glob(f"SEQ_DISTR_{symbol}_*_" + DATE)):
            return d
    raise FileNotFoundError(
        f"No SEQ_DISTR_{symbol}_*_{DATE} files found. Run the distribution "
        "stage for that symbol first, or pass --distr-dir.")


def run_one(predictor: str, distr_dir: Path, out_dir: Path,
            epochs: int, seed: int | None,
            symbol: str = SYMBOL, n_qubits: int = N_QUBITS) -> dict:
    # ---- from here on: main()'s bivariate branch, step for step ----
    title = "SEQ_DISTR_" + symbol + "_" + VARIATE + "_" + PREDICTED \
            + "-" + predictor + "_" + DATE   # dash: the name the files really have
    infname = distr_dir / title
    print(f"\n=== {predictor}: training on {infname} ===")
    with open(infname, "rb") as fh:
        distrs_samples = pickle.load(fh)

    sequences_all = distrs_samples[1]
    emp_probs_all = [i[1] for i in distrs_samples[0]]

    sequences, emp_probs = [], []
    for i in range(len(sequences_all)):
        if len(sequences_all[i]) <= MAX_SEQ_LEN and emp_probs_all[i] > MIN_SEQ_PROB:
            sequences.append(sequences_all[i])
            emp_probs.append(emp_probs_all[i])
    print("Number of examples ", len(sequences))

    if seed is not None:                     # optional; original is unseeded
        import torch
        torch.manual_seed(seed)

    t0 = time.time()
    model = lk.train(
        sequences, emp_probs, MAX_SEQ_LEN,
        M, n_qubits,
        batch_size=6 * 512,
        lr=1e-3,
        epochs=epochs,
        learn_rho0=True,
        model=None,
        num_workers=8,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        optimizer_name="adam", loss_kind="nll_seq")
    train_seconds = time.time() - t0

    # ---- post-optimization block, verbatim ----
    p_model = lk.predict_probs(model, sequences, batch_size=2 * 1024)
    total_loss = 0
    for i in range(len(p_model)):
        total_loss = total_loss + emp_probs[i] * (emp_probs[i] - p_model[i]) ** 2

    # plotDistributions draws the first 200 entries in chunks of 62 and calls
    # plt.show() per chunk -> 4 charts per predictor: [0-62], [62-124],
    # [124-186], remainder. Interactively those are 4 windows; here we
    # intercept show() so each becomes its own PNG (named like
    # NVDA_202504_log_mid-tvi_n_3q_1.png ... _4.png).
    fig_base = out_dir / f"{symbol}_{DATE}_{PREDICTED}-{predictor}_{n_qubits}q"
    fig_paths = []

    def _save_instead_of_show(*a, **k):
        fig_paths.append(f"{fig_base}_{len(fig_paths) + 1}.png")
        plt.savefig(fig_paths[-1], dpi=120, bbox_inches="tight")
        plt.close()

    orig_show = plt.show
    plt.show = _save_instead_of_show
    try:
        lk.plotDistributions(emp_probs[:200], p_model[:200], sequences[:200],
                             title[8:] + " Cost=" + str(total_loss),
                             "Target", "Model", c1="blue", c2="red")
    finally:
        plt.show = orig_show
        plt.close("all")

    print("Completed:", total_loss)
    print(title + "_" + str(n_qubits) + "q")
    save_file_name = "MOD" + title[8:] + "_" + str(n_qubits) + "q"  # original quirk kept
    model_file_name = "WGHTS_" + save_file_name + ".pt"

    with open(out_dir / save_file_name, "wb") as fh:
        pickle.dump([model, sequences, emp_probs], fh)
    lk.save_model_weights(
        out_dir / model_file_name, model,
        meta={"m": M, "n_qubits": n_qubits, "d": 2 ** n_qubits,
              "learn_rho0": True, "predictor": predictor, "symbol": symbol,
              "epochs": epochs, "seed": seed})

    return {"symbol": symbol, "predictor": predictor, "examples": len(sequences),
            "epochs": epochs, "train_seconds": round(train_seconds, 1),
            "seconds_per_epoch": round(train_seconds / max(epochs, 1), 3),
            "final_cost_weighted_mse": total_loss,
            "model_pickle": str(out_dir / save_file_name),
            "weights": str(out_dir / model_file_name),
            "plots": fig_paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--predictors", nargs="+", default=PREDICTOR_CHOICES,
                    help=f"default: {PREDICTOR_CHOICES}")
    ap.add_argument("--epochs", type=int, default=3000,
                    help="3000 = the original; use e.g. 50 for a smoke run")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix torch seed per predictor (default: unseeded, "
                         "like the original)")
    ap.add_argument("--distr-dir", default=None,
                    help="directory with SEQ_DISTR_*_202504 files")
    ap.add_argument("--symbol", default=SYMBOL,
                    help="ticker whose SEQ_DISTR files to train on (default NVDA)")
    ap.add_argument("--n-qubits", type=int, default=N_QUBITS,
                    help="system register size (default 3, the committed value)")
    args = ap.parse_args()

    distr_dir = find_distr_dir(args.distr_dir, args.symbol)
    out_dir = ROOT / "outputs" / "kraus_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"inputs:  {distr_dir}\noutputs: {out_dir}")

    results = [run_one(p, distr_dir, out_dir, args.epochs, args.seed,
                       symbol=args.symbol, n_qubits=args.n_qubits)
               for p in args.predictors]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=1))
    print(f"\n=== baseline summary ({summary_path}) ===")
    for r in results:
        print(f"  {r['symbol']:5s} {r['predictor']:16s} examples={r['examples']:7d} "
              f"{r['train_seconds']:8.1f}s ({r['seconds_per_epoch']:.3f}s/epoch) "
              f"cost={r['final_cost_weighted_mse']:.3e}")


if __name__ == "__main__":
    main()
