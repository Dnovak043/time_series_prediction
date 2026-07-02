# -*- coding: utf-8 -*-
"""
From-zero April run for the boss's request: every April-2025 trading day,
INTC and NVDA, 3 predictors each = 6 models, 6x2 result files, 6x4 images.

Stages (each skippable):

  1. generate configs/april_nvda.yaml + configs/april_intc.yaml
     - dates: every xnas-itch-202504*.dbn.zst present in data/ (21 days)
     - predictors: tvi_n, obi_L1, ofi_L1_n_norm (features_list[1..3])
     - instrument_filter: true  <-- REQUIRED for per-symbol runs: the raw
       files carry NVDA+INTC interleaved and the legacy path never filtered
  2. distribution runs (featurize -> encode -> SEQ/CLS pickles), one per
     symbol, day-parallel -> outputs/april/{SYMBOL}/
  3. six trainings via the verbatim-main() harness
     (tests/train_kraus_baseline.run_one) -> outputs/april/{SYMBOL}/models/
     After EACH model a "READY TO SEND" block lists exactly the 2 result
     files + 4 images for that model, so results can be forwarded as they
     complete. Progress persists in outputs/april/april_summary.json.

Run:
    .env/bin/python scripts/run_april.py                   # everything
    .env/bin/python scripts/run_april.py --epochs 50       # training smoke
    .env/bin/python scripts/run_april.py --only-distributions
    .env/bin/python scripts/run_april.py --skip-distributions --seed 0
    .env/bin/python scripts/run_april.py --n-qubits 4      # if the boss wants 4q

Notes:
  - data/ is only read. Distribution outputs are byte-reproducible; training
    is stochastic unless --seed is given.
  - IMPORTANT: with instrument_filter on, results are per-symbol and will
    NOT match any previous (mixed-stream) outputs — that is the point.
  - runtime, Mac: distributions ~1-2h/symbol cold (21 days, --workers 4);
    training at 3000 epochs is hours per model on CPU. On the Linux box use
    --workers 0 (one per core) and CUDA is picked up automatically.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from pipeline.config import RunConfig  # noqa: E402

SYMBOLS = ["NVDA", "INTC"]
PREDICTORS = ["tvi_n", "obi_L1", "ofi_L1_n_norm"]   # features_list[1..3]
MONTH = "202504"


def find_data_dir() -> Path:
    local = ROOT / "data/NVDA_INTC"
    if local.exists():
        return local
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    candidate = Path(common).parent / "data/NVDA_INTC"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"raw data not found at {local} or {candidate}")


def april_dates(data_dir: Path, pattern: str) -> list[str]:
    prefix, suffix = pattern.split("{date}")
    dates = sorted(p.name[len(prefix):-len(suffix)]
                   for p in data_dir.glob(prefix + MONTH + "*" + suffix))
    if not dates:
        raise FileNotFoundError(f"no {MONTH} files in {data_dir}")
    return dates


def make_config(symbol: str, data_dir: Path, dates: list[str],
                workers: int, predictors: list[str] | None = None) -> Path:
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = dates
    cfg.data.instrument_filter = True
    cfg.distributions.predictors = list(predictors or PREDICTORS)
    cfg.distributions.output_dir = f"outputs/april/{symbol}"
    cfg.featurize.workers = workers
    # complete per-symbol separation: own feature cache, outputs, models
    cfg.featurize.cache_dir = f"outputs/april/{symbol}/feature_cache"
    cfg.training.model_dir = f"outputs/april/{symbol}/models"
    path = ROOT / "configs" / f"april_{symbol.lower()}.yaml"
    cfg.save(path)
    print(f"wrote {path}  ({len(dates)} days, filter ON, workers={workers})")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--predictors", nargs="+", default=PREDICTORS)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, default=3,
                    help="3 = committed value; boss's earlier charts were 4q")
    ap.add_argument("--workers", type=int, default=4,
                    help="day-parallel workers for distributions "
                         "(4 is Mac-RAM-safe; 0 = one per core on the box)")
    ap.add_argument("--skip-distributions", action="store_true")
    ap.add_argument("--only-distributions", action="store_true")
    args = ap.parse_args()

    data_dir = find_data_dir()
    dates = april_dates(data_dir, "xnas-itch-{date}.mbp-10.dbn.zst")
    print(f"data: {data_dir}\nApril days: {len(dates)} "
          f"({dates[0]}..{dates[-1]})")

    configs = {s: make_config(s, data_dir, dates, args.workers)
               for s in args.symbols}

    # ---- stage 2: distributions -------------------------------------------------
    if not args.skip_distributions:
        from pipeline.runner import run
        for symbol, cfg_path in configs.items():
            print(f"\n=== distributions: {symbol} "
                  f"({len(dates)} days x {len(args.predictors)} predictors) ===")
            t0 = time.time()
            run(RunConfig.load(cfg_path), run_id=f"april-{symbol}")
            print(f"{symbol} distributions done in {time.time()-t0:.0f}s "
                  f"-> outputs/april/{symbol}/")
    if args.only_distributions:
        return

    # ---- stage 3: 6 trainings, send-as-you-go -----------------------------------
    from train_kraus_baseline import run_one

    summary_path = ROOT / "outputs" / "april" / "april_summary.json"
    results = []
    combos = [(s, p) for s in args.symbols for p in args.predictors]
    for i, (symbol, predictor) in enumerate(combos, 1):
        distr_dir = ROOT / "outputs" / "april" / symbol
        out_dir = distr_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== MODEL {i}/{len(combos)}: {symbol} x {predictor} "
              f"(epochs={args.epochs}, {args.n_qubits}q) ===")
        r = run_one(predictor, distr_dir, out_dir, args.epochs, args.seed,
                    symbol=symbol, n_qubits=args.n_qubits)
        results.append(r)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=1))

        print(f"\n>>> MODEL {i}/{len(combos)} COMPLETE — READY TO SEND:")
        print(f"    result file 1: {r['model_pickle']}")
        print(f"    result file 2: {r['weights']}")
        for j, png in enumerate(r["plots"], 1):
            print(f"    image {j}:       {png}")
        print(f"    cost={r['final_cost_weighted_mse']:.3e}  "
              f"({r['train_seconds']:.0f}s)")

    print(f"\nAll {len(combos)} models done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
