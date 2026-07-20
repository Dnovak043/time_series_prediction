# -*- coding: utf-8 -*-
"""
From-zero April run for the boss's request: every April-2025 trading day,
NVDA, INTC, and IBM, 3 predictors each = 9 models, 9x2 result files, 9x4
images.

Each symbol reads from its own raw-data directory (`data.asset_paths` in
`configs/default.yaml`): NVDA and INTC share `data/NVDA_INTC` (interleaved),
IBM has its own `data/IBM`.

Stages (each skippable):

  1. generate configs/april_{nvda,intc,ibm}.yaml
     - dates: every xnas-itch-202504*.dbn.zst present in each symbol's
       resolved data directory (21 days for NVDA_INTC)
     - predictors: tvi_n, obi_L1, ofi_L1_n_norm (features_list[1..3])
     - instrument_filter: true  <-- REQUIRED: NVDA_INTC carries NVDA+INTC
       interleaved and the legacy path never filtered; harmless no-op on a
       single-symbol directory like data/IBM (already just that ticker).
  2. distribution runs (featurize -> encode -> SEQ/CLS pickles), one per
     symbol, day-parallel -> outputs/april/{SYMBOL}/
  3. nine trainings via the verbatim-main() harness
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

SYMBOLS = ["NVDA", "INTC", "IBM"]
PREDICTED = "log_mid"   # features[0] in BOTH colleague files: every output
                        # is a bivariate (log_mid, predictor) pair
PREDICTORS = ["tvi_n", "obi_L1", "ofi_L1_n_norm"]   # TRAINING (boss's 3 models/symbol)
# distribution stage: colleague's full spec (his email / cls_reference.py)
DIST_PREDICTORS = ["tvi_n", "obi_L1", "ofi_L1_n", "ofi_L1_n_norm",
                   "ofi_L1_norm_n", "ofi_L3_norm_n", "ofi_L10_norm_n",
                   "micro_price", "vpin", "sigma_W"]
CLS_NAMES = ["c1", "c2", "c4", "ca2", "ca4"]        # one CLS file per class
MONTH = "202504"

# single source of truth for "which directory holds which asset's raw
# files" — the same mapping the pipeline itself uses
# (DataConfig.asset_paths, RunConfig.resolved_data_path()); several assets
# may share one directory (NVDA_INTC carries both NVDA and INTC interleaved)
ASSET_CATALOG = RunConfig.load(ROOT / "configs" / "default.yaml").data.asset_paths


def find_data_dir(symbol: str) -> Path:
    rel = ASSET_CATALOG.get(symbol)
    if rel is None:
        raise KeyError(f"no data/ directory known for symbol {symbol!r}; "
                       f"add it to data.asset_paths in configs/default.yaml "
                       f"(known: {sorted(ASSET_CATALOG)})")
    local = ROOT / rel
    if local.exists():
        return local
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    candidate = Path(common).parent / rel
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"raw data for {symbol} not found at {local} "
                            f"or {candidate}")


def detect_pattern(data_dir: Path) -> str:
    """Raw files may be zstd-compressed or plain DBN (both read identically
    by databento); use whichever variant this machine's download has."""
    for pattern in ("xnas-itch-{date}.mbp-10.dbn.zst",
                    "xnas-itch-{date}.mbp-10.dbn"):
        prefix, suffix = pattern.split("{date}")
        if any(data_dir.glob(prefix + MONTH + "*" + suffix)):
            return pattern
    raise FileNotFoundError(f"no {MONTH} .dbn/.dbn.zst files in {data_dir}")


def april_dates(data_dir: Path, pattern: str) -> list[str]:
    prefix, suffix = pattern.split("{date}")
    # exact-suffix check so *.dbn does not also swallow *.dbn.zst
    dates = sorted(p.name[len(prefix):-len(suffix)]
                   for p in data_dir.glob(prefix + MONTH + "*" + suffix)
                   if p.name.endswith(suffix))
    if not dates:
        raise FileNotFoundError(f"no {MONTH} files in {data_dir}")
    return dates


def make_config(symbol: str, data_dir: Path, dates: list[str],
                workers: int, predictors: list[str] | None = None,
                file_pattern: str | None = None, epochs: int = 3000,
                n_qubits: int = 3, seed: int = -1,
                train_predictors: list[str] | None = None) -> Path:
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = dates
    if file_pattern:
        cfg.data.file_pattern = file_pattern
    cfg.data.instrument_filter = True
    # colleague's new process_distributions spec: 10 predictors (superset of
    # the 3 training predictors -> their SEQ files come out of the same run),
    # v2 multi-class CLS sweep with (-1,0,1) column order
    cfg.distributions.predicted = PREDICTED
    cfg.distributions.predictors = list(predictors or DIST_PREDICTORS)
    cfg.distributions.class_names = list(CLS_NAMES)
    cfg.distributions.class_values = [-1, 0, 1]
    cfg.distributions.output_dir = f"outputs/april/{symbol}"
    cfg.featurize.workers = workers
    # complete per-symbol separation: own feature cache, outputs, models
    cfg.featurize.cache_dir = f"outputs/april/{symbol}/feature_cache"
    cfg.training.model_dir = f"outputs/april/{symbol}/models"
    cfg.training.epochs = epochs
    cfg.training.n_qubits = n_qubits
    cfg.training.seed = seed
    # the models this experiment trains (boss's 3) — explicit in the config,
    # and what train-all would sweep for this config too
    cfg.training.predictors = list(train_predictors or PREDICTORS)
    cfg.ensemble.output_dir = f"outputs/april/{symbol}/ensemble"
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
    ap.add_argument("--asset-path", action="append", default=[],
                    metavar="SYMBOL=DIR",
                    help="override the raw-data directory for one symbol "
                         "(repeatable), e.g. --asset-path IBM=data/IBM. "
                         "Default: configs/default.yaml's data.asset_paths "
                         "catalog, auto-discovered in/near the repo.")
    ap.add_argument("--with-ensemble", action="store_true",
                    help="also build the fixed-length ENS_TD_* ensemble "
                         "tables per symbol (25 files each; colleague's "
                         "experiment)")
    args = ap.parse_args()

    for override in args.asset_path:
        symbol, _, rel = override.partition("=")
        if not symbol or not rel:
            sys.exit(f"--asset-path expects SYMBOL=DIR, got {override!r}")
        ASSET_CATALOG[symbol] = rel

    # each symbol resolves its own data directory (NVDA/INTC share
    # NVDA_INTC; IBM has its own) — resolve once per distinct directory
    resolved: dict[str, Path] = {}
    for symbol in args.symbols:
        data_dir = find_data_dir(symbol)
        if not data_dir.is_dir():
            sys.exit(f"data dir not found for {symbol}: {data_dir}")
        resolved[symbol] = data_dir

    configs = {}
    scope_by_dir: dict[Path, tuple[str, list[str]]] = {}
    for symbol, data_dir in resolved.items():
        if data_dir not in scope_by_dir:
            pattern = detect_pattern(data_dir)
            dates = april_dates(data_dir, pattern)
            scope_by_dir[data_dir] = (pattern, dates)
            print(f"data: {data_dir}  (pattern: {pattern})  April days: "
                  f"{len(dates)} ({dates[0]}..{dates[-1]})")
        pattern, dates = scope_by_dir[data_dir]
        configs[symbol] = make_config(
            symbol, data_dir, dates, args.workers,
            None, pattern,   # None -> DIST_PREDICTORS
            epochs=args.epochs, n_qubits=args.n_qubits,
            seed=-1 if args.seed is None else args.seed,
            train_predictors=args.predictors)

    # ---- stage 2: distributions -------------------------------------------------
    if not args.skip_distributions:
        from pipeline.runner import run
        for symbol, cfg_path in configs.items():
            cfg = RunConfig.load(cfg_path)      # banner derives from the CONFIG
            print(f"\n=== distributions: {symbol} "
                  f"({len(cfg.data.dates)} days x "
                  f"{len(cfg.distributions.predictors)} predictors x "
                  f"{len(cfg.distributions.class_names) or 1} classes) ===")
            t0 = time.time()
            run(cfg, run_id=f"april-{symbol}")
            print(f"{symbol} distributions done in {time.time()-t0:.0f}s "
                  f"-> outputs/april/{symbol}/")
    # ---- optional: ensemble training tables (ENS_TD_*) --------------------------
    if args.with_ensemble:
        from pipeline.ensemble import run_ensemble
        for symbol, cfg_path in configs.items():
            cfg = RunConfig.load(cfg_path)      # banner derives from the CONFIG
            print(f"\n=== ensemble tables: {symbol} "
                  f"({len(cfg.ensemble.predictors)} channels x "
                  f"{len(cfg.ensemble.seq_lengths)} lengths x "
                  f"{len(cfg.ensemble.class_names)} classes) ===")
            t0 = time.time()
            run_ensemble(cfg, run_id=f"april-ensemble-{symbol}")
            print(f"{symbol} ensemble done in {time.time()-t0:.0f}s "
                  f"-> outputs/april/{symbol}/ensemble/")

    if args.only_distributions:
        return

    # ---- stage 3: 6 trainings, send-as-you-go -----------------------------------
    from train_kraus_baseline import run_one

    summary_path = ROOT / "outputs" / "april" / "april_summary.json"
    results = []
    combos = [(s, p) for s in args.symbols
              for p in RunConfig.load(configs[s]).training.predictors]
    for i, (symbol, predictor) in enumerate(combos, 1):
        distr_dir = ROOT / "outputs" / "april" / symbol
        out_dir = distr_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        t = RunConfig.load(configs[symbol]).training
        print(f"\n=== MODEL {i}/{len(combos)}: {symbol} x {predictor} "
              f"(epochs={t.epochs}, {t.n_qubits}q, batch={t.batch_size}, "
              f"lr={t.lr}, {t.optimizer}/{t.loss_kind}, "
              f"seed={'unseeded' if t.seed < 0 else t.seed}) ===")
        r = run_one(predictor, distr_dir, out_dir, t.epochs,
                    None if t.seed < 0 else t.seed,
                    symbol=symbol, n_qubits=t.n_qubits,
                    m=RunConfig.load(configs[symbol]).alphabet_size,
                    max_seq_len=t.max_seq_len, min_seq_prob=t.min_seq_prob,
                    batch_size=t.batch_size, lr=t.lr,
                    optimizer_name=t.optimizer, loss_kind=t.loss_kind,
                    learn_rho0=t.learn_rho0, device=t.device)
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
