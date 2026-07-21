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
  3. nine trainings via the pipeline's registered trainer (pipeline.models,
     the same path as `python -m pipeline train`), fanned one per GPU
     -> outputs/april/{SYMBOL}/models/
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
from pipeline.config import RunConfig, predictor_key  # noqa: E402

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
                train_predictors: list[str] | None = None,
                predicted: str | None = None,
                cls_names: list[str] | None = None,
                gpus: str = "auto", max_parallel: int = 0) -> Path:
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
    cfg.distributions.predicted = predicted or PREDICTED
    cfg.distributions.predictors = list(predictors or DIST_PREDICTORS)
    cfg.distributions.class_names = list(cls_names or CLS_NAMES)
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
    # stage-3 scheduling is a run parameter like any other: it lands in the
    # generated config, and the fan-out reads it back from there rather than
    # from a CLI/notebook constant (CLAUDE.md rule 4)
    cfg.training.gpus = gpus
    cfg.training.max_parallel = max_parallel
    cfg.ensemble.output_dir = f"outputs/april/{symbol}/ensemble"
    path = ROOT / "configs" / f"april_{symbol.lower()}.yaml"
    cfg.save(path)
    print(f"wrote {path}  ({len(dates)} days, filter ON, workers={workers})")
    return path


# ---------------------------------------------------------------------------
# training fan-out across GPUs
#
# The 3-qubit bivariate model is tiny (m=16 operators of d=8, ~2k parameters),
# so a single training occupies only a few percent of an A100 — it is
# kernel-launch-latency bound, not compute bound. Running the 9 models
# sequentially therefore leaves 7 of 8 GPUs idle. These helpers dispatch one
# training per device instead.
#
# The trainer is pipeline.models (the model registry) — the same code path
# as `python -m pipeline train`. Only the cross-symbol scheduling lives here,
# because `pipeline train-all` sweeps predictors within ONE config and this
# experiment needs 3 symbols x 3 predictors spread over 8 GPUs.
# ---------------------------------------------------------------------------
def visible_gpu_ids(spec: str = "auto") -> list:
    """GPU ids to schedule on — the pipeline's own discovery, so this and
    `pipeline train-all` agree on what exists. [] = no CUDA (schedule CPU)."""
    from pipeline.parallel import visible_gpus
    return visible_gpus(spec)


def _train_one(job: dict) -> dict:
    """One (symbol, predictor) training pinned to one device.

    Runs the registered pipeline trainer (pipeline.models, via the model
    registry) — not a test harness. Everything that used to be exclusive to
    tests/train_kraus_baseline.py (the 4 plotDistributions charts, seeding,
    the dataloader worker count) now lives in pipeline/models.py, so this is
    a plain `pipeline train` with training.predictor/device overridden.

    Module-level on purpose: ProcessPoolExecutor's spawn context imports the
    worker by qualified name, which a notebook-cell closure cannot satisfy.
    """
    import matplotlib
    matplotlib.use("Agg")
    from pipeline.models import train_model

    cfg = RunConfig.load(job["config"])
    cfg.training.predictor = job["predictor"]
    cfg.training.device = job["device"]
    r = train_model(cfg, run_id=job["run_id"], repo_root=ROOT)
    r["device"] = job["device"]
    return r


def training_schedule(configs: dict) -> tuple:
    """(gpu ids, concurrency) for stage 3, taken from the config.

    Both come from the generated config — training.gpus and
    training.max_parallel, the same fields `pipeline train-all` uses — so
    the schedule is recorded in configs/april_*.yaml and reproducible from
    it alone. make_config writes them identically for every symbol, so the
    first one is authoritative.
    """
    cfg = RunConfig.load(next(iter(configs.values())))
    gpus = visible_gpu_ids(cfg.training.gpus)
    n_jobs = sum(len(RunConfig.load(p).training.predictors)
                 for p in configs.values())
    n_par = cfg.training.max_parallel or (len(gpus) or 1)
    return gpus, max(1, min(n_par, max(1, n_jobs)))


def build_training_jobs(configs: dict, gpus: list | None = None) -> list:
    """One job per (symbol, training predictor), round-robin across GPUs.

    `configs` maps symbol -> generated config path; the training parameters
    are read back from that config, so the jobs stay config-authoritative.
    """
    gpus = training_schedule(configs)[0] if gpus is None else gpus
    jobs = []
    for symbol, cfg_path in configs.items():
        cfg = RunConfig.load(cfg_path)
        for predictor in cfg.training.predictors:
            device = f"cuda:{gpus[len(jobs) % len(gpus)]}" if gpus else "cpu"
            # a multivariate predictor is a list; `label` is the printable
            # '+'-joined form (predictor_key), so every display/format site
            # has a string and none of them has to re-handle the list case
            label = predictor_key(predictor)
            jobs.append({
                "symbol": symbol,
                "predictor": predictor,
                "label": label,
                "config": str(cfg_path),
                "device": device,
                # own run dir per model: progress.json + config.yaml, so each
                # training is watchable individually via `pipeline status`
                "run_id": f"april-train-{symbol}-{label}",
            })
    return jobs


def run_training_jobs(jobs: list, max_parallel: int = 0):
    """Yield each finished training as it completes (completion order, not
    submission order). max_parallel: 0 = one per distinct device (i.e. one
    per GPU), 1 = sequential (the original behavior), N = N at once."""
    if max_parallel <= 0:
        max_parallel = len({j["device"] for j in jobs})
    max_parallel = max(1, min(max_parallel, len(jobs)))

    if max_parallel == 1:
        for job in jobs:
            yield _train_one(job)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_parallel,
                             mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(_train_one, j) for j in jobs]
        for fut in as_completed(futures):
            yield fut.result()


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
    ap.add_argument("--train-parallel", type=int, default=0,
                    metavar="N", dest="max_parallel",
                    help="concurrent trainings in stage 3 -> written to the "
                         "config as training.max_parallel. 0 = auto (one per "
                         "visible GPU), 1 = sequential, N = exactly N. Each "
                         "3-qubit model uses only a few percent of an A100, "
                         "so N above the GPU count is reasonable.")
    ap.add_argument("--gpus", default="auto",
                    help="GPUs for stage 3 -> written to the config as "
                         "training.gpus. 'auto' = every CUDA device torch "
                         "sees (respects CUDA_VISIBLE_DEVICES), '0,2,5' = "
                         "those ids, 'none' = force CPU.")
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
            train_predictors=args.predictors,
            gpus=args.gpus, max_parallel=args.max_parallel)

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

    # ---- stage 3: trainings, fanned across GPUs, send-as-you-go -----------------
    summary_path = ROOT / "outputs" / "april" / "april_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    train_configs = {s: configs[s] for s in args.symbols}
    gpus, n_par = training_schedule(train_configs)   # from the CONFIG
    jobs = build_training_jobs(train_configs, gpus)

    t0cfg = RunConfig.load(configs[args.symbols[0]]).training
    print(f"\n=== {len(jobs)} models | {len(gpus) or 'no'} GPU(s) | "
          f"{n_par} at a time | epochs={t0cfg.epochs}, {t0cfg.n_qubits}q, "
          f"batch={t0cfg.batch_size}, lr={t0cfg.lr}, "
          f"{t0cfg.optimizer}/{t0cfg.loss_kind}, "
          f"seed={'unseeded' if t0cfg.seed < 0 else t0cfg.seed} ===")
    for j in jobs:
        print(f"    {j['symbol']:6s} x {j['label']:24s} -> {j['device']}")

    for i, r in enumerate(run_training_jobs(jobs, n_par), 1):
        results.append(r)
        summary_path.write_text(json.dumps(results, indent=1))
        print(f"\n>>> MODEL {i}/{len(jobs)} COMPLETE — "
              f"{r['symbol']} x {r['predictor']} on {r['device']} — "
              f"READY TO SEND:")
        print(f"    result file 1: {r['model_file']}")
        print(f"    result file 2: {r['weights_file']}")
        for j2, png in enumerate(r["plots"], 1):
            print(f"    image {j2}:       {png}")
        print(f"    cost={r['loss']:.3e}  ({r['train_seconds']:.0f}s)")

    print(f"\nAll {len(jobs)} models done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
