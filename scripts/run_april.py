# -*- coding: utf-8 -*-
"""
From-zero April run: every April-2025 trading day, NVDA, INTC, and IBM,
reproducing the colleague's two current drivers (his 2026-02
LearningKraus.py and LearningKraus_multivariate.py, run for AAPL) per
symbol: 3 bivariate models + 1 multivariate model = 4 models/symbol,
12 total, 12x2 result files + 12x4 images.

Each symbol reads from its own raw-data directory (`data.asset_paths` in
`configs/default.yaml`): NVDA and INTC share `data/NVDA_INTC` (interleaved),
IBM has its own `data/IBM`.

The two drivers train with DIFFERENT hyperparameters (his bivariate script
lands on sgd/batch 8*512/3 qubits/max_seq_len 6; his multivariate driver
uses adam/batch 6*512/6 qubits/max_seq_len 4), and one RunConfig holds one
training block — so stage 1 writes TWO configs per symbol, one per
experiment group (TRAIN_GROUPS below), and stage 3 plans over all of them.

Stages (each skippable):

  1. generate configs/april_{nvda,intc,ibm}.yaml           (bivariate group)
          + configs/april_{...}_multivariate.yaml          (multivariate group)
     - dates: every xnas-itch-202504*.dbn.zst present in each symbol's
       resolved data directory (21 days for NVDA_INTC)
     - training predictors: group 0 = vpin, ofi_L10_norm_n, micro_price
       (bivariate, one model each); group 1 = the joint
       [ofi_L10_norm_n, micro_price, vpin] predictor (one model,
       WGHTS tag `L10_micro_vpin` from his driver)
     - instrument_filter: true  <-- REQUIRED: NVDA_INTC carries NVDA+INTC
       interleaved and the legacy path never filtered; harmless no-op on a
       single-symbol directory like data/IBM (already just that ticker).
  2. distribution runs (featurize -> encode -> SEQ/CLS pickles), one per
     SYMBOL (the two group configs share every distribution setting, so the
     bivariate config runs it and the multivariate config reuses the
     outputs), day-parallel -> outputs/april/{SYMBOL}/. The predictor list
     includes the multivariate entry, so the joint SEQ_DISTR_*_multivariate_*
     file is produced in the same pass.
  3. twelve trainings via the pipeline's registered trainer (pipeline.models,
     the same path as `python -m pipeline train`), fanned one per GPU
     -> outputs/april/{SYMBOL}/models/
     After EACH model a "READY TO SEND" block lists exactly the 2 result
     files + 4 images for that model, so results can be forwarded as they
     complete. Progress persists in outputs/april/april_summary.json.

Run:
    .env/bin/python scripts/run_april.py                   # everything
    .env/bin/python scripts/run_april.py --epochs 50       # training smoke
    .env/bin/python scripts/run_april.py --only-configs    # just stage 1
    .env/bin/python scripts/run_april.py --only-distributions
    .env/bin/python scripts/run_april.py --skip-distributions --seed 0
    .env/bin/python scripts/run_april.py --only-ensemble-models
        # stage 4 alone (LearningEnsemble): per (symbol, class) multi-encoder
        # ensembles from the ENS_TD_* + WGHTS_* files of a finished run

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
import os
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
PREDICTED = "log_mid"   # features_list[0] in BOTH colleague drivers: every
                        # output pairs log_mid with the predictor(s)

# TRAINING: one group per colleague driver. Every value here is written into
# that group's generated config (stage 1) and read back from it (stage 3) —
# the dicts only feed the config generator, never the run directly.
#   group 0 = his 2026-02 LearningKraus.py bivariate script body. Its
#     opt_name reassignment chain lands on "sgd" and batch_size on 8*512;
#     he re-ran it once per predictor — the three he kept are below.
#   group 1 = his LearningKraus_multivariate.py driver: the joint
#     (log_mid, ofi_L10_norm_n, micro_price, vpin) encoding, alphabet
#     m = 4^4 = 256, 6 qubits, max_seq_len 4, adam, batch 6*512;
#     `L10_micro_vpin` is his hand-written file-name abbreviation.
MULTI_PREDICTOR = ["ofi_L10_norm_n", "micro_price", "vpin"]
TRAIN_GROUPS = [
    dict(suffix="", predictors=["vpin", "ofi_L10_norm_n", "micro_price"],
         n_qubits=3, batch_size=8 * 512, optimizer="sgd", max_seq_len=6,
         predictor_abbrev=""),
    dict(suffix="_multivariate", predictors=[MULTI_PREDICTOR],
         n_qubits=6, batch_size=6 * 512, optimizer="adam", max_seq_len=4,
         predictor_abbrev="L10_micro_vpin"),
]
PREDICTORS = TRAIN_GROUPS[0]["predictors"]          # bivariate models/symbol
# distribution stage: colleague's full spec (his email / cls_reference.py)
# + the multivariate joint entry (SEQ only — he defines no multivariate CLS)
DIST_PREDICTORS = ["tvi_n", "obi_L1", "ofi_L1_n", "ofi_L1_n_norm",
                   "ofi_L1_norm_n", "ofi_L3_norm_n", "ofi_L10_norm_n",
                   "micro_price", "vpin", "sigma_W", MULTI_PREDICTOR]
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
                workers: int, predictors: list | None = None,
                file_pattern: str | None = None, epochs: int = 3000,
                n_qubits: int = 3, seed: int = -1,
                train_predictors: list | None = None,
                predicted: str | None = None,
                cls_names: list[str] | None = None,
                gpus: str = "auto", max_parallel: int = 0,
                batch_size: int | None = None, optimizer: str | None = None,
                max_seq_len: int | None = None,
                predictor_abbrev: str | None = None,
                config_suffix: str = "") -> Path:
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(data_dir)
    cfg.data.dates = dates
    if file_pattern:
        cfg.data.file_pattern = file_pattern
    cfg.data.instrument_filter = True
    # colleague's new process_distributions spec: 10 bivariate predictors
    # (superset of the 3 training predictors -> their SEQ files come out of
    # the same run) + the multivariate joint entry, v2 multi-class CLS sweep
    # with (-1,0,1) column order
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
    # the models this config trains — explicit in the config, and what
    # train-all would sweep for this config too; `predictor` (singular) is
    # the group's first model so `pipeline train` on this config alone is
    # meaningful
    cfg.training.predictors = list(train_predictors or PREDICTORS)
    cfg.training.predictor = cfg.training.predictors[0]
    # per-group hyperparameters (None = keep the TrainingConfig default,
    # which is audited against the original LearningKraus.main())
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if optimizer is not None:
        cfg.training.optimizer = optimizer
    if max_seq_len is not None:
        cfg.training.max_seq_len = max_seq_len
    if predictor_abbrev is not None:
        cfg.training.predictor_abbrev = predictor_abbrev
    # stage-3 scheduling is a run parameter like any other: it lands in the
    # generated config, and the fan-out reads it back from there rather than
    # from a CLI/notebook constant (CLAUDE.md rule 4)
    cfg.training.gpus = gpus
    cfg.training.max_parallel = max_parallel
    cfg.ensemble.output_dir = f"outputs/april/{symbol}/ensemble"
    # stage 4 (LearningEnsemble): per-class models under this base — the
    # class name becomes a subdirectory because his ENS_MD_* file name
    # carries no class tag and would otherwise self-overwrite
    cfg.ensemble_model.model_dir = f"outputs/april/{symbol}/ensemble_models"
    path = ROOT / "configs" / f"april_{symbol.lower()}{config_suffix}.yaml"
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
    # train_model reports the device it actually resolved to; keep that as
    # the record of what ran and note the request separately, so a silent
    # fallback can never be reported as success on the requested GPU
    r["requested_device"] = job["device"]
    return r


def plan_training(configs: dict) -> tuple:
    """(jobs, concurrency) for stage 3 — one pass over the configs.

    Schedule and jobs are built together so the concurrency can never be
    clamped against a job count that differs from the list actually run,
    and so each generated config is read exactly once.

    training.gpus / training.max_parallel come from the config — the same
    fields `pipeline train-all` uses — so the schedule is recorded in
    configs/april_*.yaml and reproducible from it alone. make_config writes
    them identically for every symbol; a disagreement means someone
    hand-edited one, so it is reported rather than silently first-wins.
    """
    if not configs:
        raise ValueError("no configs to schedule: nothing to train")

    loaded = {s: RunConfig.load(p) for s, p in configs.items()}
    schedules = {(c.training.gpus, c.training.max_parallel)
                 for c in loaded.values()}
    if len(schedules) > 1:
        raise ValueError(
            "configs disagree on the stage-3 schedule (training.gpus, "
            f"training.max_parallel): {sorted(schedules)}. They are "
            "generated together, so this means one was hand-edited.")

    first = next(iter(loaded.values())).training
    gpus = visible_gpu_ids(first.gpus)
    jobs = build_training_jobs(configs, gpus, loaded=loaded)
    if not jobs:
        raise ValueError(
            "no trainings to run: every config has empty training.predictors "
            "and empty distributions.predictors. Set one of them.")
    n_par = first.max_parallel or (len(gpus) or 1)
    return jobs, max(1, min(n_par, len(jobs) or 1))


def build_training_jobs(configs: dict, gpus: list | None = None,
                        loaded: dict | None = None) -> list:
    """One job per (symbol, training predictor), round-robin across GPUs.

    `configs` maps symbol -> generated config path; the training parameters
    are read back from that config, so the jobs stay config-authoritative.
    `loaded` lets a caller that already parsed those configs avoid re-reading
    them.
    """
    if gpus is None:
        gpus = visible_gpu_ids(
            RunConfig.load(next(iter(configs.values()))).training.gpus)
    jobs = []
    for symbol, cfg_path in configs.items():
        cfg = (loaded or {}).get(symbol) or RunConfig.load(cfg_path)
        # training.predictors is documented as "empty = train all of
        # distributions.predictors" and pipeline.parallel.train_all honours
        # that; iterating it directly made an empty list silently mean
        # "train nothing"
        for predictor in (list(cfg.training.predictors)
                          or list(cfg.distributions.predictors)):
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


def run_training_jobs(jobs: list, max_parallel: int = 0, worker=None):
    """Yield each finished job as it completes (completion order, not
    submission order). max_parallel: 0 = one per distinct device (i.e. one
    per GPU), 1 = sequential (the original behavior), N = N at once.
    `worker` is the module-level job function (default: the stage-3 Kraus
    trainer; stage 4 passes _train_one_ensemble_model)."""
    worker = worker or _train_one
    if max_parallel <= 0:
        max_parallel = len({j["device"] for j in jobs})
    max_parallel = max(1, min(max_parallel, len(jobs)))

    if max_parallel == 1:
        for job in jobs:
            yield worker(job)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_parallel,
                             mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in as_completed(futures):
            yield fut.result()


# ---------------------------------------------------------------------------
# stage 4: LearningEnsemble — one multi-encoder ensemble per (symbol, class)
# ---------------------------------------------------------------------------
def _train_one_ensemble_model(job: dict) -> dict:
    """One (symbol, class) ensemble training pinned to one device.
    Module-level for the same spawn-context reason as _train_one."""
    import matplotlib
    matplotlib.use("Agg")
    from pipeline.ensemble_model import train_ensemble_model

    cfg = RunConfig.load(job["config"])
    cfg.ensemble_model.device = job["device"]
    r = train_ensemble_model(cfg, job["class_name"], repo_root=ROOT)
    r["requested_device"] = job["device"]
    return r


def plan_ensemble_models(configs: dict) -> tuple:
    """(jobs, concurrency) for stage 4: one job per (symbol, class), from
    ensemble_model.class_names of each symbol's config, scheduled by the
    same training.gpus / training.max_parallel fields as stage 3."""
    if not configs:
        raise ValueError("no configs to schedule: nothing to train")

    loaded = {s: RunConfig.load(p) for s, p in configs.items()}
    schedules = {(c.training.gpus, c.training.max_parallel)
                 for c in loaded.values()}
    if len(schedules) > 1:
        raise ValueError(
            "configs disagree on the schedule (training.gpus, "
            f"training.max_parallel): {sorted(schedules)}. They are "
            "generated together, so this means one was hand-edited.")

    first = next(iter(loaded.values())).training
    gpus = visible_gpu_ids(first.gpus)
    jobs = []
    for symbol, cfg_path in configs.items():
        for class_name in loaded[symbol].ensemble_model.class_names:
            device = f"cuda:{gpus[len(jobs) % len(gpus)]}" if gpus else "cpu"
            jobs.append({
                "symbol": symbol,
                "class_name": class_name,
                "label": class_name,
                "config": str(cfg_path),
                "device": device,
                "run_id": f"april-ensmodel-{symbol}-{class_name}",
            })
    if not jobs:
        raise ValueError("no ensemble models to train: every config has an "
                         "empty ensemble_model.class_names")
    n_par = first.max_parallel or (len(gpus) or 1)
    return jobs, max(1, min(n_par, len(jobs)))


def run_ensemble_model_stage(dist_configs: dict) -> list:
    """Stage 4 at the console: plan, fan out, report each model."""
    jobs, n_par = plan_ensemble_models(dist_configs)
    devices = sorted({j["device"] for j in jobs})
    print(f"\n=== {len(jobs)} ensemble models (symbol x class) | "
          f"{len(devices)} device(s) | {n_par} at a time ===")
    for j in jobs:
        em = RunConfig.load(j["config"]).ensemble_model
        print(f"    {j['symbol']:6s} x {j['label']:4s} -> {j['device']}"
              f"  (epochs={em.epochs}, batch={em.batch_size}, lr={em.lr}, "
              f"{em.prediction_loss}, seq_lens={em.seq_lens})")

    summary_path = ROOT / "outputs" / "april" / "ensemble_models_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, r in enumerate(run_training_jobs(jobs, n_par,
                                            worker=_train_one_ensemble_model),
                          1):
        results.append(r)
        summary_path.write_text(json.dumps(results, indent=1))
        print(f"\n>>> ENSEMBLE MODEL {i}/{len(jobs)} COMPLETE — "
              f"{r['symbol']} x {r['class_name']} on {r['device']}:")
        print(f"    model file: {r['model_file']}")
        if "agreement_pct" in r:
            print(f"    agreement: {r['agreement_pct']:.2f}% unique | "
                  f"{r['weighted_agreement_pct']:.2f}% weighted | "
                  f"{r['count_weighted_agreement_pct']:.2f}% by occurrence")
        print(f"    ({r['train_seconds']:.0f}s, "
              f"{r['n_sequences']} joint sequences)")
    print(f"\nAll {len(jobs)} ensemble models done. Summary: {summary_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--predictors", nargs="+", default=None,
                    help="override the BIVARIATE group's training "
                         "predictors (default: vpin ofi_L10_norm_n "
                         "micro_price, the colleague's three runs); the "
                         "multivariate group is fixed by his driver")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, default=None,
                    help="override the BIVARIATE group's register size "
                         "(default 3, his driver); the multivariate group "
                         "stays at its driver's 6")
    ap.add_argument("--workers", type=int, default=4,
                    help="day-parallel workers for distributions "
                         "(4 is Mac-RAM-safe; 0 = one per core on the box)")
    ap.add_argument("--only-configs", action="store_true",
                    help="stop after writing configs/april_*.yaml")
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
    ap.add_argument("--with-ensemble-models", action="store_true",
                    help="stage 4: also train the LearningEnsemble multi-"
                         "encoder models — one per (symbol, class in "
                         "ensemble_model.class_names), fanned across GPUs")
    ap.add_argument("--only-ensemble-models", action="store_true",
                    help="ONLY stage 4, from existing outputs (needs the "
                         "ENS_TD_* tables and the 4 WGHTS_* encoders from "
                         "a previous run; skips stages 2 and 3)")
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

    # one config per (symbol, experiment group): the groups differ in their
    # training hyperparameters and one config holds one training block
    configs = {}                       # "{symbol}{suffix}" -> config path
    dist_configs = {}                  # symbol -> group-0 path (stage 2 runs once)
    scope_by_dir: dict[Path, tuple[str, list[str]]] = {}
    for symbol, data_dir in resolved.items():
        if data_dir not in scope_by_dir:
            pattern = detect_pattern(data_dir)
            dates = april_dates(data_dir, pattern)
            scope_by_dir[data_dir] = (pattern, dates)
            print(f"data: {data_dir}  (pattern: {pattern})  April days: "
                  f"{len(dates)} ({dates[0]}..{dates[-1]})")
        pattern, dates = scope_by_dir[data_dir]
        for i, group in enumerate(TRAIN_GROUPS):
            bivariate = group["suffix"] == ""
            configs[symbol + group["suffix"]] = make_config(
                symbol, data_dir, dates, args.workers,
                None, pattern,   # None -> DIST_PREDICTORS
                epochs=args.epochs,
                n_qubits=(args.n_qubits if bivariate and args.n_qubits
                          else group["n_qubits"]),
                seed=-1 if args.seed is None else args.seed,
                train_predictors=(args.predictors if bivariate and
                                  args.predictors else group["predictors"]),
                gpus=args.gpus, max_parallel=args.max_parallel,
                batch_size=group["batch_size"], optimizer=group["optimizer"],
                max_seq_len=group["max_seq_len"],
                predictor_abbrev=group["predictor_abbrev"],
                config_suffix=group["suffix"])
            if i == 0:
                dist_configs[symbol] = configs[symbol + group["suffix"]]

    if args.only_configs:
        return

    if args.only_ensemble_models:
        # stage 4 alone, over outputs a previous run already produced
        run_ensemble_model_stage(dist_configs)
        return

    # ---- stage 2: distributions -------------------------------------------------
    # once per SYMBOL, not per config: the group configs share every
    # distribution setting (predictor list incl. the multivariate entry,
    # output dir, cache), so one pass serves both training groups
    if not args.skip_distributions:
        from pipeline.runner import run
        for symbol, cfg_path in dist_configs.items():
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
        for symbol, cfg_path in dist_configs.items():
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

    jobs, n_par = plan_training(configs)             # from the CONFIG
    devices = sorted({j["device"] for j in jobs})

    print(f"\n=== {len(jobs)} models | {len(devices)} device(s) | "
          f"{n_par} at a time ===")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # the single most common cause of "everything ran on one GPU":
        # torch only sees (and renumbers) the devices this names, and
        # training.gpus 'auto' deliberately schedules within it
        print(f"    NOTE: CUDA_VISIBLE_DEVICES={cvd!r} restricts this run — "
              f"torch sees only these GPUs, renumbered from cuda:0")
    # per-model hyperparameters, not one symbol's standing in for all nine
    for j in jobs:
        t = RunConfig.load(j["config"]).training
        print(f"    {j['symbol']:6s} x {j['label']:24s} -> {j['device']}"
              f"  (epochs={t.epochs}, {t.n_qubits}q, batch={t.batch_size}, "
              f"lr={t.lr}, {t.optimizer}/{t.loss_kind}, "
              f"seed={'unseeded' if t.seed < 0 else t.seed})")

    for i, r in enumerate(run_training_jobs(jobs, n_par), 1):
        results.append(r)
        summary_path.write_text(json.dumps(results, indent=1))
        ran_on = r["device"]
        if r.get("requested_device") not in (None, ran_on):
            ran_on = f"{ran_on} (requested {r['requested_device']})"
        print(f"\n>>> MODEL {i}/{len(jobs)} COMPLETE — "
              f"{r['symbol']} x {r['predictor']} on {ran_on} — "
              f"READY TO SEND:")
        print(f"    result file 1: {r['model_file']}")
        print(f"    result file 2: {r['weights_file']}")
        for j2, png in enumerate(r["plots"], 1):
            print(f"    image {j2}:       {png}")
        if r.get("chart_error"):
            print(f"    NOTE: charts failed ({r['chart_error']}); "
                  f"model + weights above are complete")
        print(f"    cost={r['loss']:.3e}  ({r['train_seconds']:.0f}s)")

    print(f"\nAll {len(jobs)} models done. Summary: {summary_path}")

    # ---- stage 4 (optional): LearningEnsemble multi-encoder models --------------
    if args.with_ensemble_models:
        run_ensemble_model_stage(dist_configs)


if __name__ == "__main__":
    main()
