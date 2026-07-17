# Pipeline Guide — configurable architecture & control panel

Companion to `ARCHITECTURE_AND_PERFORMANCE.md` (which maps the research code
and its bottlenecks). This document maps the **configurable layer** added on
top: what the pieces are, every variable you can turn, and how to drive it
from the ipywidgets control panel, the CLI, or plain Python — locally or on
the Linux compute box over SSH / JupyterLab.

## 1. The architecture in one picture

```
                       run.yaml  (RunConfig — single source of truth)
                          │  edited by: control panel · CLI · any text editor
     ┌────────────┬───────┴─────┬───────────────┬───────────────┐
     ▼            ▼             ▼               ▼               ▼
 [data]      [featurize]    [encode]     [distributions]   [training]
 which raw   events/sec/    EWMA z +     subsequence &     model registry
 days/files  volume bars,   quantile     class-conditional (kraus today,
 RTH window  OFI/VPIN/vol   symbols      histograms        pluggable)
     │            │             │               │               │
     ▼            ▼             ▼               ▼               ▼
 data/*.dbn.zst → DayFeatureCache → DistributionBuilder → SEQ_/CLS_DISTR_* → MOD_/WGHTS_*
 (read-only!)    (parquet cache,    (pure per-day        (pickles,          (trained models)
                  1 decode/day)      function)            same format
                                                          as before)
```

- **`pipeline/config.py`** — one dataclass per stage; YAML in/out; field
  help-text and choices live here, and the UI/CLI are generated from it.
  *Add a field here and it appears in the control panel automatically.*
- **`pipeline/features.py`** — `DayFeatureCache`: each raw day is decoded and
  featurized **once** per parameter-set (keyed cache under
  `outputs/feature_cache/`), then shared by every predictor. The original
  driver refeaturized per predictor (8× waste, bottleneck #2). `data/` is
  never written.
- **`pipeline/distributions.py`** — `DistributionBuilder`: featurized day →
  per-day sequence/class counts. Same math, same call arguments as the
  original driver; every hardcoded constant is now a config field.
- **`pipeline/runner.py`** — the inverted-loop batch driver (days outer,
  predictors inner), incremental monthly aggregation identical to the
  original, atomic `progress.json` per run, detached-subprocess launching.
- **`pipeline/models.py`** — model registry. `kraus` wraps
  `LearningKraus.py`'s training unchanged; future model families register
  with a decorator and appear in the UI/CLI dropdowns.
- **`pipeline/ui.py` + `pipeline_control.ipynb`** — the control panel.
- **Legacy scripts still work**: `process_distributions.py` and
  `LearningKraus.py` now only execute when run directly (`__main__` guards);
  their functions are imported unchanged by the pipeline.

## 2. Every knob, by stage

Hover any field in the control panel for the same help text.

### data — which raw market data (never modified)
| field | default | meaning |
|---|---|---|
| `symbol` | NVDA | ticker used in file naming and outputs |
| `data_path` | data/NVDA_INTC | directory with raw `.dbn.zst` files |
| `file_pattern` | xnas-itch-{date}.mbp-10.dbn.zst | raw file name per day |
| `dates` | [20250401, 20250402] | trading days (yyyymmdd) |
| `instrument_filter` | false | **true = filter events to `symbol` before featurizing.** The raw files carry NVDA+INTC interleaved; false reproduces the legacy (mixed-stream) behavior and the frozen baseline. Set true for per-symbol runs (see `scripts/run_april.py`). |
| `session_start` / `session_end` | 09:30 / 15:30 | Eastern-time RTH window |

### featurize — event stream → resampled LOB feature bars
| field | default | meaning |
|---|---|---|
| `resampling` | events | bar clock: `events`, `seconds`, or `volume` |
| `frequency` | 100 | bar size: N events / N seconds / N shares |
| `forward_intervals` | [1,2,3,4] | horizons for `log_mid_return_fwd_k` etc. |
| `vol_window` | 0 (auto) | trailing window W for `sigma_W`; 0 = 3×frequency |
| `workers` | 1 | parallel day workers for `run`: 0 = auto (per core), N = exactly N; output byte-identical for any value |
| `use_cache` / `cache_dir` / `cache_format` | true / outputs/feature_cache / parquet | featurized-day cache |

### encode — continuous features → discrete symbols
| field | default | meaning |
|---|---|---|
| `n_symbols` | 4 | symbols per variate (bivariate alphabet = n²=16) |
| `alpha` | 0.05 | EWMA decay of the predictive z-score |
| `bins_mode` | expanding_quantile | how bin edges are fitted (5 modes) |
| `fill_mode` | bfill | warmup/NaN handling of symbols |
| `min_periods` | 2 | observations before z is defined |

### distributions — symbol series → empirical distributions
| field | default | meaning |
|---|---|---|
| `predicted` | log_mid | first variate (the thing being predicted) |
| `predictors` | 8 features | second variate; one SEQ/CLS output pair each |
| `max_seq_length` | 6 | max n-gram length |
| `sequence_calculation` / `class_calculation` | true / true | which outputs to compute |
| `class_name` | c1 | forward-move class: `c{k}` return-sign, `ca{k}` fwd-vs-bwd sum |
| `class_names` | [] (legacy) | **v2 multi-class sweep** (colleague's new process_distributions, vendored as `cls_reference.py`): one CLS file per class, named `CLS_DISTR_{sym}__{predicted}-{predictor}_{month}_{cls}`, columns ordered by `class_values`. Empty = legacy single-class mode: old `[P(0),P(+1),P(−1)]` order and naming, matching the frozen baseline. Verified by user-run `tests/verify_cls_v2.py`. |
| `class_values` | [-1,0,1] | class column order in v2 mode (`[P(−1),P(0),P(+1)]`); ignored in legacy mode |
| `class_theta` | 0 (auto) | class threshold θ; 0 = built-in default per class |
| `num_classes` | 3 | down / flat / up |
| `sample_size` / `sample_after_length` / `random_state` | 1.0 / 30 / 42 | optional support subsampling for long n-grams |
| `output_dir` | . | where SEQ_DISTR_*/CLS_DISTR_* land (repo root = legacy behavior) |

### ensemble — fixed-length multi-channel training tables (ENS_TD_*)
| field | default | meaning |
|---|---|---|
| `reference` | v2 | which vendored colleague program the stage reproduces (see below) |
| `seq_lengths` | [1..5] | fixed window lengths; one output set per length |
| `class_names` | [c1,c2,ca2,ca4] | class definitions swept; one output set per (length, class); v1's driver also swept c4 |
| `class_values` | [-1,0,1] | class column order of the output distributions |
| `predictors` | 3 bivariate + 1 joint channel | a string = one bivariate encoding vs `predicted`; a nested list (v2 only) = one joint encoding of predicted + the listed features (alphabet n_symbols^(1+len)); all timestamp-aligned by inner join |
| `smoothing` | 0.0 | Dirichlet pseudocount for target class distributions |
| `output_dir` | outputs/ensemble | where ENS_TD_* pickles land |

Counting math is imported verbatim from `TrainingDistributions/`
`ensemble_reference.py` (v1, the colleague's ensemble_training_data.py) or
`ensemble_reference_2.py` (v2, his ensemble_training_data_2.py — adds
multivariate joint channels via `get_z_ts`, drops the fixed alphabet_size
validation, and names files `..._ALL` instead of `..._ALL_{n}`), both
vendored with an import guard; the stage replaces only the plumbing —
cached featurize + encode-once instead of thousands of redundant decodes
per month. Verification (run it yourself):
`tests/verify_ensemble_reference.py` byte-compares the v1 stage and
`tests/verify_ensemble_v2.py` the v2 stage (default: 1 day, 20 files)
against the colleague's own functions run his way.

### training — SEQ_DISTR_* → trained model
| field | default | meaning |
|---|---|---|
| `model` | kraus | trainer from the registry (future models plug in here) |
| `predictor` | tvi_n | which predictor's distribution to train on |
| `seq_distr_file` | (derived) | explicit pickle path override |
| `n_qubits` | 3 | Hilbert dim d = 2^n_qubits |
| `epochs` / `batch_size` / `lr` / `optimizer` | 3000 / 3072 / 1e-3 / adam | optimization |
| `loss_kind` | nll_seq | `nll_seq` or `mse_prob` |
| `length_mixture` | uniform | length reweighting (`uniform`/`geometric`/`none`) |
| `learn_rho0` | true | learn the initial state vs fixed \|0⟩⟨0\| |
| `max_seq_len` / `min_seq_prob` | 6 / 0 | training-set filters |
| `device` | auto | auto = cuda→cpu; `mps` opt-in (complex-op support is limited) |
| `continue_from` | — | WGHTS_*.pt to resume |
| `model_dir` | . | where MOD_*/WGHTS_* are written |
| `predictors` | [] (= all) | which predictors `train-all` sweeps over |
| `gpus` | auto | GPUs for `train-all`: all visible / `0,2,5` / `none` |
| `max_parallel` | 0 (auto) | concurrent trainings; auto = one per GPU, else 1 |

## 3. Three ways to drive it (same config file)

**Control panel (JupyterLab or VS Code notebook):** open
`pipeline_control.ipynb`, run the first two cells (build, then display — kept
separate because VS Code's renderer can drop a large widget tree displayed in
the cell that creates it). Edit fields in the stage tabs →
*Save* → *Run distributions* / *Train model* in the **run & monitor** tab.
Launches are **detached subprocesses**: they survive kernel restarts and SSH
drops; use *Attach* to re-monitor any run after reconnecting. The **results**
tab plots any saved SEQ/CLS distribution.

**CLI (any SSH terminal):**
```bash
python -m pipeline init-config run.yaml    # write defaults, edit in any editor
python -m pipeline validate  --config run.yaml
python -m pipeline featurize --config run.yaml   # warm the day cache only
python -m pipeline run       --config run.yaml   # distributions
python -m pipeline ensemble  --config run.yaml   # ENS_TD_* ensemble tables
python -m pipeline train     --config run.yaml   # model training
python -m pipeline status                        # latest run's progress
```

**Python:**
```python
from pipeline.config import RunConfig
from pipeline import runner, models
cfg = RunConfig.load("run.yaml"); cfg.encode.alpha = 0.08
runner.run(cfg)               # blocking
models.train_model(cfg)
```

## 4. Working from the Windows VM

Both target environments work the same way because everything renders in a
browser/notebook and heavy work runs detached on the Linux box:

- **JupyterLab server:** open `pipeline_control.ipynb` in JupyterLab; done.
  If the server isn't exposed, tunnel it:
  `ssh -L 8888:localhost:8888 user@linux-box` then browse
  `http://localhost:8888`.
- **VS Code Remote-SSH:** open the repo remotely, open the notebook with the
  Jupyter extension (ipywidgets render natively), select the project venv as
  kernel.
- **Plain SSH terminal:** the CLI above; `python -m pipeline status` or
  `cat outputs/runs/<id>/progress.json` to watch progress; `run.log` in the
  same folder has full output.

A run's folder `outputs/runs/<run_id>/` always contains the exact
`config.yaml` it was launched with (provenance), `progress.json`, and
`run.log`.

## 5. Extending

- **New parameter:** add a field to the right dataclass in
  `pipeline/config.py` (with `help=`/`choices=` metadata), consume it in the
  stage code. UI + YAML + CLI pick it up automatically.
- **New model family** (planned): implement
  `@register_model("name") def train_name(cfg, progress=None, repo_root=None)`
  in `pipeline/models.py`, add its hyperparameters to `TrainingConfig`, and
  add the name to the `model` field's `choices`. It becomes launchable from
  the UI/CLI like `kraus`.
- **New feature column:** add it in `add_event_features_and_resample`
  (process_distributions.py) as before; it's then valid in
  `distributions.predictors`.

## 6. Verification & the frozen baseline

**Trust model.** `main` is frozen at commit `0182a85` as the permanent ground
truth: it is **locked on GitHub** (read-only until unlocked in repo settings
→ Branches) and additionally pinned by the annotated git tag **`baseline`**,
which survives even if branches move. All PRs target the `dev` integration
branch, never main. Any change, on any branch, must reproduce the baseline's
outputs byte-for-byte — verified by a script that is run **by the user**, not
by automation.

- `tests/verify_against_baseline.py` — the ground-truth check. Checks the
  `baseline` tag out into a temporary worktree, runs the **untouched**
  original `process_distributions.py` from it as committed (own hardcoded
  scope: 2 dates × 8 predictors, ~45–60 min, Mac only — the baseline
  hardcodes the absolute data path), reads that scope back out of the
  baseline *source* via ast parsing (nothing hand-copied), runs the new
  pipeline with it (cache off), and byte-compares every SEQ/CLS output in
  both directions with sha256 printed per file. Exit 0 = identical.
- `tests/parity_check.py` — fast development check (1 date × 2 predictors,
  ~8 min): guarded legacy driver vs new runner, plus cold-vs-cache-served.
  Uses the legacy code *on the current branch*, so it is a convenience
  check, not the trust anchor — `verify_against_baseline.py` is.
- `tests/test_pipeline_units.py` — config/YAML/UI-widget roundtrips, cache
  keying, import-safety of the guarded legacy scripts (seconds).
- `compare_main_vs_dev.ipynb` — notebook view of the same evidence: hash
  summary table plus numeric/visual overlays of main's vs dev's
  distributions, re-openable anytime after a `verify_against_baseline.py`
  run (it only reads `outputs/baseline_verify/`).

Note: the torch unfold+bincount histogram from PR #1 is merged and active in
**both** paths — the legacy driver and `DistributionBuilder.class_counts` —
so the fast dev parity check compares like against like. The pure-Python
original lives on in the frozen `baseline` tag, which is exactly what
`verify_against_baseline.py` runs against.

## 7. GPU & parallel execution (any hardware, 8× A100 target)

Design rule: **GPU where the FLOPs are, vectorized CPU where bandwidth is**,
and every path falls back gracefully so the same code runs on the Mac, a
1-GPU box, or the 8× A100 machine.

| stage | where it runs | why |
|---|---|---|
| zstd decode of `.dbn.zst` | CPU | inherently sequential I/O |
| featurization (`sigma_W`, trade signs) | CPU, **vectorized** (`fast_ops.py`) | single memory-bound pass after vectorization — a GPU round-trip would cost more than it saves; results are bit-identical to the old per-window Python loops |
| subsequence counting — **both** SEQ (`estimate_observed_subsequence_counts_torch`) and CLS (`estimate_subsequence_class_probabilities_torch`) | GPU if present (cuda→mps→cpu) | integer unfold + unique/bincount, exact on any device; replaces the last per-window Python loops |
| the day loop (`run`) | `featurize.workers` **parallel processes** | days are independent; results are folded in date order so output is byte-identical for any worker count. Workers count on CPU for histograms to avoid N CUDA contexts |
| Kraus training | GPU per model (`device: auto` = cuda→cpu) | complex matmuls; the model is small, so one GPU per *model*, not one model across GPUs |
| training sweep | **`train-all`**: one predictor per GPU, in parallel | 8 predictors × 8 A100s = the whole sweep in one wall-clock run |

On the compute box, set `featurize.workers: 0` (auto) in the config — the
45-day month then featurizes and counts with one process per core. The
default is 1 (serial) so behavior only changes where you opt in.

**`train-all`** — the multi-GPU sweep (`pipeline/parallel.py`):

```bash
python -m pipeline train-all --config run.yaml
```

- schedules one `python -m pipeline train` subprocess per predictor
  (`training.predictors`, default: all of `distributions.predictors`),
  pinning each to a GPU via `CUDA_VISIBLE_DEVICES`;
- concurrency = number of visible GPUs (8 on the A100 box → all 8 at once;
  1 GPU → a rolling queue; no GPU → `max_parallel` CPU processes, default 1);
- every child has its own run dir, log, config copy, and progress.json under
  the sweep's `outputs/runs/train-all-*/`; the sweep itself reports
  finished/total, so the control panel's *▶ Train all (multi-GPU)* button and
  `python -m pipeline status` both track it, and it survives SSH drops when
  launched from the panel (detached) or under `nohup`/`tmux`.

Verification for this layer (run them yourself):

- `tests/test_fast_ops.py` (seconds, any machine) — exact zero-tolerance
  equality of the vectorized ops vs the original Python loops, plus edge
  cases the real data may hit (NaN warmups, flat-mid zero runs).
- `tests/test_train_all_smoke.py` (~1–2 min, any machine) — real train-all
  run on synthetic distributions for 3 fake predictors; on the A100 box the
  report shows children landing on cuda:0/1/2, on the Mac they run on cpu.
- `tests/test_seq_counts_torch.py` (seconds, any machine) — zero-tolerance
  exact equality of the torch SEQ counter vs the untouched pure-Python
  original in read_databento_new.py, incl. subsampling rng reproduction,
  sort modes, and a pickle-bytes-identical case.
- `tests/test_parallel_run.py` (real data, ~minutes) — serial vs 2-worker
  parallel run must produce byte-identical SEQ/CLS outputs; this is the
  test that pins the date-order fold.
- `tests/verify_against_baseline.py` (Mac, ~45–60 min) — the byte-for-byte
  ground-truth check; it covers the featurization vectorization end to end
  since sigma_W feeds the distributions.
