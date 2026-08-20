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
| `asset_paths` | {} | **per-asset raw-data directory**, e.g. `{NVDA: data/NVDA_INTC, INTC: data/NVDA_INTC, AAPL: data/AAPL, IBM: data/IBM}`. Raw files share names across directories but hold different assets; several assets may map to one directory. The run reads from the entry for `symbol`; unlisted symbols (or an empty mapping) fall back to `data_path`. The feature cache keys on the resolved directory, so same-named files from different directories never collide. |
| `data_path` | data/NVDA_INTC | directory with raw `.dbn.zst` files; fallback when `symbol` has no `asset_paths` entry |
| `file_pattern` | xnas-itch-{date}.mbp-10.dbn.zst | raw file name per day |
| `dates` | [20250401, 20250402] | trading days (yyyymmdd) |
| `instrument_filter` | true | **true = filter events to `symbol` before featurizing.** The raw files carry NVDA+INTC interleaved; false reproduces the legacy (mixed-stream) behavior and the frozen baseline. Defaults true: his current `generate_timeseries` filters unconditionally. |
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
| `predictors` | his `features[1:]` | second variate; one SEQ/CLS output pair each. A **nested list entry** is one multivariate predictor: predicted + the listed features jointly encoded (same `get_z_ts` math as ensemble v2 channels, alphabet n_symbols^(1+len)) into one `SEQ_DISTR_{sym}_multivariate_{predicted}-{first}-{last}_{month}` file — the input to the multivariate Kraus model. SEQ only: the colleague defines no multivariate CLS output. Verified by user-run `tests/verify_multivariate_seq.py`. |
| `max_seq_length` | 6 | max n-gram length |
| `sequence_calculation` / `class_calculation` | true / true | which outputs to compute |
| `class_name` | c1 | forward-move class: `c{k}` return-sign, `ca{k}` fwd-vs-bwd sum |
| `class_names` | [ca4] | **v2 multi-class sweep** (colleague's new process_distributions, vendored as `cls_reference.py`): one CLS file per class, named `CLS_DISTR_{sym}_{predicted}-{predictor}_{month}_{cls}` (his earlier double underscore is retired), columns ordered by `class_values`. Empty = legacy single-class mode: old `[P(0),P(+1),P(−1)]` order and naming, matching the frozen baseline. Verified by user-run `tests/verify_cls_v2.py`. |
| `class_values` | [-1,0,1] | class column order in v2 mode (`[P(−1),P(0),P(+1)]`); ignored in legacy mode |
| `class_theta` | 0 (auto) | class threshold θ; 0 = built-in default per class |
| `num_classes` | 3 | down / flat / up |
| `sample_size` / `sample_after_length` / `random_state` | 1.0 / 30 / 42 | optional support subsampling for long n-grams |
| `output_dir` | . | where SQ_PRB_*/CLS_DISTR_* land (repo root = legacy behavior) |

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

### training — SQ_PRB_* (bivariate) / SEQ_DISTR_* (multivariate) → trained model
| field | default | meaning |
|---|---|---|
| `model` | kraus | trainer from the registry (future models plug in here) |
| `predictor` | tvi_n | which predictor's distribution to train on. A **list** (e.g. `[ofi_L10_norm_n, micro_price, vpin]`) trains on that multivariate SEQ_DISTR file with alphabet m = n_symbols^(1+len) — the colleague's LearningKraus_multivariate driver (his settings: n_qubits 6, max_seq_len 4, otherwise the usual training defaults). Same `KrausInstrument` math, byte-identical library code. |
| `predictor_abbrev` | "" | short tag replacing the predictor part of MOD_/WGHTS_ names in multivariate mode (his hand-written `L10_micro_vpin`); empty = `{first}-{last}` from the predictor list. Multivariate naming follows his driver verbatim, including `WGHTS_` without the `MOD_` infix: `MOD_{sym}_multivariate_{predicted}-{tag}_{month}_{q}q` / `WGHTS_{sym}_multivariate_..._{q}q.pt` |
| `seq_distr_file` | (derived) | explicit pickle path override |
| `n_qubits` | 3 | Hilbert dim d = 2^n_qubits |
| `epochs` / `batch_size` / `lr` / `optimizer` | 3000 / 3072 / 1e-3 / adam | optimization |
| `loss_kind` | nll_seq | `nll_seq` or `mse_prob` |
| `length_mixture` | uniform | length reweighting (`uniform`/`geometric`/`none`) |
| `learn_rho0` | true | learn the initial state vs fixed \|0⟩⟨0\| |
| `max_seq_len` / `min_seq_prob` | 6 / 0 | training-set filters |
| `num_workers` | 0 | DataLoader worker processes. Speed only, never results (shuffling is done by the sampler in the parent; `SeqDataset.__getitem__` is a pure index lookup). **`LearningKraus.train()` ignored this until `[vendoring fix 1]`**, so the original `main()`'s `num_workers=8` never took effect — 8 now reproduces its stated intent. Keep low when trainings are fanned across GPUs: each is already a subprocess and workers nest beneath it. |
| `eval_batch_size` | 2048 | batching of the post-training `predict_probs` pass (original used 2×1024). Memory/speed only; probabilities are identical for any value |
| `plot_entries` | 200 | sequences charted by `plotDistributions`. It draws in chunks of 62, so 200 → the characteristic **4 PNGs per model**. 0 skips charting. Fewer than 200 surviving sequences yields fewer charts |
| — | — | *Capture mechanism:* `plotDistributions` calls `plt.show()` per chunk, so `pipeline.models.capture_plots_as_png` intercepts `show` to save each figure instead. pyplot is a process-global state machine, so the interception is **lock-guarded** — concurrent in-process trainings each get their own complete, correctly-named set. It closes only the figures it opened and does **not** change the caller's backend, so training from a notebook leaves that notebook's inline plotting intact. |
| `plot_dpi` | 120 | resolution of the saved chart PNGs — a run parameter, since it changes the delivered image files |
| `seed` | -1 | torch seed. -1 = unseeded (original `main()` behavior). **Declared but never applied before the training consolidation** — it now works |
| `device` | auto | auto = cuda→cpu; `mps` opt-in (complex-op support is limited). **Free-form**: besides the four presets it accepts an indexed device (`cuda:3`, `mps:0`), which is how the April fan-out pins one training per GPU and what lands in each per-model `config.yaml`. The control panel renders it as a preset dropdown **plus a specification textbox** (see §2b); `validate()` rejects anything that is neither a preset nor `cuda|mps:<int>`. |
| `continue_from` | — | WGHTS_*.pt to resume |
| `model_dir` | . | where MOD_*/WGHTS_* are written |
| `predictors` | [] (= all) | which predictors `train-all` sweeps over |
| `gpus` | auto | GPUs to schedule trainings on, for **both** `train-all` and the April stage-3 fan-out: `auto` = every CUDA device torch sees (respects an externally set `CUDA_VISIBLE_DEVICES`), `0,2,5` = those ids, `none` = force CPU |
| `max_parallel` | 0 (auto) | concurrent trainings, for **both** `train-all` and the April fan-out; auto = one per GPU, else 1. Each 3-qubit model uses only a few percent of an A100, so values above the GPU count are reasonable |

Stage-3 scheduling is config-driven: `scripts/run_april.py --gpus/--train-parallel`
and the notebook's `GPUS`/`TRAIN_PARALLEL` are written *into* the generated
`configs/april_*.yaml` by `make_config` (exactly as `--epochs`, `--n-qubits` and
`--seed` already are), and `training_schedule()` reads them back from there. The
printed YAML is therefore the authority for how stage 3 runs, and the schedule is
reproducible from the config alone.

**Model file naming (bivariate):** `MOD_{sym}_bivariate_{predicted}-{predictor}_{month}_{q}q`
and `WGHTS_{sym}_bivariate_..._{q}q.pt` — verbatim from the colleague's current
(2026-02) `LearningKraus.py` driver, which builds both names from the run
components and puts **no `MOD_` infix** in the `WGHTS_` name (matching his
multivariate driver). Two older conventions are retired: pre-consolidation runs
(the April harness) emitted `MODR_*` / `WGHTS_MODR_*` — a stray `R` from slicing
only 8 of the 10 characters in `SEQ_DISTR_` — and the 2026-07-20 consolidation
briefly emitted `WGHTS_MOD_*` (his older driver's form). Filenames from current
runs match neither batch.

**The SQ_PRB scheme (his 2026-08 `process_distributions`).** This branch cuts
the distribution stage over to his newer scheme; `TrainingDistributions/
process_distributions_v2.py` is his file vendored verbatim (two disclosed
fixes: a guard on the module-level driver, and the missing local `plotting`
import commented out) and serves as the byte-oracle. Sequence outputs are
`SQ_PRB_{sym}_{predicted}-{predictor}_{month}` and class outputs
`CLS_DISTR_{sym}_{predicted}-{predictor}_{month}_{cls}` (his earlier double
underscore is gone). Fields this adds:

| field | default | meaning |
|---|---|---|
| `distributions.output_mode` | monthly | which of his two branches to reproduce. `monthly` = his training branch: aggregate every date into one file per predictor, month-tagged, payloads `[distrs, samples]` / 4-field CLS rows. `daily` = his validation branch: one file per (predictor, day), full-date tagged, no aggregation, payloads `[sequences, seq_probs]` / `[[subsequence, class_probs], ...]`. The payload difference is his, not ours |
| `distributions.class_tag_in_name` | true | put the class in CLS names as he does — note he orders the fields differently per branch: monthly `..._{month}_{cls}`, daily `..._{cls}_{date}`. False drops the tag, unambiguous only while one class is swept |
| `distributions.save_seq_prob_weight` | false | also emit his per-day `SQ_PRB_WT_{sym}_{date}` = `[sequences, seq_probs, global_weights]`. Built from the `(predicted, predicted)` encoding and carrying no predictor tag — both his quirks, preserved |
| `training.weights_scheme` | ensemble | encoder filename convention: `ensemble` = `WGHTS_{sym}_{variate}_{predicted}-{tag}_{month}_{q}q.pt` (what LearningEnsemble loads), `qmod` = `WGHTS_QMOD_{sym}_{predicted}-{tag}_{q}q_{month}.pt` (his newest trainer). His two files disagree; both stages here read this one field via `RunConfig.model_names()`, so they cannot drift |
| `ensemble_model.exclude_last_channel` | true | ensemble only the first n−1 encoders, as his driver does (his last channel is the multivariate one he excludes). With an all-bivariate channel list this would silently drop a real feature — set false there |

`scripts/run_sqprb.py` drives the whole chain for this scheme:

```bash
python scripts/run_sqprb.py --stages all --workers 0     # the four stages
python scripts/run_sqprb.py --stages train               # any subset, resumable
```

Stages run stage-major, so each sees every symbol: the CPU stages
(distributions, ensemble-tables) process securities concurrently
(`--symbol-parallel`, since the day loop alone is capped at the trading-day
count), and the GPU stages schedule every (symbol × predictor) job across
every visible device through the same `plan_training`/`run_training_jobs`
fan-out stage 3 of the April run uses.

**`ensemble_model` (stage 4, LearningEnsemble.py).** Frozen pre-trained Kraus
encoders + a trained QuantumDecoder predicting the class distribution
(`pipeline/ensemble_model.py`; CLI `python -m pipeline ensemble-model`; the
vendored math is `LearningEnsemble.py` at the repo root, `[vendoring fix 1]`
= guarded module-level `sys.exit()`, `[vendoring fix 2]` = the driver's
hardcoded `gpu_id=1`, superseded by `ensemble_model.device`). Inputs: the v2
`ENS_TD_*` tables (`ensemble.output_dir`) and the four `WGHTS_*` encoders
(`training.model_dir`). Field defaults are his driver verbatim: classes
`[c2, ca4]` (one trained model per class — his single `clsName` re-run),
`seq_lens [1,2,3,4]`, channels = 3 bivariate + the joint multivariate
(only the first n−1 encoders enter the trained ensemble — his driver
excludes the multivariate channel), 200 epochs, batch 6·512, lr 2e-4,
`ce` loss, decoder-only (encoders frozen). Output:
`{model_dir}/{cls}/ENS_MD_{sym}_{month}_` — his file name has no class tag,
so the per-class directory is the disambiguator. `validate()` checks that
every requested class/length has a corresponding ENS_TD source and that the
three channel lists agree in length.

### 2b. The no-invisible-parameters invariant

Every value that changes a run must be a `RunConfig` field. The inverse also
holds and is easier to violate: **a field that nothing reads is worse than no
field**, because it looks like a working knob. Two shipped before the audit —
`training.num_workers` (which `LearningKraus.train()` ignored) and the
notebook's `PREDICTED` (shadowed by `run_april.py`'s module constant).

`tests/test_pipeline_units.py` now enforces both directions:

- `test_no_dead_knobs` — every field of every stage dataclass is read somewhere
  in `pipeline/` or `scripts/` (67 fields at time of writing).
- `test_training_knobs_reach_the_trainer` — the consolidated trainer reads
  `seed`, `num_workers`, `eval_batch_size`, `plot_entries`, `plot_dpi`, and
  `LearningKraus.train()`'s DataLoader honours `num_workers` rather than
  hardcoding it.

**Closed vs free-form choices.** A field declared with `choices=` is closed:
`validate()` rejects any value outside the set, and the panel renders a plain
dropdown. A field declared `choices=[...], free_form=True` treats those as
*presets* — other values are legal and get their own validation rule, and the
panel renders `ChoiceOrCustom`: the preset dropdown plus a second chooser that
activates on the `custom…` entry.

Adding `options_provider="<name>"` makes that second chooser a **dropdown of
real values** rather than a text box; the provider is a callable registered in
`pipeline/ui.py`'s `_OPTION_PROVIDERS`, which keeps `pipeline/config.py` free
of torch and UI imports.

`training.device` is the one such field today: presets `auto/cuda/cpu/mps`,
plus `options_provider="devices"` → `pipeline.models.available_devices()`,
which lists the accelerators actually present (`cuda:0…N` via the same
`visible_gpus` the fan-out and `train-all` use, or `mps:0`, always `cpu`).
This exists because the GPU fan-out writes `cuda:3` into every per-model
`config.yaml`; before it, such a value passed `validate()` silently and then
raised `TraitError` at widget construction, making exactly the configs a real
run produces unopenable.

Two consequences worth knowing:

- **Only real devices are offerable**, so the panel cannot produce an invalid
  device at all — an improvement over validating free text after the fact.
- **Configs are portable**, so a value the loaded config already names is
  always included even when absent locally. Opening the compute box's
  `cuda:3` config on the Mac shows `cuda:3` alongside the local `mps:0`, and
  saving round-trips it unchanged. Availability is deliberately *not* a
  `validate()` check for the same reason.

`ControlPanel.save()` also validates on the way out: the file is still
written (edits are never lost) but any problems are reported in the status
bar rather than passing unnoticed into a run.

**Deliberate non-knobs** (fixed because varying them would break an output
contract, not because they were overlooked):

| value | where | why fixed |
|---|---|---|
| `sort="lexicographic"`, `include_prob=True` | `distributions.sequence_counts` | the SEQ_DISTR file format and ordering the colleague's loaders expect |
| chunk size 62 | `LearningKraus.plotDistributions` | vendored drawing code; it is what makes `plot_entries=200` yield 4 charts |
| `weight_decay=1e-4` | `LearningKraus.make_optimizer` call inside `train()` | vendored optimizer construction; not exposed by the original either |
| progress-update cadence, cache-key hash length | `runner.py`, `features.py` | internal bookkeeping, no effect on outputs |

**One documented asymmetry:** `distributions.class_theta` applies to the
distribution stage's CLS outputs but **not** to the ensemble stage. That is
correct, not a gap — the colleague's v2 `add_class_label`
(`ensemble_reference_2.py`) has no `theta` parameter at all, and the
byte-equivalence harness `tests/verify_ensemble_v2.py` passed against his code
on that basis.

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
  keying, import-safety of the guarded legacy scripts, plus the
  no-dead-knobs audit of §2b (11 tests, seconds).
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
