# time_series_prediction — project instructions

Market-microstructure research pipeline: NASDAQ ITCH order-book data
(one directory per asset source under `data/` — `NVDA_INTC/`, `AAPL/`,
`IBM/`, same per-day file names, different contents) → LOB features → EWMA
z-encoded
symbol sequences → empirical subsequence/class distributions → Kraus-operator
(quantum-channel) models and ensemble training tables.

## Non-negotiable rules

1. **`data/` is immutable.** Only ever read from it. Never write, move, or
   modify anything under `data/`.
2. **The frozen baseline is the ground truth.** `main` is locked on GitHub at
   `0182a85`, pinned by the annotated tag **`baseline`**. Never push to main,
   never move the tag, never merge into main. All work goes through feature
   branches → PRs → **`dev`**.
3. **The user runs all verification tests — never Claude.** Claude builds
   test harnesses; the user executes them and judges results. Compile/AST/
   import checks by Claude are fine; anything that verifies *results* is the
   user's to run. Result equivalence means **byte-for-byte identical output
   files**, not "close".
4. **No invisible parameters.** Every knob that affects a run must be a
   `RunConfig` field (auto-exposed in YAML/UI/CLI via `pipeline/config.py`
   metadata). Run surfaces (notebooks/scripts) derive every value — including
   progress banners and test assertions — from the *loaded config*, never
   from parallel constants. Notebook constants may only feed the config
   generator, and the notebook prints the full generated YAML before running.
   **The inverse is equally binding: never ship a knob nothing reads.** If a
   parameter turns out to be ignored by the code it claims to drive, make it
   work (a disclosed `[vendoring fix N]` if the ignoring code is vendored) —
   do not delete it and do not leave it as decoration. `test_no_dead_knobs`
   and `test_training_knobs_reach_the_trainer` enforce this; PIPELINE_GUIDE
   §2b lists the deliberate non-knobs.
5. **Vendor colleague code verbatim.** External research code is committed
   byte-for-byte (CRLF preserved) with a `__main__` guard; any unavoidable
   fix is marked `[vendoring fix N]` inline and disclosed. Pipeline stages
   import the vendored math functions unchanged; only plumbing is replaced,
   and a user-run byte-equivalence harness proves it.

## Architecture

- `pipeline/` — config-driven layer: `config.py` (dataclass schema; a field
  added here appears in YAML/UI/CLI automatically), `features.py`
  (DayFeatureCache: one decode+featurize per (symbol, day), parquet cache),
  `distributions.py` (DistributionBuilder), `runner.py` (day-parallel driver,
  byte-identical for any worker count), `ensemble.py`, `models.py` (registry;
  `kraus`), `parallel.py` (`train-all`: one predictor per GPU), `ui.py`
  (ipywidgets panel, `pipeline_control.ipynb`), `cli.py`
  (`python -m pipeline run|ensemble|train|train-all|...`).
- `TrainingDistributions/` — legacy + vendored research code.
  `process_distributions.py` and `LearningKraus.py` are the original scripts
  (CRLF, `__main__`-guarded, optimized call sites swapped in);
  `subsequence_torch.py` (torch histograms, device-exact), `fast_ops.py`
  (vectorized sigma_W/trade-sign, bit-identical), `ensemble_reference.py`,
  `ensemble_reference_2.py` and `cls_reference.py` (colleague's programs,
  vendored).
- Key semantics: `data.asset_paths` maps each asset to its raw-data
  directory (several assets may share one, e.g. NVDA/INTC); the run reads
  the entry for `data.symbol`, unlisted symbols fall back to
  `data.data_path`, and the feature cache keys on the resolved directory.
  NVDA_INTC raw files carry BOTH symbols interleaved —
  `data.instrument_filter: true` is required for per-symbol runs (false =
  legacy mixed-stream = frozen-baseline behavior). CLS class-column order:
  legacy files are `[P(0), P(+1), P(−1)]` (list[-1] wrap); **v2**
  (`distributions.class_names` non-empty) is `[P(−1), P(0), P(+1)]` with
  colleague naming `CLS_DISTR_{sym}__{pred}-{predictor}_{month}_{cls}`. Old
  CLS files are being scratched; v2 is canonical for new outputs. Ensemble:
  `ensemble.reference` selects the vendored program — **v2** (default,
  `ensemble_reference_2.py`) allows list channels (joint multivariate
  encoding of predicted + listed features, `get_z_ts` math, alphabet
  n_symbols^(1+len)) and names files `..._ALL`; v1 is bivariate-only with
  `..._ALL_{n}` names and swept c4. Multivariate Kraus: a list entry in
  `distributions.predictors` emits `SEQ_DISTR_{sym}_multivariate_{pred}-`
  `{first}-{last}_{month}` (same joint encoding; SEQ only — no colleague
  multivariate CLS); a list `training.predictor` trains on it with
  m = n_symbols^(1+len). LearningKraus_multivariate.py's library code
  matches LearningKraus.py apart from our two disclosed deltas (the
  `on_epoch` callback and `[vendoring fix 1]`), so there is no second
  vendored copy — only his driver differs; MOD_/WGHTS_ names follow his
  driver verbatim, incl. `WGHTS_` without `MOD_` and the
  `training.predictor_abbrev` tag (`L10_micro_vpin`).
- **`[vendoring fix 1]` in `LearningKraus.py`** (CRLF, byte-level edit):
  `train()`'s DataLoader hardcoded `num_workers=0`, so its own
  `num_workers` argument was dead and the original `main()`'s
  `num_workers=8` never took effect. Now honoured, making
  `training.num_workers` a real knob. Results-neutral (shuffling is in the
  parent sampler; `SeqDataset.__getitem__` is a pure index lookup).
  `train_old` is legacy and left untouched.

## Environments

- **Intel Mac (this machine):** venv `.env/` (Python 3.12); torch capped at
  2.2.2 (broken numpy bridge — code uses `.tolist()` fallbacks); MPS works
  for integer ops. Mac-safe featurize workers: 4.
- **Linux compute box (8× A100):** same repo + `pip install -r
  requirements.txt`; use `featurize.workers: 0` (one per core); CUDA is
  auto-selected. Long runs: JupyterLab kernel survives browser disconnects,
  or `nohup scripts/run_april.py` under tmux.

## Verification suite (user-run; see PIPELINE_GUIDE.md §6-7)

- `tests/verify_against_baseline.py` — pristine `baseline`-tag code vs
  pipeline, byte-for-byte (Mac only). The trust anchor. Full mode (~1h)
  re-derives the ground truth and on PASS mints
  `tests/baseline_manifest.json` (golden sha256s + scope + environment;
  committed). `--fast` (~4 min) verifies against the manifest without
  re-running the baseline — use for routine regression; environment
  mismatch prints a loud warning (re-mint on that machine if so).
- `tests/parity_check.py`, `tests/test_pipeline_units.py`,
  `tests/test_parallel_run.py`, `tests/test_fast_ops.py`,
  `tests/test_seq_counts_torch.py`, `tests/test_train_all_smoke.py`.
- `tests/verify_ensemble_reference.py`, `tests/verify_ensemble_v2.py`,
  `tests/verify_cls_v2.py` — vendored colleague code run his way vs
  pipeline stages, byte-for-byte. All three PASSED (v2 ensemble:
  2026-07-16, 1-day scope).
- `tests/verify_multivariate_seq.py` — multivariate SEQ_DISTR (input to
  the multivariate Kraus model) + its training-load filtering vs the
  colleague's code composed his way (his get_z_ts, the pure-Python
  original counting, his naming), byte-for-byte. PASSED (user-run).
- `tests/test_pipeline_units.py` also carries the config audit and the
  stage-3 guards: `test_no_dead_knobs` (every config field is read by
  `pipeline/` or `scripts/`), `test_training_knobs_reach_the_trainer`,
  `test_stage3_schedule_comes_from_the_config`,
  `test_chart_capture_is_concurrency_safe`, `test_device_*`.
  **20 tests, Claude-runnable** (no market data) — run these after any
  change to `pipeline/`.
- `april_smoke.ipynb` — 1-day / 5-epoch check of every April stage,
  **including the stage-3 fan-out** (`plan_training` +
  `run_training_jobs`, not just a direct `train_model` call — the cell
  that only called the trainer let a real fan-out bug through) and an
  exact-filename check of the `WGHTS_*` deliverables. Run before
  `april_run.ipynb`. The pre-group-split version PASSED (user-run,
  2026-07-20); the reworked 2-group version (6 smoke trainings, incl.
  multivariate) has **not been user-run yet**.

## Current state (2026-07-27)

- **PRs #1–#10 all merged into `dev`; no open PRs, no feature branches.**
  Branch picture: `main` (locked, frozen baseline) + `dev` (everything).
  Every equivalence suite was user-run and PASSED before its merge.
- The April experiment covers **NVDA, INTC, and IBM** and reproduces the
  colleague's two current drivers (his 2026-02 `LearningKraus.py` +
  `LearningKraus_multivariate.py`, run by him for AAPL): run
  `april_smoke.ipynb` first, then `april_run.ipynb` (Linux params baked in:
  workers=0, ensemble on) or `scripts/run_april.py`. Per symbol: 11 SEQ
  (10 bivariate + 1 multivariate joint) + 50 CLS-v2 + 20 v2 ENS_TD files +
  **4 trained Kraus models** (2 result files + 4 charts each, sent
  per-model) — 12 models total. Two **training groups** per symbol, each
  its own generated config (`run_april.TRAIN_GROUPS`; one config holds one
  training block): group 0 = his bivariate driver — vpin, ofi_L10_norm_n,
  micro_price, **sgd**, batch 8*512, 3q, max_seq_len 6; group 1 = his
  multivariate driver — [ofi_L10_norm_n, micro_price, vpin], abbrev
  `L10_micro_vpin`, adam, batch 6*512, 6q, max_seq_len 4. Both: 3000
  epochs, lr 1e-3, nll_seq, learn_rho0, unseeded. Stage 2 runs once per
  symbol (the group configs share every distribution setting). NVDA/INTC
  read from `data/NVDA_INTC` (interleaved, filtered per symbol); IBM from
  `data/IBM`, resolved via `data.asset_paths`.
  **The reworked smoke (2 groups, exact-WGHTS-name checks) has NOT been
  user-run yet**; the 2026-07-20 smoke PASS predates the group split. The
  April run has not been executed at full scale.
- Ensemble v2 (PR #6) is the default; multivariate Kraus (PR #7) is
  integrated (256-symbol alphabet — a compute-box job, not a Mac job);
  per-asset data paths (PR #8) resolve each symbol's directory.
- **Training lives in `pipeline/models.py` (PR #10).** The April run calls
  the registered trainer — the same path as `python -m pipeline train` —
  and no longer imports from `tests/`. Charts (`training.plot_entries`),
  `training.seed`, `num_workers`, `eval_batch_size` and `plot_dpi` are all
  config fields the trainer reads. `tests/train_kraus_baseline.py` remains
  only as a standalone independent cross-check.
- **Model files are `MOD_*` / `WGHTS_*` (no `MOD_` infix in WGHTS), both
  modes** — verbatim from the colleague's current drivers, e.g.
  `WGHTS_{sym}_bivariate_log_mid-vpin_202504_3q.pt` and
  `WGHTS_{sym}_multivariate_log_mid-L10_micro_vpin_202504_6q.pt`. Retired
  conventions: `MODR_*`/`WGHTS_MODR_*` (pre-2026-07-20, the `title[8:]`
  off-by-one) and `WGHTS_MOD_*` (2026-07-20 consolidation, his older
  driver's form). **Filenames from current runs match neither earlier
  batch**, and a byte-comparison against `train_kraus_baseline.py` must
  account for it.
- **Stage-3 GPU fan-out (PR #10).** The 3-qubit model is tiny (m=16
  operators of d=8, ~2k params) and is kernel-launch-latency bound, so one
  training uses a few percent of an A100 — sequential training left 7 of 8
  GPUs idle. `plan_training()` + `run_training_jobs()` in
  `scripts/run_april.py` dispatch one training per device and yield
  results in **completion order**. Schedule comes from `training.gpus` /
  `training.max_parallel` (the same fields `train-all` uses), written into
  `configs/april_*.yaml` by `make_config` — `--gpus` / `--train-parallel`
  and the notebook's `GPUS` / `TRAIN_PARALLEL` feed the config, they are
  never read directly. Two schedulers still exist (`pipeline.parallel.
  train_all` is per-config and subprocess-based; the April fan-out is
  cross-symbol and pool-based); unifying them is an open follow-up that
  would cost the notebook's inline send-as-you-go charts unless
  `train_all` learns to stream results.
- **Ordering that matters in `train_kraus`:** the model and weights are
  persisted **before** charts are rendered, and a chart failure is caught
  and reported as `result["chart_error"]` rather than raised. Charting
  first meant a plotting error discarded a completed training.
- `pipeline/models.py::capture_plots_as_png` intercepts `plt.show` under a
  lock to save `plotDistributions`' chunked figures. It closes only the
  figures it opened and **does not touch the caller's matplotlib backend**
  — training from a notebook leaves that notebook's inline plotting
  intact.
- Config schema gained three metadata hooks (PIPELINE_GUIDE §2b):
  `free_form` (choices are presets, other values legal), `options_provider`
  (the panel's second chooser lists real values — `devices` →
  `pipeline.models.available_devices()`), and `validator` (a named checker
  applied generically by `validate()`). A `free_form` field that declares
  no validator is itself a validation error. `validate()` also enforces
  closed `choices` — nothing did before, so a typo in
  `optimizer`/`resampling` used to survive save/load.
- `data/AAPL` and `data/IBM` do not exist on the Mac; config validation is
  cheap and does not stat directories. **Three raw `.dbn.zst` files
  (~143 MB) are tracked at the repository root** from an early AAPL commit
  — they are not under `data/`, not gitignored, and have bloated `.git` to
  ~4.5 GB, which is enough to break some tooling (`/code-review ultra`
  refuses on repo size). Removing them from history needs a `dev` rewrite
  + force-push: **the user's call, not to be done unilaterally.**
- `tests/baseline_manifest.json` not yet minted — first full
  `verify_against_baseline.py` PASS writes it; commit it, then use
  `--fast`.

## Conventions

- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`;
  PR body trailer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Draft PRs; the user merges after running the verification.
- Edits to CRLF legacy files: byte-level Python scripts (`open(p, 'rb')`,
  `\r\n` in replacements), never plain text edits — keep diffs reviewable.
- Outputs (`outputs/`, `SEQ_DISTR_*`, `CLS_DISTR_*`, `run.yaml`) are
  gitignored; run provenance lives in `outputs/runs/<id>/config.yaml`.
