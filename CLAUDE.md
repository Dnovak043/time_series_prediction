# time_series_prediction — project instructions

Market-microstructure research pipeline: NASDAQ ITCH order-book data
(NVDA/INTC, `data/NVDA_INTC/*.dbn[.zst]`) → LOB features → EWMA z-encoded
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
- Key semantics: raw files carry BOTH symbols interleaved —
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
  m = n_symbols^(1+len). LearningKraus_multivariate.py's library code is
  byte-identical to LearningKraus.py (no second vendored copy — only his
  driver differs); MOD_/WGHTS_ names follow his driver verbatim, incl.
  `WGHTS_` without `MOD_` and the `training.predictor_abbrev` tag
  (`L10_micro_vpin`).

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
  original counting, his naming), byte-for-byte. NOT YET RUN.
- `april_smoke.ipynb` — 1-day plumbing check of every April stage; expected
  counts derived from the config. Run before `april_run.ipynb`.

## Current state (2026-07-16)

- PRs #1–#6 all merged into `dev`; no open feature branches. Branch picture:
  `main` (locked, frozen baseline) + `dev` (everything). All equivalence
  suites user-run and PASSED before their merges.
- The April experiment is ready to run on the compute box:
  `april_smoke.ipynb` first, then `april_run.ipynb` (Linux params baked in:
  workers=0, ensemble on). Per symbol: 10 SEQ + 50 CLS-v2 + 20 v2 ENS_TD
  files + 3 trained Kraus models (2 result files + 4 charts each, sent
  per-model). Training defaults = original `LearningKraus.main()` values.
- Ensemble v2 (`ensemble_training_data_2.py`, PR #6) is integrated, is the
  default, and its byte-equivalence harness PASSED (user-run, 1 day).
- Multivariate Kraus (`LearningKraus_multivariate.py`, PR #7) is integrated:
  list predictors in distributions/training, his file naming, harness
  `tests/verify_multivariate_seq.py` — awaiting the user's harness run.
  Training on the 256-symbol alphabet (m·d² = 256·64² complex params) is a
  compute-box job, not a Mac job.
- `tests/baseline_manifest.json` not yet minted — first full
  `verify_against_baseline.py` PASS writes it; commit it, then use `--fast`.

## Conventions

- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`;
  PR body trailer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Draft PRs; the user merges after running the verification.
- Edits to CRLF legacy files: byte-level Python scripts (`open(p, 'rb')`,
  `\r\n` in replacements), never plain text edits — keep diffs reviewable.
- Outputs (`outputs/`, `SEQ_DISTR_*`, `CLS_DISTR_*`, `run.yaml`) are
  gitignored; run provenance lives in `outputs/runs/<id>/config.yaml`.
