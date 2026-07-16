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
  (vectorized sigma_W/trade-sign, bit-identical), `ensemble_reference.py` and
  `cls_reference.py` (colleague's programs, vendored).
- Key semantics: raw files carry BOTH symbols interleaved —
  `data.instrument_filter: true` is required for per-symbol runs (false =
  legacy mixed-stream = frozen-baseline behavior). CLS class-column order:
  legacy files are `[P(0), P(+1), P(−1)]` (list[-1] wrap); **v2**
  (`distributions.class_names` non-empty) is `[P(−1), P(0), P(+1)]` with
  colleague naming `CLS_DISTR_{sym}__{pred}-{predictor}_{month}_{cls}`. Old
  CLS files are being scratched; v2 is canonical for new outputs.

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
  pipeline, byte-for-byte (Mac only; ~1h). The trust anchor.
- `tests/parity_check.py`, `tests/test_pipeline_units.py`,
  `tests/test_parallel_run.py`, `tests/test_fast_ops.py`,
  `tests/test_seq_counts_torch.py`, `tests/test_train_all_smoke.py`.
- `tests/verify_ensemble_reference.py`, `tests/verify_cls_v2.py` — vendored
  colleague code run his way vs pipeline stages, byte-for-byte. Both PASSED.
- `april_smoke.ipynb` — 1-day plumbing check of every April stage; expected
  counts derived from the config. Run before `april_run.ipynb`.

## Current state (2026-07-14)

- Branch chain: `dev` ← PR #4 `ensemble-integration` ← PR #5
  `distributions-v2` (both draft, verified, unmerged; merge #4 then #5,
  retarget #5 to dev after #4 lands).
- The April experiment (`april_run.ipynb` = notebook driver,
  `scripts/run_april.py` = CLI twin, on `distributions-v2`): both symbols
  fully separated (filter ON, own caches/outputs under
  `outputs/april/{SYMBOL}/`), per symbol: 10 SEQ + 50 CLS-v2 + 25 ENS_TD
  files + 3 trained Kraus models (boss's spec: 2 result files + 4 charts
  each, sent per-model as they finish). Training defaults = original
  `LearningKraus.main()` values (3000 epochs, 3q, batch 3072, lr 1e-3, adam,
  nll_seq, unseeded).
- Open question for colleague: ensemble `seq_lengths` stops at 5 while
  `max_seq_length = 6` sits unused in his file — intended?

## Conventions

- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`;
  PR body trailer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Draft PRs; the user merges after running the verification.
- Edits to CRLF legacy files: byte-level Python scripts (`open(p, 'rb')`,
  `\r\n` in replacements), never plain text edits — keep diffs reviewable.
- Outputs (`outputs/`, `SEQ_DISTR_*`, `CLS_DISTR_*`, `run.yaml`) are
  gitignored; run provenance lives in `outputs/runs/<id>/config.yaml`.
