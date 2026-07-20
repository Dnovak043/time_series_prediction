# time_series_prediction

Market-microstructure research: can short symbolic patterns in NASDAQ
order-book activity predict near-term price moves? This repository turns raw
tick data into empirical pattern statistics and trains quantum-channel-inspired
("Kraus operator") sequence models and multi-channel ensembles on them.

This README is the entry point for reading the codebase from zero. Companion
documents:

- **`PIPELINE_GUIDE.md`** — every configurable knob, the three ways to drive
  the pipeline (notebook panel / CLI / Python), GPU & parallel execution.
- **`ARCHITECTURE_AND_PERFORMANCE.md`** — the original research code mapped,
  its bottlenecks, and the optimization roadmap (written *before* the
  refactor; kept as the historical rationale).
- **`CLAUDE.md`** — the working agreement for AI-assisted development
  (trust model, non-negotiable rules, current branch state).

---

## 1. The idea, end to end

```
raw order-book events (.dbn.zst, ~5.9M events/day, NVDA+INTC interleaved)
      │  decode + featurize: mid-price, log-mid returns, order-flow imbalance
      │  (OFI at 1/3/10 levels), trade-volume imbalance, VPIN, micro-price,
      │  rolling volatility sigma_W … resampled every 100 events
      ▼
feature bars (~50k rows/day, one row per 100 events)
      │  z-encode: EWMA z-score of each feature → 4 quantile bins
      │  bivariate combine: symbol = predicted_bin + 4 × predictor_bin
      ▼
one 16-letter symbol stream per (predicted, predictor) pair per day
      │  count subsequences (length 1–6) and the class outcome
      │  (down/flat/up) observed at the end of each occurrence
      ▼
empirical distributions, aggregated over a month
  ├── SEQ_DISTR_*  P(pattern)                      → trains Kraus models
  ├── CLS_DISTR_*  P(class | pattern)              → analysis / class prediction
  └── ENS_TD_*     joint + per-channel tables      → trains ensembles
      ▼
models
  ├── KrausInstrument (LearningKraus.py): 16 complex d×d operators, d=2^3;
  │   P(s₁…s_T) = Tr(K_{s_T}···K_{s_1} ρ₀ K†…) fitted to SEQ_DISTR by NLL
  │   — or, multivariate: 256 operators, d=2^6, fitted to the joint-channel
  │   SEQ_DISTR (LearningKraus_multivariate driver; identical library code)
  └── ensemble of channel models (colleague's line of work; ENS_TD_* is
      its training data — the models themselves are not in this repo yet)
```

Two research programs share this machinery:

1. **Per-pair Kraus models** — one model per (log_mid, predictor) pair,
   fitted to reproduce the pattern distribution. The multivariate variant
   promotes the ensemble's joint channel to a full Kraus model: a list
   entry in `distributions.predictors` (e.g.
   `[ofi_L10_norm_n, micro_price, vpin]`) emits a
   `SEQ_DISTR_{sym}_multivariate_*` file over the 4⁴ = 256-symbol joint
   alphabet, and a list `training.predictor` trains on it (his settings:
   n_qubits 6, max_seq_len 4).
2. **Channel ensemble** — instead of one model over many features jointly
   (alphabet would explode as 4^n), build small-channel models and combine
   them; `ENS_TD_*` records both each channel's marginal statistics and the
   exact joint co-occurrences the combination must explain. The v2 spec
   (current default) uses 3 bivariate channels plus one deliberate 4-feature
   joint channel (alphabet 4⁴ = 256) as a reference point.

**`log_mid` is the `predicted` variate in every single artifact.** Every
encoding pairs it with one predictor feature (the v2 ensemble's joint
channel pairs it with several at once); it appears in every filename.

## 2. Repository map

| path | what it is |
|---|---|
| `pipeline/` | The maintained, config-driven layer. One dataclass per stage in `config.py` — a field added there automatically appears in YAML, control panel, and CLI. `features.py` = per-day featurize cache; `distributions.py` = per-day encode+count; `runner.py` = day-parallel batch driver; `ensemble.py` = ENS_TD stage; `models.py` = model registry (`kraus`); `parallel.py` = `train-all` (one predictor per GPU); `ui.py` = ipywidgets control panel; `cli.py` = `python -m pipeline …`. |
| `TrainingDistributions/process_distributions.py` | The original research script (featurization, z-encoding, distribution estimators, guarded legacy driver). The pipeline imports its functions; its hot spots were swapped for verified equivalents (see §6). CRLF line endings — see §7. |
| `TrainingDistributions/read_databento_new.py` | Raw-file decoding (`dbn_to_df`) + older utilities. Also holds the pure-Python `estimate_observed_subsequence_counts` kept as the reference for its torch replacement. |
| `TrainingDistributions/subsequence_torch.py` | Torch histogram counting (SEQ + CLS), device-agnostic (cuda→mps→cpu), integer-exact on any device. Supports both class-column conventions (§4). |
| `TrainingDistributions/fast_ops.py` | Vectorized `sigma_W` rolling RMS and trade-sign carry — bit-identical replacements for per-row Python loops. |
| `TrainingDistributions/ensemble_reference.py` | Colleague's `ensemble_training_data.py` (v1), vendored **verbatim** (import-guarded). Selected by `ensemble.reference: v1`. |
| `TrainingDistributions/ensemble_reference_2.py` | Colleague's `ensemble_training_data_2.py` (v2, the default), vendored **verbatim** (import-guarded). Adds `get_z_ts` — joint multivariate encoding — and drops the fixed-alphabet validation. The ensemble stage imports its counting math unchanged. |
| `TrainingDistributions/cls_reference.py` | Colleague's rewritten `process_distributions.py`, vendored with four `[vendoring fix N]`-marked corrections. Source of the v2 multi-class CLS semantics. |
| `TrainingDistributions/integrate_day_distributions.py` | Merges per-day counts into monthly aggregates (pure dict math). |
| `TrainingDistributions/plot_distributions.py` | Chart helpers; `plotDistributions` draws in chunks of 62 → the characteristic 4 charts per trained model. |
| `LearningKraus.py` | The Kraus model + training loop (guarded original script). `pipeline/models.py` and the April harness call `train()` with explicit parameters. The colleague's `LearningKraus_multivariate.py` has byte-identical library code (only its driver differs: 256-symbol joint alphabet, n_qubits 6), so the multivariate trainer imports this same module — no second vendored copy. |
| `april_run.ipynb` / `scripts/run_april.py` | The current flagship experiment (notebook and identical CLI): NVDA, INTC, and IBM, all April, full spec — see §5. |
| `scripts/export_feb_features.py` | One-off deliverable for the colleague: raw feature columns (no encoding/distributions) for 3 Feb days × AAPL/NVDA × {100 events, 1 second} × feature set → one file each, 12 total. Feature-name mapping documented in its docstring. |
| `april_smoke.ipynb` | 1-day plumbing check of every April stage; run before the real thing. |
| `pipeline_control.ipynb` | The interactive control panel (all knobs, launch/monitor, results plots). |
| `compare_main_vs_dev.ipynb` | Visual/hash comparison of frozen-baseline outputs vs current outputs. |
| `tests/` | Verification harnesses — see §6. `train_kraus_baseline.py` is a standalone verbatim-`main()` training harness, kept as an independent cross-check of `pipeline/models.py`; production runs use the pipeline trainer, not this file. |
| `configs/` | Generated experiment configs (`april_nvda.yaml`, `april_intc.yaml`) + `default.yaml`. |
| `data/` | Raw Databento files, one directory per asset source (`NVDA_INTC/`, `AAPL/`, `IBM/`, …), **gitignored, immutable — never write here**. |
| `outputs/` | Everything produced: feature caches, distributions, models, run logs (`outputs/runs/<id>/` has `config.yaml` + `progress.json` + `run.log` per run). Gitignored. |

## 3. The data

- One file per trading day: `xnas-itch-YYYYMMDD.mbp-10.dbn[.zst]` — Databento
  MBP-10 order-book snapshots + trades. Plain `.dbn` and zstd-compressed
  read identically; discovery auto-detects.
- **Several source directories can coexist under `data/`** — files share the
  same per-day names but hold different assets (`data/NVDA_INTC/` = NVDA+INTC
  interleaved, `data/AAPL/` = AAPL, `data/IBM/` = IBM). `data.asset_paths`
  (config) maps each asset to its directory — several assets may share one —
  and the run reads from the entry for `data.symbol`, falling back to
  `data.data_path` for unlisted symbols. The feature cache keys on the
  resolved directory, so same-named files never collide.
- **Each `NVDA_INTC` file contains BOTH symbols interleaved** (20250401: ~5.2M NVDA +
  ~0.64M INTC events). The original code never filtered — its outputs are
  mixed-stream statistics. `data.instrument_filter: true` (config) enables
  true per-symbol runs; `false` reproduces the legacy/frozen-baseline
  behavior exactly.
- April 2025 = 21 trading days (no 2025-04-18, Good Friday).

## 4. Artifact formats (pickle schemas)

**`SEQ_DISTR_{sym}_bivariate_{predicted}-{predictor}_{yyyymm}`**
`[distrs, samples]` where `distrs = [[sequence, probability], …]`
(sequence = list of ints 0–15, lengths 1–6, observed patterns only) and
`samples = [sequence, …]` in the same order. Training input for Kraus models.

**`SEQ_DISTR_{sym}_multivariate_{predicted}-{first}-{last}_{yyyymm}`**
(a list entry in `distributions.predictors`; name carries the first and
last listed feature, per the colleague's driver): same `[distrs, samples]`
schema over the joint alphabet 0–4^(1+len)−1 (256 for his 3-predictor
spec). Training input for the multivariate Kraus model.

**`CLS_DISTR_*` — two conventions, know which you're reading:**
- *Legacy* (`…_bivariate_{predicted}-{predictor}_{yyyymm}`, produced when
  `distributions.class_names` is empty): flat list of rows
  `[sequence, class_counts, class_probs, total]` with columns in the
  accidental order **[P(0), P(+1), P(−1)]** (Python `list[-1]` wrap).
  Matches the frozen baseline; being retired.
- *V2* (`CLS_DISTR_{sym}__{predicted}-{predictor}_{yyyymm}_{cls}` — note the
  double underscore, and one file per class definition): same row shape,
  columns in **[P(−1), P(0), P(+1)]** = down/flat/up, per
  `distributions.class_values`. Canonical going forward.

**Class definitions** (`add_class_label`): `c{k}` = sign of
`log_mid_return_fwd_k` vs θ (c1/c2: θ=4.15e-5, c4: 7.5e-5); `ca{k}` = forward
vs backward k-step sums vs θ·{1,1.5,2} (θ=7e-6). Labels ∈ {−1, 0, +1}.

**`ENS_TD_{sym}_{yyyymm}_SL_{k}_CL_{cls}_{predicted}_ALL`** (v2, current) /
**`…_ALL_{n}`** (v1)
`[joint_data, component_data]`. `joint_data` (dict): `X` int16
`[N, n_channels, k]` (the N unique joint patterns across all channels),
`counts`, `class_counts` `[N, 3]`, `seq_probs`, `target_distributions`
`[N, 3]` (columns per `class_values`, optional Dirichlet smoothing),
`daily_sample_counts`, plus `keys/index/class_values/sequence_length/
n_channels`. `component_data`: list of per-channel dicts of the same shape
with `X` `[N_i, k]` and `channel_index`. Channels are timestamp-aligned by
inner join; windows never cross day boundaries. A channel is one bivariate
encoding (symbols 0–15) or, in v2, optionally a joint encoding of predicted
+ several predictors (a list entry in `ensemble.predictors`; symbols
0–4^(1+len)−1) — channel alphabets may differ within one file.

**Models**: `MOD…_{n}q` = pickle `[model, sequences, emp_probs]` (full
nn.Module + training set); `WGHTS_MOD…_{n}q.pt` = `torch.save` of
`{"model_state", "meta"}` (meta: m, n_qubits, d, learn_rho0, symbol,
predictor, epochs, seed). Multivariate models follow the colleague's
driver naming instead:
`MOD_{sym}_multivariate_{predicted}-{tag}_{yyyymm}_{n}q` /
`WGHTS_{sym}_multivariate_…_{n}q.pt` (no `MOD_` infix — his convention),
where `tag` is `training.predictor_abbrev` (his `L10_micro_vpin`) or
`{first}-{last}`. Per trained model you also get 4 PNG charts
(`{sym}_{yyyymm}_{predicted}-{predictor}_{n}q_1..4.png`) — target-vs-model
bars for pattern windows [0-62], [62-124], [124-186], remainder.

**Encoding detail worth knowing**: combined symbol =
`predicted_bin + 4 × predictor_bin` (predicted is the low digit), bins from
expanding-quantile EWMA z-scores (α=0.05, backfilled warm-up). The v2 joint
channel generalizes this base-4 positional scheme: variate i (predicted
first) contributes `bin_i × 4^i`.

## 5. Running things

```bash
python3 -m venv .env && source .env/bin/activate
pip install -r requirements.txt          # Linux: plain torch = CUDA build
python -m ipykernel install --user --name tsp
```

- **Quick sanity**: `python tests/test_pipeline_units.py` (seconds), then
  `april_smoke.ipynb` (1-day pass through every stage, ~15 min).
- **The April experiment**: `april_run.ipynb` (Run All) or
  `nohup python scripts/run_april.py --workers 0 --with-ensemble &`.
  It prints the complete generated YAML for all three symbols before running
  (that printout is the authoritative parameter record), then per symbol:
  distributions (10 predictors × 5 classes), ensemble tables (v2: 3
  bivariate + 1 joint channel, 5 lengths × 4 classes = 20 files), and the
  3 Kraus trainings with per-model READY-TO-SEND bundles. NVDA and INTC
  read from `data/NVDA_INTC` (interleaved, filtered per symbol); IBM reads
  from its own `data/IBM` — resolved per symbol via `data.asset_paths`.
- **Interactive**: `pipeline_control.ipynb` — every knob, detached
  launches that survive SSH drops, live monitoring, results plots.
- **Anything ad hoc**: `python -m pipeline init-config run.yaml`, edit, then
  `python -m pipeline run|ensemble|train|train-all --config run.yaml`.

Environment notes: Intel Mac dev box pins torch 2.2.2 (broken torch↔numpy
bridge → code uses `.tolist()` fallbacks; harmless warning noise on import).
Linux box: `featurize.workers: 0` = one day-worker per core; training and
`train-all` pick up CUDA automatically.

## 6. Correctness: how we know the refactor changed nothing

The project's standard is **byte-for-byte identical output files**, and the
authority chain is:

1. **Frozen ground truth**: branch `main` is locked on GitHub at `0182a85`,
   pinned by the annotated tag `baseline`. All development happens on
   feature branches merged into `dev`.
2. **`tests/verify_against_baseline.py`**: checks the tag out into a
   temporary worktree, runs the *untouched original script* at its own
   hardcoded scope, runs the current pipeline at the same scope (parsed out
   of the baseline source by AST, nothing hand-copied), and byte-compares
   every output. On PASS it mints `tests/baseline_manifest.json` (golden
   sha256s + environment); `--fast` verifies against the manifest in ~4 min.
3. **Vendored-code equivalence** (`tests/verify_ensemble_reference.py`,
   `tests/verify_ensemble_v2.py`, `tests/verify_cls_v2.py`,
   `tests/verify_multivariate_seq.py`): the colleague's programs, run their
   way from the vendored copies, vs the pipeline stages — byte-compared.
4. **Mechanical invariants**: `tests/test_fast_ops.py` and
   `tests/test_seq_counts_torch.py` (zero-tolerance exact equality of the
   optimized ops vs the original Python, incl. rng reproduction),
   `tests/test_parallel_run.py` (serial vs parallel byte-identity),
   `tests/parity_check.py` (fast legacy-vs-runner regression),
   `tests/test_train_all_smoke.py` (GPU scheduling).

All of these are **run by the human, not by the AI assistant** — that
division is deliberate (see `CLAUDE.md`). Every optimization below passed
this gauntlet before merging:

| change | effect | proof |
|---|---|---|
| day-featurize cache + inverted loop | 8× redundant decodes → 1 | baseline byte-identity |
| torch CLS histogram (unfold+bincount) | removes 16⁶ Python enumeration | exact-equality tests + baseline |
| torch SEQ counting | removes last per-window Python loop | pickle-byte equality tests |
| vectorized `sigma_W` / trade signs | per-event Python lambda → one pass | bit-identical by construction + tests |
| day-parallel runner | ~core-count× throughput | serial-vs-parallel byte-identity |
| ensemble stage | colleague's ~6,300 decodes/month → 21 | byte-identity vs his code |
| ensemble v2 (joint multivariate channel) | his new experiment, config-driven | byte-identity vs his v2 code |
| multivariate SEQ_DISTR + Kraus (256-symbol joint alphabet) | his LearningKraus_multivariate input, config-driven | byte-identity of the SEQ file + load filtering vs his code |

Representative timings (this Mac): original driver 3,305s vs pipeline 254s
for 2 days × 8 predictors, identical bytes; colleague's ensemble scope
2,706s vs 174s, identical bytes.

## 7. Conventions and gotchas

- **Class-column order** is the one live semantic trap — see §4. Legacy and
  v2 CLS files are silently incompatible permutations of each other.
- **Legacy files are CRLF** (Windows-authored). Edit them with byte-level
  scripts preserving `\r\n`, or diffs become unreviewable.
- **No invisible parameters**: every run knob is a `RunConfig` field; run
  surfaces derive all values (including progress banners and test
  assertions) from the loaded config; notebooks print the full YAML before
  running.
- **Vendored colleague code is byte-verbatim** except guard wrappers and
  `[vendoring fix N]`-marked corrections; pipeline stages import its math
  functions rather than reimplementing them.
- Naming quirks are preserved deliberately where outputs must match the
  original programs (`MOD` + `title[8:]` → `MODR_…`; the v2 CLS double
  underscore).
- The original scripts ran their whole pipeline at import; they are now
  `__main__`-guarded but otherwise behave identically when run directly.
