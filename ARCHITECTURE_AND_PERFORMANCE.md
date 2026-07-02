# time_series_prediction — Architecture & Performance Map

Prepared for an agent picking up work in this repo. This is a read-only research
document — **no data files were touched or modified while producing it**, and
none of the recommendations below should be applied to `data/` (raw market data
is treated as immutable ground truth throughout this project).

## 1. What this project does

It's a market-microstructure research pipeline built on NASDAQ ITCH order-book
data for NVDA/INTC (Databento MBP-10, `data/NVDA_INTC/xnas-itch-*.mbp-10.dbn.zst`,
45 trading days ≈ 10 GB compressed). The goal is to:

1. Reconstruct limit-order-book (LOB) features from raw tick events (mid-price,
   spread, order-flow imbalance at multiple depths, VPIN, micro-price, ...).
2. Symbolically encode pairs of features (a "predicted" series and a
   "predictor" series) into a small discrete alphabet via EWMA z-scoring +
   quantile binning.
3. Estimate empirical probability distributions over short symbol
   *subsequences* (n-grams, length 1–6) and over the forward price-move class
   conditional on each subsequence.
4. Aggregate those distributions across a month of trading days and persist
   them (`SEQ_DISTR_*`, `CLS_DISTR_*` pickle files in the repo root).
5. Train a small "Kraus operator" (quantum-channel-inspired) PyTorch model on
   the sequence→probability data (`LearningKraus.py`) to predict next-symbol
   distributions.

## 2. Module map

| File | Role |
|---|---|
| `TrainingDistributions/read_databento_new.py` | Two eras of code in one file: (a) **legacy** `.mbp-1.csv.zst` toolkit — `zstd_to_df`, `getVolatility1-4`, `xCorrelation`, MI/transfer-entropy diagnostics, `X_pred_Y` — not on the active pipeline path; (b) shared helpers still in use: `dbn_to_df`, z-encoding primitives, `estimate_subsequence_counts` / `estimate_observed_subsequence_counts`. |
| `TrainingDistributions/process_distributions.py` | **The active pipeline.** Feature engineering (`add_event_features_and_resample*`, OFI/VPIN builders), z-encoding (`z_encoding`), subsequence/class distribution estimators, and — at module level, lines ~2015–2123 — the **batch driver** that actually runs everything and pickles results. Executing this file (or importing it) runs the whole pipeline as a side effect; there's no `if __name__ == "__main__":` guard. |
| `TrainingDistributions/integrate_day_distributions.py` | Pure, efficient dict-based aggregation of per-day distributions across a date range (`integrate_distributions`, `integrate_conditional_class_distributions`). Not a bottleneck. |
| `TrainingDistributions/plot_distributions.py` / `plot_saved_distributions.py` | Matplotlib bar-chart comparisons of saved distributions; `plot_saved_distributions.py` globs `SEQ_DISTR_*`/`CLS_DISTR_*` from the repo root and renders `outputs/*.png`. |
| `TrainingDistributions/time_series_analysis.py` | Standalone correlation / mutual-information / transfer-entropy toolkit for symbolic series; not called from the batch driver. |
| `LearningKraus.py` | PyTorch training script: loads one `SEQ_DISTR_*`/`CLS_DISTR_*` pickle, trains a Kraus-channel model (`KrausInstrument`) to match empirical sequence probabilities, saves weights. Also a top-level script (runs on import). |
| `CLS_DISTR_*`, `SEQ_DISTR_*` (repo root) | Pickled outputs of the batch driver — one pair per predictor feature, e.g. `SEQ_DISTR_NVDA_bivariate_log_mid-tvi_n_202504`. |
| `data/NVDA_INTC/*.dbn.zst` | Raw immutable input — **never write here.** |

## 3. Pipeline / data flow

```
 .dbn.zst (raw ticks, 1 file/day, ~250-300MB compressed)
        │  dbn_to_df()                         [full decompress + parse]
        ▼
 event-level DataFrame (millions of rows/day)
        │  add_event_features_and_resample()   [rolling windows, OFI, VPIN,
        │                                        event→n-event resampling]
        ▼
 resampled feature DataFrame (10-100x smaller)
        │  z_encoding()  (predicted + predictor columns)
        ▼
 bivariate symbol series (alphabet size 16 = 4x4)
        │  estimate_observed_subsequence_counts()   → sequence distributions
        │  estimate_subsequence_class_probabilities() → class-conditional distributions
        ▼
 per-day counts  ──integrate_day_distributions.py──▶  monthly aggregate
        │
        ▼
 SEQ_DISTR_* / CLS_DISTR_*  (pickled)  ──▶  LearningKraus.py (PyTorch training)
```

**Execution model:** everything is procedural top-level scripts with hardcoded
globals (`dates`, `features_list`, `fPath`, `symbol`), not functions/CLI/config
files. The batch driver (`process_distributions.py:2040-2122`) is:

```python
for predictor in features[1:]:        # 8 predictor features
    for date in dates:                # currently 2 dates hardcoded (line 2031);
                                       # full 45-day list present but commented out
                                       # at lines 2027-2030
        time_series = get_timeseries_by_date(...)   # full decode + featurize
        ...
        cl_distributions = estimate_subsequence_class_probabilities(...)
    pickle.dump(...)                  # one SEQ_DISTR_*/CLS_DISTR_* pair per predictor
```

It is **fully serial** — no `multiprocessing`, `joblib`, `concurrent.futures`,
or `dask` anywhere in `TrainingDistributions/` — and there is **no caching**:
the per-day featurized DataFrame is recomputed from the raw file independently
for every predictor.

## 4. Performance bottlenecks, ranked by impact

Evidence: the repo already contains one completed 2-date × 8-predictor run.
The 16 output-file timestamps show each predictor took **6–7 minutes** for
just 2 days, back-to-back with no overlap (confirmed via `stat` on the
`*_202504` files: 22:08 → 22:54, one file pair every ~6.5 min). Extrapolated
linearly to the full 45-day list that's already sitting commented-out in the
driver, that's **≈ 8 predictors × ~2.4 hours/predictor ≈ 19 hours** for one
full monthly run — for what is fundamentally a redundant computation (see
#1/#2 below). Exact split between causes needs profiling (recommended before
optimizing — see §6), but both are structurally confirmed by reading the code.

| Rank | Location | Issue | Why it matters |
|---|---|---|---|
| **1** | `process_distributions.py:190` `estimate_subsequence_class_probabilities`, called at `process_distributions.py:2058` | `possible_subseqs = list(product(unique_symbols, repeat=length))` enumerates the **full alphabet^length Cartesian product**, unconditionally, for every length 1..6. Alphabet = 16 symbols (bivariate 4×4 encoding) ⇒ length-6 alone is 16⁶ ≈ 16.8M tuples, ~18M total across lengths 1-6, built and dict-probed in pure Python — **every single (date, predictor) call**, i.e. up to 360 times for the full 45-day×8-predictor run. Note the sibling function used for the plain sequence distribution (`estimate_observed_subsequence_counts`, used at line 1947) already has an "observed-only" fast path — this class-distribution twin does not. |
| **2** | `process_distributions.py:2040-2046` (driver loop nesting) + `generate_timeseries` → `dbn_to_df` (`process_distributions.py:1386`, `read_databento_new.py:119-123`) | Loop order is `for predictor { for date { full decode+featurize } }`. The featurized time series (`get_timeseries_by_date`) does **not depend on which predictor** is being processed, yet it's fully rebuilt from the raw `.dbn.zst` (~250MB decompress + parse + rolling-window feature engineering) for every predictor. Full 45-day run: raw files get decoded and re-featurized **360 times instead of 45** — an 8x waste of the single largest I/O+compute stage. |
| **3** | `process_distributions.py:732-1020` `add_event_features_and_resample` (active hot path via `generate_timeseries`) | `sigma_W` is computed via `.rolling(W=300, min_periods=W).apply(lambda x: np.sqrt(np.mean(x*x)), raw=True)` — a Python-level callback re-executed once per event, over millions of events/day, recomputing the window sum from scratch each time rather than an incremental/vectorized RMS. Also a `for i, val in enumerate(s): ...` Python loop (line ~860) for the trade-sign fallback on `side == 'N'` rows. Compounds with bottleneck #2 (runs 8x more often than necessary). |
| **4** | `read_databento_new.py:308-343` `getBuckets2` (legacy, currently only reachable via the inactive univariate path) | `x = x + [current_bucket]` / `y = y + [mList]` inside a `for i in range(1, len(df))` loop — each `+` reallocates and copies the whole list, making bucket accumulation **O(n²)** in the number of ticks. Not on the currently-executed bivariate path, but a landmine if the univariate branch (`variate="univariate"`, still present and switchable) or `get_data_1`/`get_data` are ever re-enabled on tick-level data. |
| **5** | `read_databento_new.py:126-150` `zstd_to_df`; `read_databento_new.py:1456-1591` TE estimators (`transfer_entropy`, `transferEntropy`) | Legacy diagnostic path (`X_pred_Y`, volatility/xcorr/MI functions): `zstd_to_df` decompresses an entire day into one in-memory `StringIO` buffer before `pd.read_csv`; the transfer-entropy estimators use nested Python loops up to O(bins³) and are invoked repeatedly across `max_lag` sweeps. Not touched by the batch driver today, but expensive if any of `getVolatility1-4`/`X_pred_Y` are called ad hoc — each call independently re-decompresses the same day file (no cross-call caching). |
| **6** | Whole batch driver | **No parallelism at all.** Both the day loop and the (I/O-bound) predictor loop are naturally embarrassingly parallel — different days/predictors touch independent files and produce independent outputs merged only by the cheap, already-efficient `integrate_day_distributions.py`. Running serially throws away trivial multi-core speedup on top of the algorithmic waste above. |

## 5. Proposed interface

The root cause behind #1, #2 and #6 is architectural: there's no seam between
"read+featurize a day" and "encode+distribute for a predictor," so the driver
can't reuse work across predictors or dispatch days to workers. A minimal
interface fixes this without touching the underlying math:

```python
# pipeline.py (proposed)

@dataclass
class RunConfig:
    symbol: str
    data_path: Path
    dates: list[str]                 # explicit, no more commented-out lists
    predicted: str
    predictors: list[str]
    n_symbols: int = 4
    max_seq_length: int = 6
    alpha: float = 0.05
    cache_dir: Path | None = None    # parquet/feather cache for featurized days

class DayFeatureCache:
    """Reads + featurizes a raw .dbn.zst exactly once per day, regardless of
    how many predictors need it. Optionally persists to cache_dir so a rerun
    with a different predictor list doesn't re-touch data/ at all."""
    def get(self, date: str) -> pd.DataFrame: ...

class DistributionBuilder:
    """Pure function of (featurized_day, predicted, predictor) -> per-day
    counts. No I/O. This is what becomes a `ProcessPoolExecutor.map` target."""
    def sequence_counts(self, day_df, predicted, predictor) -> ...: ...
    def class_counts(self, day_df, predicted, predictor) -> ...: ...

def run(cfg: RunConfig) -> None:
    cache = DayFeatureCache(cfg)
    day_frames = {d: cache.get(d) for d in cfg.dates}      # loop over DAYS once
    for predictor in cfg.predictors:                        # reuse day_frames
        results = parallel_map(                             # across days
            DistributionBuilder().process,
            [(day_frames[d], cfg.predicted, predictor) for d in cfg.dates],
        )
        aggregate_and_dump(results, cfg, predictor)
```

What this buys, mapped back to the ranked list:

- **Fixes #2 directly** by inverting the loop nest — one decode+featurize per
  day (45, not 360) shared across all predictors.
- **Sets up #1's real fix**: once class-distribution enumeration is isolated
  in `DistributionBuilder`, swapping `estimate_subsequence_class_probabilities`
  to an observed-only variant (mirroring `estimate_observed_subsequence_counts`,
  which already exists and is used for the sibling sequence-distribution call)
  removes the unconditional 16⁶ enumeration without touching its call sites.
- **Fixes #6** — `DistributionBuilder.process` is a pure function of one day's
  data, so it's a natural `ProcessPoolExecutor`/`joblib` target; `DayFeatureCache`
  keys are read-only against `data/`, so parallel workers never write there.
- Turns `dates`/`predictors`/`symbol` from module-level globals requiring a
  code edit into a `RunConfig`, which also makes `LearningKraus.py`'s
  hardcoded `dates = ['20250430','20250501']` and `.env`-relative `fPath`
  driveable from the same config instead of a second copy of the same
  constants.
- A thin CLI (`python -m pipeline run --config run.yaml`) on top of `RunConfig`
  would let an agent (or you) launch the full 45-day/8-predictor run without
  editing source, and rerun just the training stage (`LearningKraus.py`)
  against a cached `SEQ_DISTR_*` without recomputing distributions.

## 6. Before optimizing

Recommend a quick `cProfile` pass on the **current 2-date config** (not the
full 45-day set) to confirm the actual split between bottleneck #1
(combinatorial enumeration) and #2/#3 (I/O + rolling-window featurization)
before investing in the refactor above — the ranking here is by structural/
algorithmic evidence (confirmed by reading the code and the observed
per-predictor cadence), not by a profiler trace. This does not require
touching or modifying any file under `data/`.

## 7. GPU acceleration roadmap (8× A100 target)

Context: an 8× A100 box is available on a separate machine. This section maps
each pipeline stage to whether/how it benefits, so effort goes where the
FLOPs and memory bandwidth actually help — not every stage in §4 is GPU-shaped,
and one stage (#1) is a better fit for a GPU *algorithm change* than for
"just add CUDA" to the existing Python.

**Sizing check first:** the full month is ~10GB compressed, likely 20-40GB
decompressed across all 45 days of tick data. That fits entirely in a single
A100's 80GB HBM — this is not a "shard across GPUs for memory reasons"
problem. The multi-GPU value here is running **independent** work
(predictors, days, hyperparameter variants) concurrently, not partitioning
one big tensor.

### 7.1 Highest impact: replace the combinatorial enumeration (bottleneck #1) with a GPU histogram

`estimate_subsequence_class_probabilities` (`process_distributions.py:190`)
builds `list(product(unique_symbols, repeat=length))` — up to 16⁶ ≈ 16.8M
Python tuples — then counts occurrences via a dict. This maps almost exactly
onto a GPU histogram:

1. Encode each length-*k* window of the symbol stream as a single mixed-radix
   integer key: `key = Σ symbol[i] * alphabet_size**i` (the code already does
   this exact trick when combining the two 4-symbol variates into one
   16-symbol stream at `process_distributions.py:1936-1938` — the same idea
   just needs to be applied along the *time* axis for each window length).
2. Build all windows at once with `tensor.unfold(0, k, 1)` (or
   `torch.as_strided`), producing a `[n_windows, k]` tensor with zero copies.
3. Reduce each row to its integer key with a vectorized weighted sum, then
   call `torch.bincount(keys, minlength=alphabet_size**k)` — one kernel
   launch computes the count for **every** possible subsequence, observed or
   not, replacing the Python `for subseq in possible_subseqs` loop entirely.
4. For the class-conditional version, do the same keying trick jointly with
   the class label (`key * num_classes + class_label`), bincount once, then
   reshape to `[alphabet_size**k, num_classes]`.

This doesn't require a training loop or even much GPU memory — it's pure
integer/scatter-add work, and it's the single change most likely to collapse
the current ~19-hour projected serial run, because it removes the dominant
term outright rather than making it faster per-op. It also batches trivially:
stack the (date, predictor) dimension as an extra batch axis and one GPU call
can compute all 45×8 histograms at once instead of the current 360 separate
Python passes.

### 7.2 Feature engineering (`add_event_features_and_resample`) → RAPIDS cuDF / CuPy

The rolling-window feature stage (bottleneck #3: `sigma_W` via
`.rolling(300).apply(python lambda)`, plus the various `.diff()`/`.shift()`/
`resample()` calls) is the kind of vectorized-but-CPU-bound work RAPIDS
targets directly:

- **Lowest effort:** run the script under `cudf.pandas` (`python -m
  cudf.pandas process_distributions.py`) — a zero-code-change accelerator
  that transparently offloads supported pandas operations to GPU and falls
  back to CPU for the rest. Worth trying first since it requires no rewrite.
- **Higher payoff, more effort:** port `add_event_features_and_resample` to
  explicit `cudf`/`cupy`. In particular, `sigma_W`'s rolling RMS Python
  callback is exactly the kind of user-defined rolling function that cuDF
  JIT-compiles via Numba CUDA instead of invoking a CPython callback per
  window — this alone should remove most of bottleneck #3's cost.
- Since a full day's tick DataFrame comfortably fits in GPU memory, the whole
  featurization stage for a day can run without round-tripping to host RAM.

### 7.3 EWMA z-encoding — batch across series rather than parallelize within one

`z_encoding`'s EWMA is a sequential recurrence (`y_t = a*y_{t-1} + (1-a)*x_t`),
which in principle has an O(log n) parallel-scan formulation (the same trick
used for linear-recurrence/state-space models), but this is low priority:
pandas' `.ewm()` is already a fast vectorized C loop, and per-series length
post-resampling is modest (thousands to low-hundred-thousands of points).
The actual GPU win available here is **batching across the 45×8 = 360
independent (date, predictor) series** as one 2D tensor `[360, T]` and running
the recurrence once across the batch dimension with vectorized ops, instead
of looping over predictors in Python. Treat this as a nice-to-have once §7.1
and §7.2 land, not a first step.

### 7.4 Still fix bottleneck #2 (redundant re-read/re-featurize) — GPU doesn't replace it

Moving feature engineering to GPU makes each pass cheaper, but the driver
still calls it 8× per day (once per predictor) unless the loop nesting from
§5's `DayFeatureCache` proposal is also applied. The two changes are
complementary: cache the featurized day once (on GPU, or spilled to Parquet),
then feed it to both the §7.1 histogram stage and §7.2's GPU feature builder
for every predictor without re-touching `data/`.

### 7.5 Multi-GPU: parallelize across predictors, not within the Kraus model

`LearningKraus.py`'s `KrausInstrument` is small — `d = 2**n_qubits` with
`n_qubits=3` means 8×8 complex matrices, `m=16` Kraus operators. A training
step is a handful of tiny batched matmuls; it's already effectively instant
per step on one GPU, and the loop over sequence length (`for t in range(T)`
in `sequence_prob_batch`) tops out at `T ≤ max_seq_len = 6`. Data-parallelizing
*this* model across 8 A100s would be dominated by inter-GPU communication
overhead for essentially no compute — not a good use of the hardware.

The better fit: today, training one predictor's model means editing globals
and running `LearningKraus.py` once per predictor, serially. With 8 GPUs,
assign each of the 8 predictor features to its own GPU
(`CUDA_VISIBLE_DEVICES=i`) and launch all 8 trainings concurrently as
separate processes — no model or algorithm changes required, just
orchestration (a small job-array / `multiprocessing` launcher keyed off the
`RunConfig.predictors` list from §5). That turns "N sequential model-training
runs" into one wall-clock run. Multi-GPU data-parallelism (DDP) only becomes
relevant if `n_qubits` or batch/sequence sizes grow substantially beyond
current settings.

### 7.6 Leave on CPU

- **zstd decompression of `.dbn.zst`** — inherently sequential; GPU zstd
  decompressors exist (nvCOMP) but add real complexity for a stage that isn't
  in the top-3 bottlenecks. Not worth it here.
- **Legacy diagnostics** (`time_series_analysis.py`, `pyinform`-based
  transfer entropy, `arch` GARCH fits in `read_databento_new.py`) — small,
  exploratory, not on the active driver path (§2/§4 rank 5). `pyinform` and
  `arch` have no GPU backends; porting them would be effort spent on code
  that isn't currently the cost driver.

### 7.7 Suggested phasing

1. **§7.1** (GPU histogram for subsequence/class counting) — highest ROI,
   single-GPU, ~1 day of work, removes the dominant cost outright.
2. **§7.4** (`DayFeatureCache`, from §5) — cheap, eliminates the 8× redundant
   re-read/re-featurize regardless of what else changes.
3. **§7.2** (`cudf.pandas` first, hand-port `sigma_W` if needed) — quick win
   once profiling (§6) confirms feature engineering is still worth the effort
   after #1 and #2 land.
4. **§7.5** (one predictor per GPU for `LearningKraus.py`) — pure
   orchestration, no model changes, immediate 8× wall-clock win on the
   training sweep.

Net effect: the ~19-hour projected serial run for distribution estimation
should collapse to low minutes (GPU histogram + cached single-pass
featurization), and the training sweep goes from 8 sequential single-GPU runs
to 1 concurrent 8-GPU run.
