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
| `session_start` / `session_end` | 09:30 / 15:30 | Eastern-time RTH window |

### featurize — event stream → resampled LOB feature bars
| field | default | meaning |
|---|---|---|
| `resampling` | events | bar clock: `events`, `seconds`, or `volume` |
| `frequency` | 100 | bar size: N events / N seconds / N shares |
| `forward_intervals` | [1,2,3,4] | horizons for `log_mid_return_fwd_k` etc. |
| `vol_window` | 0 (auto) | trailing window W for `sigma_W`; 0 = 3×frequency |
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
| `class_theta` | 0 (auto) | class threshold θ; 0 = built-in default per class |
| `num_classes` | 3 | down / flat / up |
| `sample_size` / `sample_after_length` / `random_state` | 1.0 / 30 / 42 | optional support subsampling for long n-grams |
| `output_dir` | . | where SEQ_DISTR_*/CLS_DISTR_* land (repo root = legacy behavior) |

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

## 3. Three ways to drive it (same config file)

**Control panel (JupyterLab or VS Code notebook):** open
`pipeline_control.ipynb`, run the first cell. Edit fields in the stage tabs →
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

## 6. Verification

- `tests/test_pipeline_units.py` — config/YAML/UI-widget roundtrips, cache
  keying, import-safety of the guarded legacy scripts.
- `tests/parity_check.py` — runs the **original driver** and the **new
  runner** on the same real day and asserts the SEQ_DISTR_*/CLS_DISTR_*
  outputs are identical, and that a cache-served rerun changes nothing.

Note: the GPU histogram from PR #1 (torch unfold+bincount) is intentionally
**not** on this branch — it forked from main per review isolation. Once PR #1
merges, `DistributionBuilder.class_counts` is the single call site to switch
over.
