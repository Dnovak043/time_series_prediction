# -*- coding: utf-8 -*-
"""
Configuration schema for the whole pipeline: one dataclass per stage.

Field ``metadata`` carries UI hints: ``help`` (tooltip text), ``choices``
(renders as a dropdown), ``advanced`` (collapsed by default). The ipywidgets
panel and the CLI are both generated from this schema, so adding a field
here is all that's needed to expose a new knob everywhere.

Defaults reproduce the hardcoded values of the original scripts exactly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


_DEVICE_PRESETS = ("auto", "cuda", "cpu", "mps")


def _validate_device(device) -> list[str]:
    """training.device is free-form: the presets, or an indexed accelerator
    like 'cuda:3' / 'mps:0'. Anything else is a typo that would otherwise
    surface only when torch failed mid-training."""
    import re

    if not isinstance(device, str):
        return [f"training.device {device!r} must be a string"]
    if device in _DEVICE_PRESETS:
        return []
    if re.fullmatch(r"(cuda|mps):\d+", device):
        return []
    return [f"training.device {device!r} is not one of "
            f"{list(_DEVICE_PRESETS)} or an indexed device like 'cuda:0'"]


def predictor_key(predictor) -> str:
    """Stable string key for a predictor spec: a feature name, or a
    '+'-joined tag for a multivariate (list) predictor. Used for dict
    keys and per-predictor file names inside outputs/runs."""
    if isinstance(predictor, str):
        return predictor
    return "+".join(predictor)


def _f(default, help="", choices=None, advanced=False, free_form=False,
       options_provider=None, **kw):
    """free_form=True: `choices` are the common presets, but other values are
    legal too (validated by RunConfig.validate). The control panel renders
    such a field as the preset dropdown *plus* a second chooser, so e.g.
    training.device='cuda:3' is representable.

    options_provider: name of a provider registered in pipeline.ui that
    enumerates the extra values for that second chooser (e.g. 'devices' ->
    the accelerators present on this machine). With a provider the second
    chooser is a dropdown, so only real values can be picked; without one it
    falls back to a free-text box."""
    md = {"help": help}
    if choices:
        md["choices"] = choices
    if free_form:
        md["free_form"] = True
    if options_provider:
        md["options_provider"] = options_provider
    if advanced:
        md["advanced"] = True
    if callable(default):
        return field(default_factory=default, metadata=md, **kw)
    return field(default=default, metadata=md, **kw)


# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Which raw market data to read (read-only; data/ is never written)."""
    symbol: str = _f("NVDA", "Ticker symbol (used in file naming and outputs).")
    asset_paths: dict = _f(lambda: {},
                           "Per-asset raw-data directory, e.g. {NVDA: "
                           "data/NVDA_INTC, AAPL: data/AAPL}. Files share "
                           "names across directories but hold different "
                           "assets; several assets may map to the same "
                           "directory (NVDA and INTC share one file). The "
                           "run reads from the entry for `symbol`; symbols "
                           "not listed fall back to data_path. Empty = "
                           "always use data_path (legacy behavior).")
    data_path: str = _f("data/NVDA_INTC",
                        "Directory with raw .dbn.zst files, relative to repo "
                        "root. Fallback when `symbol` has no asset_paths "
                        "entry.")
    file_pattern: str = _f("xnas-itch-{date}.mbp-10.dbn.zst",
                           "Raw file name pattern; {date} is replaced per day.",
                           advanced=True)
    dates: list = _f(lambda: ["20250401", "20250402"],
                     "Trading days to process, as yyyymmdd strings.")
    instrument_filter: bool = _f(False,
                                 "Filter the raw event stream to `symbol` "
                                 "before featurizing. The raw files carry "
                                 "every subscribed symbol (NVDA+INTC "
                                 "interleaved); false = legacy behavior "
                                 "(unfiltered mixed stream, matches the "
                                 "frozen baseline), true = required for "
                                 "per-symbol runs.")
    session_start: str = _f("09:30", "Session start, Eastern time (HH:MM).")
    session_end: str = _f("15:30", "Session end, Eastern time (HH:MM).")

    def resolved_data_path(self) -> str:
        """Directory holding this run's raw files: the `symbol` entry in
        asset_paths when present, else data_path."""
        return (self.asset_paths or {}).get(self.symbol, self.data_path)


@dataclass
class FeaturizeConfig:
    """Event-stream -> resampled LOB feature DataFrame (per day)."""
    resampling: str = _f("events", "Resampling clock.",
                         choices=["events", "seconds", "volume"])
    frequency: int = _f(100, "Resampling window: N events, N seconds, or N shares "
                             "depending on the resampling mode.")
    forward_intervals: list = _f(lambda: [1, 2, 3, 4],
                                 "Forward horizons (in resampled steps) for "
                                 "log_mid_return_fwd_k / sum_fwd_k target columns.")
    vol_window: int = _f(0, "Trailing event window W for sigma_W volatility. "
                            "0 = automatic (3 x frequency, the original default).")
    workers: int = _f(1, "Parallel day workers for the distribution run: "
                         "1 = serial, 0 = auto (one per CPU core, capped at "
                         "the day count), N = exactly N. Output is "
                         "byte-identical for any value; each worker holds one "
                         "day's event data in RAM.")
    use_cache: bool = _f(True, "Reuse a cached featurized day if the parameters "
                               "match, instead of re-decoding the raw file.")
    cache_dir: str = _f("outputs/feature_cache",
                        "Where cached featurized days are stored.", advanced=True)
    cache_format: str = _f("parquet", "On-disk format for cached days.",
                           choices=["parquet", "pickle"], advanced=True)


@dataclass
class EncodeConfig:
    """Continuous features -> discrete symbols (EWMA z-score + binning)."""
    n_symbols: int = _f(4, "Symbols per variate; bivariate alphabet is n_symbols^2.")
    alpha: float = _f(0.05, "EWMA decay for the predictive z-score.")
    bins_mode: str = _f("expanding_quantile", "How z-score bin edges are fitted.",
                        choices=["expanding_quantile", "daily_quantile",
                                 "prefix_quantile", "fixed_uniform", "fixed_normal"])
    fill_mode: str = _f("bfill", "Warmup/NaN handling for symbols. bfill matches the "
                                 "original driver (non-causal at series start).",
                        choices=["bfill", "ffill", "bfill_ffill", "none"])
    min_periods: int = _f(2, "Minimum observations before the z-score is defined.",
                          advanced=True)


@dataclass
class DistributionConfig:
    """Symbol series -> subsequence and class-conditional distributions."""
    predicted: str = _f("log_mid", "Feature being predicted (first variate).")
    predictors: list = _f(lambda: ["tvi_n", "obi_L1", "ofi_L1_n_norm",
                                   "ofi_L3_norm_n", "ofi_L10_norm_n", "ofi_L1_n",
                                   "ofi_L1_norm_n", "micro_price"],
                          "Predictor features (second variate); one SEQ/CLS "
                          "output pair is produced per predictor. A nested "
                          "list entry is one multivariate predictor: "
                          "predicted + the listed features jointly encoded "
                          "(get_z_ts math, alphabet n_symbols^(1+len)) into "
                          "one SEQ_DISTR_{sym}_multivariate_* output "
                          "(SEQ only — the colleague defines no "
                          "multivariate CLS; class-conditional joint stats "
                          "live in the ensemble stage).")
    max_seq_length: int = _f(6, "Maximum subsequence (n-gram) length.")
    sequence_calculation: bool = _f(True, "Compute sequence distributions "
                                          "(SEQ_DISTR_* outputs).")
    class_calculation: bool = _f(True, "Compute class-conditional distributions "
                                       "(CLS_DISTR_* outputs).")
    class_name: str = _f("c1", "Forward-move class definition (c{k}: k-step "
                               "return sign; ca{k}: fwd vs bwd sum).",
                         choices=["c1", "c2", "c4", "ca1", "ca2", "ca4"])
    class_names: list = _f(lambda: [],
                           "V2 multi-class sweep (cls_reference.py): one "
                           "CLS output per listed class, named "
                           "CLS_DISTR_{sym}__{predicted}-{predictor}_{month}_"
                           "{cls}, with class columns ordered by "
                           "class_values. EMPTY = legacy single-class mode "
                           "(class_name above, old column order "
                           "[P(0),P(+1),P(-1)], old naming) — the frozen-"
                           "baseline behavior.")
    class_values: list = _f(lambda: [-1, 0, 1],
                            "Class column order for the v2 sweep "
                            "([P(-1),P(0),P(+1)] by default). Ignored in "
                            "legacy mode.", advanced=True)
    class_theta: float = _f(0.0, "Class threshold theta. 0 = built-in default "
                                 "for the chosen class_name.")
    num_classes: int = _f(3, "Number of classes (down / flat / up).", advanced=True)
    sample_size: float = _f(1.0, "Fraction of observed subsequences kept for "
                                 "lengths above sample_after_length.", advanced=True)
    sample_after_length: int = _f(30, "Apply sample_size only to lengths above "
                                      "this.", advanced=True)
    random_state: int = _f(42, "Seed for subsequence sampling.", advanced=True)
    output_dir: str = _f(".", "Where SEQ_DISTR_*/CLS_DISTR_* pickles are written "
                              "(repo root matches the original scripts).")


@dataclass
class EnsembleConfig:
    """Fixed-length multi-channel ensemble training tables (ENS_TD_* pickles).

    Implements the colleague's ensemble_training_data experiment: for each
    (sequence length, class definition), align every channel's encoding on
    identical timestamps and count joint + per-channel marginal occurrences
    with class-conditional distributions. The counting math is imported
    verbatim from the vendored reference selected by `reference`."""
    reference: str = _f("v2", "Which vendored colleague program the stage "
                              "reproduces. v2 = ensemble_reference_2.py: "
                              "channels may be lists (jointly encoded with "
                              "the predicted feature, get_z_ts), file suffix "
                              "'ALL'. v1 = ensemble_reference.py: bivariate "
                              "string channels only, suffix 'ALL_{n}', fixed "
                              "16-symbol alphabet validation.",
                        choices=["v1", "v2"])
    seq_lengths: list = _f(lambda: [1, 2, 3, 4, 5],
                           "Fixed sequence lengths, one ENS_TD_* output set "
                           "per length. (Colleague's script: 1-5.)")
    class_names: list = _f(lambda: ["c1", "c2", "ca2", "ca4"],
                           "Class definitions to sweep; one output set per "
                           "class per length. (v2 driver default; the v1 "
                           "driver also swept c4.)")
    class_values: list = _f(lambda: [-1, 0, 1],
                            "Ordered class labels; output distribution "
                            "columns follow this order: [P(-1), P(0), P(1)].",
                            advanced=True)
    predictors: list = _f(lambda: ["ofi_L10_norm_n", "micro_price", "vpin",
                                   ["ofi_L10_norm_n", "micro_price", "vpin"]],
                          "Ensemble channels. A string is one bivariate "
                          "encoding vs `distributions.predicted`; a nested "
                          "list (v2 only) is one joint encoding of predicted "
                          "+ the listed features, alphabet n_symbols^(1+len). "
                          "Default = v2 driver: 3 bivariate + 1 joint channel.")
    smoothing: float = _f(0.0, "Symmetric Dirichlet pseudocount for the "
                               "target class distributions; 0 = raw empirical.",
                          advanced=True)
    output_dir: str = _f("outputs/ensemble",
                         "Where ENS_TD_* pickles are written.")


@dataclass
class TrainingConfig:
    """SEQ_DISTR_* pickle -> trained sequence model."""
    model: str = _f("kraus", "Model family; registered trainers appear in "
                             "pipeline.models.REGISTRY. More can be plugged in.",
                    choices=["kraus"])
    predictor: str = _f("tvi_n", "Which predictor's SEQ_DISTR_* file to train "
                                 "on. A list (e.g. [ofi_L10_norm_n, "
                                 "micro_price, vpin]) trains on that "
                                 "multivariate SEQ_DISTR_* file with alphabet "
                                 "m = n_symbols^(1+len) — the "
                                 "LearningKraus_multivariate driver.")
    predictor_abbrev: str = _f("", "Short tag replacing the predictor part of "
                                   "MOD_/WGHTS_ file names for multivariate "
                                   "(list) training — e.g. 'L10_micro_vpin', "
                                   "the colleague's hand-written abbreviation. "
                                   "Empty = derive from the SEQ_DISTR file "
                                   "name. Ignored for string predictors.",
                               advanced=True)
    seq_distr_file: str = _f("", "Explicit path to a SEQ_DISTR_* pickle; empty = "
                                 "derive from the distribution settings above.")
    n_qubits: int = _f(3, "System register size; Hilbert dim d = 2^n_qubits.")
    epochs: int = _f(3000, "Training epochs.")
    batch_size: int = _f(3072, "Batch size (sequences per step).")
    lr: float = _f(1e-3, "Learning rate.")
    optimizer: str = _f("adam", "Optimizer.",
                        choices=["adam", "adamw", "sgd", "rmsprop"])
    loss_kind: str = _f("nll_seq", "Training objective.",
                        choices=["nll_seq", "mse_prob"])
    length_mixture: str = _f("uniform", "Reweighting of sequence lengths.",
                             choices=["uniform", "geometric", "none"], advanced=True)
    learn_rho0: bool = _f(True, "Learn the initial state rho0 (vs fixed |0><0|).",
                          advanced=True)
    max_seq_len: int = _f(6, "Drop training sequences longer than this.")
    min_seq_prob: float = _f(0.0, "Drop training sequences with empirical "
                                  "probability below this.", advanced=True)
    num_workers: int = _f(0, "DataLoader worker processes for training. "
                             "0 = load in the training process. Speed only, "
                             "never results: shuffling is done by the sampler "
                             "in the parent and SeqDataset is a pure index "
                             "lookup, so batch composition is identical for "
                             "any worker count. NOTE: LearningKraus.train() "
                             "ignored this until [vendoring fix 1] made it "
                             "live, so the original main()'s num_workers=8 "
                             "never actually took effect — 8 reproduces its "
                             "stated intent. Keep low (or 0) when trainings "
                             "are fanned across GPUs: each is already a "
                             "subprocess, and workers nest under it.",
                          advanced=True)
    eval_batch_size: int = _f(2048, "Batch size for the post-training "
                                    "predict_probs evaluation pass (the "
                                    "original main() used 2*1024). Memory/"
                                    "speed only — the probabilities are "
                                    "identical for any batch size.",
                              advanced=True)
    plot_entries: int = _f(200, "How many sequences to chart with "
                                "plotDistributions after training. It draws "
                                "in chunks of 62, so 200 -> the "
                                "characteristic 4 PNGs per model. 0 = skip "
                                "charting.")
    plot_dpi: int = _f(120, "Resolution of the saved chart PNGs. Affects the "
                            "delivered image files, so it is a run "
                            "parameter, not a display preference.",
                       advanced=True)
    device: str = _f("auto", "Compute device. auto = cuda if available else cpu "
                             "(mps is opt-in: complex-tensor support is limited). "
                             "Besides these presets an explicit 'cuda:N' pins "
                             "one training to one GPU — that is how the April "
                             "stage-3 fan-out schedules, and such values are "
                             "written into each per-model config.yaml. The "
                             "panel offers only the accelerators present on "
                             "this machine, plus whatever the loaded config "
                             "already names (configs are portable between the "
                             "Mac and the compute box).",
                     choices=["auto", "cuda", "cpu", "mps"], free_form=True,
                     options_provider="devices")
    continue_from: str = _f("", "Path to WGHTS_*.pt weights to resume from; "
                                "empty = fresh start.", advanced=True)
    model_dir: str = _f(".", "Where MOD_*/WGHTS_* model files are written.")
    predictors: list = _f(lambda: [],
                          "Predictors for `train-all` (one model per predictor); "
                          "empty = train all of distributions.predictors.",
                          advanced=True)
    gpus: str = _f("auto", "GPUs for `train-all`: 'auto' = every CUDA device "
                           "visible to torch, a comma list like '0,2,5', or "
                           "'none' to force CPU.", advanced=True)
    max_parallel: int = _f(0, "Max concurrent trainings in `train-all`. "
                              "0 = auto: one per GPU, else 1 (CPU).")
    seed: int = _f(-1, "Torch seed for training. -1 = unseeded, the original "
                       "main() behavior (results vary run to run).")


# ---------------------------------------------------------------------------
STAGES = {
    "data": DataConfig,
    "featurize": FeaturizeConfig,
    "encode": EncodeConfig,
    "distributions": DistributionConfig,
    "ensemble": EnsembleConfig,
    "training": TrainingConfig,
}


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    featurize: FeaturizeConfig = field(default_factory=FeaturizeConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    distributions: DistributionConfig = field(default_factory=DistributionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # -- derived values -----------------------------------------------------
    @property
    def alphabet_size(self) -> int:
        return self.encode.n_symbols ** 2   # bivariate encoding

    def alphabet_size_for(self, predictor) -> int:
        """Symbol-alphabet size for one predictor spec: n_symbols^2 for a
        string (bivariate), n_symbols^(1+len) for a list (multivariate)."""
        if isinstance(predictor, str):
            return self.alphabet_size
        return self.encode.n_symbols ** (1 + len(predictor))

    def vol_window(self) -> int:
        return self.featurize.vol_window or 3 * self.featurize.frequency

    def seq_distr_name(self, predictor) -> str:
        if not isinstance(predictor, str):
            # verbatim from the LearningKraus_multivariate driver: predicted
            # + first + last predictor, dash-joined
            p = list(predictor)
            return ("SEQ_DISTR_" + self.data.symbol + "_multivariate_"
                    + self.distributions.predicted + "-" + p[0] + "-" + p[-1]
                    + "_" + self.data.dates[0][:6])
        return ("SEQ_DISTR_" + self.data.symbol + "_bivariate_"
                + self.distributions.predicted + "-" + predictor
                + "_" + self.data.dates[0][:6])

    def cls_distr_name(self, predictor: str, cls_name: str | None = None) -> str:
        if cls_name is None:   # legacy single-class naming (frozen baseline)
            return ("CLS_DISTR_" + self.data.symbol + "_bivariate_"
                    + self.distributions.predicted + "-" + predictor
                    + "_" + self.data.dates[0][:6])
        # v2 naming, verbatim from cls_reference.py's driver (including its
        # double underscore and dropped "bivariate")
        return ("CLS_DISTR_" + self.data.symbol + "_" + "_"
                + self.distributions.predicted + "-" + predictor
                + "_" + self.data.dates[0][:6] + "_" + cls_name)

    # -- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {name: dataclasses.asdict(getattr(self, name)) for name in STAGES}

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        kwargs = {}
        for name, stage_cls in STAGES.items():
            stage_d = dict(d.get(name, {}) or {})
            known = {f.name for f in fields(stage_cls)}
            unknown = set(stage_d) - known
            if unknown:
                raise ValueError(f"Unknown key(s) in '{name}' section: "
                                 f"{sorted(unknown)}")
            kwargs[name] = stage_cls(**stage_d)
        return cls(**kwargs)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False,
                           default_flow_style=None)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    # -- validation -----------------------------------------------------------
    def validate(self) -> list[str]:
        """Returns a list of problems (empty = ok). Cheap checks only."""
        problems = []
        # closed `choices` fields must hold one of their options. Nothing
        # enforced this before, so an out-of-set value survived save/load and
        # only blew up later at widget construction (ipywidgets Dropdown
        # raises TraitError when value is not in options).
        for stage_name in STAGES:
            stage_obj = getattr(self, stage_name)
            for info in field_info(stage_obj):
                if not info["choices"] or info["free_form"]:
                    continue
                if info["value"] not in info["choices"]:
                    problems.append(
                        f"{stage_name}.{info['name']} {info['value']!r} is not "
                        f"one of {info['choices']}")
        problems += _validate_device(self.training.device)
        if not self.data.dates:
            problems.append("data.dates is empty")
        for d in self.data.dates:
            if not (len(str(d)) == 8 and str(d).isdigit()):
                problems.append(f"data.dates entry {d!r} is not yyyymmdd")
        if not isinstance(self.data.asset_paths, dict):
            problems.append("data.asset_paths must be a mapping "
                            "{symbol: directory}")
        else:
            for sym, p in self.data.asset_paths.items():
                if not isinstance(sym, str) or not isinstance(p, str):
                    problems.append(f"data.asset_paths entry {sym!r}: {p!r} "
                                    "must be string: string")
        if self.encode.n_symbols < 2:
            problems.append("encode.n_symbols must be >= 2")
        if self.distributions.max_seq_length < 1:
            problems.append("distributions.max_seq_length must be >= 1")
        if not 0.0 <= self.distributions.sample_size <= 1.0:
            problems.append("distributions.sample_size must be in [0, 1]")
        dist_str = [p for p in self.distributions.predictors
                    if isinstance(p, str)]
        dist_lists = [p for p in self.distributions.predictors
                      if not isinstance(p, str)]
        if self.distributions.predicted in dist_str:
            problems.append("distributions.predicted also listed in predictors")
        for p in dist_lists:
            variables = [self.distributions.predicted] + list(p)
            if len(set(variables)) != len(variables):
                problems.append(f"multivariate predictor {list(p)!r}: "
                                f"duplicate variables in joint encoding "
                                f"{variables}")
        dist_keys = [predictor_key(p) for p in self.distributions.predictors]
        if predictor_key(self.training.predictor) not in dist_keys:
            problems.append(f"training.predictor {self.training.predictor!r} not "
                            "in distributions.predictors")
        list_channels = [ch for ch in self.ensemble.predictors
                         if not isinstance(ch, str)]
        if self.ensemble.reference == "v1" and list_channels:
            problems.append("ensemble.reference 'v1' supports only string "
                            "(bivariate) channels; list channels need 'v2'")
        for ch in list_channels:
            variables = [self.distributions.predicted] + list(ch)
            if len(set(variables)) != len(variables):
                problems.append(f"ensemble channel {list(ch)!r}: duplicate "
                                f"variables in joint encoding {variables} "
                                "(get_z_ts would raise)")
        return problems


def field_info(stage_obj) -> list[dict[str, Any]]:
    """UI helper: name/value/type/help/choices/advanced for each field."""
    out = []
    for f in fields(stage_obj):
        out.append({
            "name": f.name,
            "value": getattr(stage_obj, f.name),
            "type": f.type,
            "help": f.metadata.get("help", ""),
            "choices": f.metadata.get("choices"),
            "free_form": f.metadata.get("free_form", False),
            "options_provider": f.metadata.get("options_provider"),
            "advanced": f.metadata.get("advanced", False),
        })
    return out
