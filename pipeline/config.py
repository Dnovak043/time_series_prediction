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


def _f(default, help="", choices=None, advanced=False, **kw):
    md = {"help": help}
    if choices:
        md["choices"] = choices
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
    data_path: str = _f("data/NVDA_INTC",
                        "Directory with raw .dbn.zst files, relative to repo root.")
    file_pattern: str = _f("xnas-itch-{date}.mbp-10.dbn.zst",
                           "Raw file name pattern; {date} is replaced per day.",
                           advanced=True)
    dates: list = _f(lambda: ["20250401", "20250402"],
                     "Trading days to process, as yyyymmdd strings.")
    session_start: str = _f("09:30", "Session start, Eastern time (HH:MM).")
    session_end: str = _f("15:30", "Session end, Eastern time (HH:MM).")


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
                          "output pair is produced per predictor.")
    max_seq_length: int = _f(6, "Maximum subsequence (n-gram) length.")
    sequence_calculation: bool = _f(True, "Compute sequence distributions "
                                          "(SEQ_DISTR_* outputs).")
    class_calculation: bool = _f(True, "Compute class-conditional distributions "
                                       "(CLS_DISTR_* outputs).")
    class_name: str = _f("c1", "Forward-move class definition (c{k}: k-step "
                               "return sign; ca{k}: fwd vs bwd sum).",
                         choices=["c1", "c2", "c4", "ca1", "ca2", "ca4"])
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
class TrainingConfig:
    """SEQ_DISTR_* pickle -> trained sequence model."""
    model: str = _f("kraus", "Model family; registered trainers appear in "
                             "pipeline.models.REGISTRY. More can be plugged in.",
                    choices=["kraus"])
    predictor: str = _f("tvi_n", "Which predictor's SEQ_DISTR_* file to train on.")
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
    device: str = _f("auto", "Compute device. auto = cuda if available else cpu "
                             "(mps is opt-in: complex-tensor support is limited).",
                     choices=["auto", "cuda", "cpu", "mps"])
    continue_from: str = _f("", "Path to WGHTS_*.pt weights to resume from; "
                                "empty = fresh start.", advanced=True)
    model_dir: str = _f(".", "Where MOD_*/WGHTS_* model files are written.")


# ---------------------------------------------------------------------------
STAGES = {
    "data": DataConfig,
    "featurize": FeaturizeConfig,
    "encode": EncodeConfig,
    "distributions": DistributionConfig,
    "training": TrainingConfig,
}


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    featurize: FeaturizeConfig = field(default_factory=FeaturizeConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    distributions: DistributionConfig = field(default_factory=DistributionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # -- derived values -----------------------------------------------------
    @property
    def alphabet_size(self) -> int:
        return self.encode.n_symbols ** 2   # bivariate encoding

    def vol_window(self) -> int:
        return self.featurize.vol_window or 3 * self.featurize.frequency

    def seq_distr_name(self, predictor: str) -> str:
        return ("SEQ_DISTR_" + self.data.symbol + "_bivariate_"
                + self.distributions.predicted + "-" + predictor
                + "_" + self.data.dates[0][:6])

    def cls_distr_name(self, predictor: str) -> str:
        return ("CLS_DISTR_" + self.data.symbol + "_bivariate_"
                + self.distributions.predicted + "-" + predictor
                + "_" + self.data.dates[0][:6])

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
        if not self.data.dates:
            problems.append("data.dates is empty")
        for d in self.data.dates:
            if not (len(str(d)) == 8 and str(d).isdigit()):
                problems.append(f"data.dates entry {d!r} is not yyyymmdd")
        if self.encode.n_symbols < 2:
            problems.append("encode.n_symbols must be >= 2")
        if self.distributions.max_seq_length < 1:
            problems.append("distributions.max_seq_length must be >= 1")
        if not 0.0 <= self.distributions.sample_size <= 1.0:
            problems.append("distributions.sample_size must be in [0, 1]")
        if (self.distributions.predicted in self.distributions.predictors):
            problems.append("distributions.predicted also listed in predictors")
        if self.training.predictor not in self.distributions.predictors:
            problems.append(f"training.predictor {self.training.predictor!r} not "
                            "in distributions.predictors")
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
            "advanced": f.metadata.get("advanced", False),
        })
    return out
