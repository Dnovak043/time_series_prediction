# -*- coding: utf-8 -*-
"""
Bridge back to the pipeline: turn a feature set chosen in the lab into a
RunConfig YAML, so its effect on the real Kraus models can be measured with
the pipeline itself (the lab never trains Kraus models on its own).

    path = bridge.pipeline_config("configs/april_nvda.yaml", ["tvi_n", "vpin"],
                                  tag="pc_parents", multivariate=True)
    # then:  python -m pipeline run --config <path>   (and `train`)

Outputs are redirected under outputs/lab/<tag>/ so nothing collides with the
April run; the feature cache dir is kept, so featurized days are reused.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import LAB_ROOT, REPO_ROOT
from . import data as _data


def _rel(p: Path) -> Path:
    try:
        return p.relative_to(REPO_ROOT)
    except ValueError:
        return p


def pipeline_config(base, predictors: Sequence[str], tag: str,
                    multivariate: bool = False, train_predictor=None,
                    out_dir: Path | None = None) -> Path:
    """
    predictors      pipeline feature names (level names like `log_mid`, not the
                    lab's `d_log_mid`; the pipeline's predicted column is fixed
                    by the base config)
    multivariate    also emit one joint SEQ_DISTR over all predictors and train
                    on it (alphabet n_symbols^(1+len) — mind the size)
    train_predictor override which predictor stage 3 trains on (default: the
                    joint list if multivariate, else the first predictor)
    """
    cfg = _data.load_config(base)
    preds = [p.removeprefix("d_") if p.removeprefix("d_") in _data.LEVEL_COLUMNS else p
             for p in predictors]
    preds = [p for p in preds if p != cfg.distributions.predicted]
    if not preds:
        raise ValueError("no predictors left after removing the predicted column")
    out = f"outputs/lab/{cfg.data.symbol}/{tag}"
    cfg.distributions.predictors = list(preds) + ([list(preds)] if multivariate else [])
    cfg.distributions.output_dir = out
    cfg.ensemble.output_dir = f"{out}/ensemble"
    cfg.ensemble.predictors = [cfg.distributions.predicted] + list(preds)
    cfg.training.model_dir = f"{out}/models"
    cfg.training.predictor = (train_predictor if train_predictor is not None
                              else (list(preds) if multivariate else preds[0]))
    cfg.training.seq_distr_file = ""
    errs = cfg.validate() if hasattr(cfg, "validate") else []
    if errs:
        raise ValueError(f"generated config is invalid: {errs}")
    path = Path(out_dir or LAB_ROOT / "runs" / "configs") / f"{cfg.data.symbol}_{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(path)
    print(f"wrote {_rel(path)}\n"
          f"  predictors: {cfg.distributions.predictors}\n"
          f"  train on:   {cfg.training.predictor}\n"
          f"  outputs:    {out}/")
    return path
