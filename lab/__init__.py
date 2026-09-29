# -*- coding: utf-8 -*-
"""
lab — a sandbox for feature-relationship experiments.

Causal discovery, dependence metrics and model-ablation studies over the
LOB features the pipeline produces. The lab only *reads* from the pipeline
(RunConfig, DayFeatureCache, the vendored z-encoding); it never changes
pipeline behaviour, and nothing in pipeline/ imports it.

    from lab import data, metrics, causal, effects, plots
    panel = data.load("configs/april_nvda.yaml", dates=2)

Modules
    data       load feature panels (one row per resampled block, `day` column)
    metrics    pairwise / lead-lag dependence: corr, MI, Granger, transfer entropy
    causal     PC (contemporaneous), PCMCI (time-lagged), optional tigramite/lingam
    effects    walk-forward model evaluation, feature-set ablation, importance
    plots      heatmaps, causal graphs, ablation charts
    runs       Experiment: save spec + tables + figures under lab/runs/
    synthetic  VAR processes with a known causal graph (to sanity-check methods)
    bridge     turn a chosen feature set into a pipeline RunConfig for Kraus runs
"""
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LAB_ROOT.parent
RUNS_DIR = LAB_ROOT / "runs"
