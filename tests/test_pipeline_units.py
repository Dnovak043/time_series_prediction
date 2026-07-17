# -*- coding: utf-8 -*-
"""
Fast unit tests for the pipeline package (no market data needed).

Run:  .env/bin/python tests/test_pipeline_units.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import RunConfig  # noqa: E402
from pipeline.features import DayFeatureCache, _parse_hhmm  # noqa: E402


def test_yaml_roundtrip():
    cfg = RunConfig()
    cfg.encode.alpha = 0.123
    cfg.data.dates = ["20250415"]
    cfg.distributions.predictors = ["tvi_n"]
    cfg.training.predictor = "tvi_n"
    with tempfile.TemporaryDirectory() as d:
        path = cfg.save(Path(d) / "cfg.yaml")
        cfg2 = RunConfig.load(path)
    assert cfg.to_dict() == cfg2.to_dict()
    print("  PASS yaml roundtrip")


def test_unknown_key_rejected():
    try:
        RunConfig.from_dict({"encode": {"n_symbols": 4, "typo_field": 1}})
    except ValueError as e:
        assert "typo_field" in str(e)
        print("  PASS unknown key rejected")
        return
    raise AssertionError("unknown key was accepted")


def test_validate():
    cfg = RunConfig()
    cfg.data.dates = ["2025-04-01"]          # wrong format
    cfg.training.predictor = "nonexistent"
    problems = cfg.validate()
    assert any("yyyymmdd" in p for p in problems), problems
    assert any("nonexistent" in p for p in problems), problems
    assert RunConfig().validate() == []
    print("  PASS validate")


def test_cache_key():
    cfg = RunConfig()
    c1 = DayFeatureCache(cfg.data, cfg.featurize)
    k1 = c1.params_key()
    assert c1.params_key() == k1                      # stable
    cfg.featurize.frequency = 200
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() != k1
    cfg.featurize.frequency = 100
    cfg.featurize.use_cache = False                   # non-output param:
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() == k1
    print("  PASS cache key semantics")


def test_parse_hhmm():
    import datetime
    assert _parse_hhmm("09:30") == datetime.time(9, 30)
    assert _parse_hhmm(570) == datetime.time(9, 30)   # YAML base-60 footgun
    print("  PASS session time parsing")


def test_legacy_imports_side_effect_free():
    import matplotlib
    matplotlib.use("Agg")
    import pipeline  # noqa: F401  (sets sys.path)
    import process_distributions as pdst
    import LearningKraus as lk
    assert callable(pdst.run_legacy_driver) and callable(lk.main)
    print("  PASS legacy modules import without running")


def test_model_registry():
    from pipeline.models import REGISTRY
    assert "kraus" in REGISTRY
    print("  PASS model registry")


def test_multivariate_predictor():
    from pipeline.config import predictor_key

    trio = ["ofi_L10_norm_n", "micro_price", "vpin"]
    cfg = RunConfig()
    cfg.data.symbol = "NVDA"
    cfg.data.dates = ["20250401"]
    cfg.distributions.predictors = ["tvi_n", trio]
    cfg.training.predictor = list(trio)

    # keys, alphabet, naming
    assert predictor_key("tvi_n") == "tvi_n"
    assert predictor_key(trio) == "ofi_L10_norm_n+micro_price+vpin"
    assert cfg.alphabet_size_for("tvi_n") == 16
    assert cfg.alphabet_size_for(trio) == 4 ** 4
    # verbatim LearningKraus_multivariate naming: predicted-first-last
    assert cfg.seq_distr_name(trio) == (
        "SEQ_DISTR_NVDA_multivariate_log_mid-ofi_L10_norm_n-vpin_202504")
    assert cfg.seq_distr_name("tvi_n") == (
        "SEQ_DISTR_NVDA_bivariate_log_mid-tvi_n_202504")

    # validation: list training.predictor accepted iff listed; duplicate
    # variables in the joint encoding rejected
    assert cfg.validate() == [], cfg.validate()
    cfg.training.predictor = ["micro_price", "vpin"]
    assert any("training.predictor" in p for p in cfg.validate())
    cfg.training.predictor = list(trio)
    cfg.distributions.predictors = ["tvi_n", ["log_mid", "vpin"]]
    assert any("duplicate variables" in p.lower() for p in cfg.validate())

    # yaml roundtrip with nested predictors
    cfg.distributions.predictors = ["tvi_n", trio]
    with tempfile.TemporaryDirectory() as d:
        cfg2 = RunConfig.load(cfg.save(Path(d) / "cfg.yaml"))
    assert cfg2.distributions.predictors == ["tvi_n", trio]
    assert cfg2.training.predictor == trio
    print("  PASS multivariate predictor (keys/alphabet/naming/validation)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for fn in [test_yaml_roundtrip, test_unknown_key_rejected, test_validate,
               test_cache_key, test_parse_hhmm,
               test_legacy_imports_side_effect_free, test_model_registry,
               test_multivariate_predictor]:
        fn()
    print("ALL UNIT TESTS PASSED")
