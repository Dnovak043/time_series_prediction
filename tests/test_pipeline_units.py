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


def test_asset_paths():
    catalog = {"NVDA": "data/NVDA_INTC", "INTC": "data/NVDA_INTC",
               "AAPL": "data/AAPL", "IBM": "data/IBM"}

    # resolution: empty mapping and unlisted symbols fall back to data_path
    cfg = RunConfig()
    assert cfg.data.resolved_data_path() == cfg.data.data_path
    cfg.data.asset_paths = dict(catalog)
    cfg.data.symbol = "AAPL"
    assert cfg.data.resolved_data_path() == "data/AAPL"
    cfg.data.symbol = "INTC"
    assert cfg.data.resolved_data_path() == "data/NVDA_INTC"
    cfg.data.symbol = "TSLA"                        # not in the catalog
    assert cfg.data.resolved_data_path() == cfg.data.data_path
    assert cfg.validate() == [], cfg.validate()
    cfg.data.asset_paths = {"NVDA": 3}
    assert any("asset_paths" in p for p in cfg.validate())

    # cache keys: legacy configs (no resolution) keep their existing entries;
    # a resolved symbol keys on its directory, since same-named files in
    # different directories hold different assets
    legacy = RunConfig()
    legacy.data.symbol = "TSLA"
    legacy_key = DayFeatureCache(legacy.data, legacy.featurize).params_key()
    cfg = RunConfig()
    cfg.data.asset_paths = dict(catalog)
    cfg.data.symbol = "TSLA"                        # falls back -> same key
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() == legacy_key
    cfg.data.asset_paths["TSLA"] = "data/TSLA"      # resolves -> keyed on dir
    k_tsla = DayFeatureCache(cfg.data, cfg.featurize).params_key()
    assert k_tsla != legacy_key
    cfg.data.asset_paths["TSLA"] = "data/OTHER"     # different dir, new key
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() != k_tsla

    # yaml roundtrip with the mapping
    cfg = RunConfig()
    cfg.data.asset_paths = dict(catalog)
    with tempfile.TemporaryDirectory() as d:
        cfg2 = RunConfig.load(cfg.save(Path(d) / "cfg.yaml"))
    assert cfg2.data.asset_paths == catalog

    # the checked-in default config carries the catalog and validates
    cfg3 = RunConfig.load(Path(__file__).resolve().parent.parent
                          / "configs" / "default.yaml")
    assert cfg3.data.asset_paths == catalog
    assert cfg3.validate() == [], cfg3.validate()

    # control-panel round trip of a dict field
    from pipeline.ui import _make_widget, _read_widget
    from pipeline.config import field_info
    info = [i for i in field_info(cfg.data) if i["name"] == "asset_paths"][0]
    w = _make_widget(info)
    assert _read_widget(w, {}) == catalog
    w.value = ""
    assert _read_widget(w, {}) == {}
    print("  PASS asset paths (resolution/cache key/yaml/default.yaml/ui)")


def test_no_dead_knobs():
    """Every config field must actually be read by the code.

    Guards the project's no-invisible-parameters rule from its inverse: a
    knob that is exposed in YAML/UI/CLI but that nothing consumes, so
    changing it silently does nothing. Two of these have already shipped
    (training.num_workers, which LearningKraus.train() ignored until
    [vendoring fix 1], and the notebook's PREDICTED).
    """
    import glob
    import re
    from dataclasses import fields as dc_fields

    from pipeline.config import STAGES

    root = Path(__file__).resolve().parent.parent
    sources = []
    for pattern in ("pipeline/*.py", "scripts/*.py"):
        for path in glob.glob(str(root / pattern)):
            if path.endswith("config.py"):
                continue          # declaration site, not a use
            sources.append(Path(path).read_text())
    blob = "\n".join(sources)

    dead = []
    for stage, cls in STAGES.items():
        for f in dc_fields(cls):
            # attribute access (cfg.training.epochs) or getattr/string form
            # (getattr(d, "instrument_filter", False))
            if not (re.search(rf"\.{f.name}\b", blob)
                    or re.search(rf'["\']{f.name}["\']', blob)):
                dead.append(f"{stage}.{f.name}")
    assert not dead, f"config fields nothing reads: {dead}"
    n = sum(len(dc_fields(c)) for c in STAGES.values())
    print(f"  PASS no dead knobs ({n} config fields, all consumed)")


def test_training_knobs_reach_the_trainer():
    """The knobs added when training was consolidated must be wired, not
    just declared — models.py should read each of them."""
    root = Path(__file__).resolve().parent.parent
    models_src = (root / "pipeline" / "models.py").read_text()
    for field in ("seed", "num_workers", "eval_batch_size",
                  "plot_entries", "plot_dpi"):
        assert f"t.{field}" in models_src, f"models.py never reads t.{field}"

    # and LearningKraus.train() must actually honour num_workers
    # ([vendoring fix 1]) rather than hardcoding it
    lk_src = (root / "LearningKraus.py").read_text()
    train_body = lk_src[lk_src.index("def train("):]
    loader = train_body[train_body.index("DataLoader("):]
    # delimit on collate_fn rather than the first ')' — the [vendoring fix 1]
    # comment inside the call contains "main()'s"
    loader = loader[:loader.index("collate_fn")]
    # drop comment lines: the [vendoring fix 1] note quotes both the old
    # "num_workers=0" and "main()'s", which would confuse either check
    code = "\n".join(ln for ln in loader.splitlines()
                     if not ln.strip().startswith("#"))
    assert "num_workers=num_workers" in code, (
        "LearningKraus.train() DataLoader is not honouring num_workers")
    assert "num_workers=0" not in code, (
        "LearningKraus.train() DataLoader still hardcodes num_workers=0")
    print("  PASS training knobs reach the trainer (incl. vendoring fix 1)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for fn in [test_yaml_roundtrip, test_unknown_key_rejected, test_validate,
               test_cache_key, test_parse_hhmm,
               test_legacy_imports_side_effect_free, test_model_registry,
               test_multivariate_predictor, test_asset_paths,
               test_no_dead_knobs, test_training_knobs_reach_the_trainer]:
        fn()
    print("ALL UNIT TESTS PASSED")
